import os
import six
import sys
import posixpath
import hashlib
import logging
import netifaces
import traceback
import copy
import tempfile as tmp
import numpy as np
import tables as tb
from multiprocessing import Pool
from line_profiler import LineProfiler
from contextlib import contextmanager
from itertools import accumulate, repeat, chain, product
from collections import Iterable, OrderedDict, MutableMapping


def do_profile(follow=[], out=''):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                if out:
                    path = os.path.abspath(os.path.expandvars(out))
                    with open(path, 'a') as f:
                        profiler.print_stats(stream=f)
                else:
                    profiler.print_stats()
        return profiled_func
    return inner


def is_iterable(arg):
    """
    Returns True if object is an iterable and is not a string, false otherwise
    """
    return isinstance(arg, Iterable) and not isinstance(arg, six.string_types)


@contextmanager
def tempfile(suffix='', dir=None, npz=True):
    """ Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving, even in case of an exception.

    suffix : string
        optional file suffix
    dir : string
        optional directory to save temporary file in
    """

    tf = tmp.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)
    tf.file.close()
    try:
        yield tf.name
    finally:
        try:
            if npz:
                os.remove(tf.name)
                os.remove(tf.name+'.npz')
            else:
                os.remove(tf.name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise


@contextmanager
def open_atomic(filepath, npz=True):
    """Get a temporary file path in the same directory as filepath. The temp
    file is used as a placeholder for filepath to make atomic write operations
    possible. The file will not be moved to destination in case of an exception.

    filepath : string
        the actual filepath we wish to write to
    """
    with tempfile(npz=npz, dir=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        yield tmppath
        if npz:
            os.rename(tmppath+'.npz', filepath+'.npz')
        else:
            os.rename(tmppath, filepath)


class LogExceptions:
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            # Here we add some debugging help.
            log = logging.getLogger(__name__)
            log.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can clean up.
            # This kills the parent process if it calls .get or .wait on the
            # AsyncResult object returned by apply_async
            # raise
        # It was fine, give a normal answer
        return result


# class LoggingPool(Pool):
#     def apply_async(self, func, args=(), kwds={}, callback=None):
#         return Pool.apply_async(self, LogExceptions(func), args, kwds,
#                                 callback)


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)


class IdFilter(logging.Filter):
    """
    A filter to either only pass log records with a matching ID, or reject all
    log records with an ID attribute. This is configurable via a kwarg to the
    init method
    """

    def __init__(self, ID=None, name="", reject=False):
        super(IdFilter, self).__init__(name=name)
        self.ID = ID
        if reject:
            self.filter = self.reject_filter
        else:
            self.filter = self.pass_filter

    def pass_filter(self, record):
        if not hasattr(record, 'ID'):
            return 0
        if record.ID == self.ID:
            return 1
        else:
            return 0

    def reject_filter(self, record):
        if hasattr(record, 'ID'):
            return 0
        else:
            return 1


def configure_logger(level='info', name=None, console=False, logfile=None,
                     propagate=True):
    """
    Creates a logger providing some arguments to make it more configurable.

    :param str name:
        Name of logger to be created. Defaults to the root logger
    :param str level:
        The log level of the logger, defaults to INFO. One of: ['debug', 'info',
        'warning', 'error', 'critical']
    :param bool console:
        Add a stream handler to send messages to the console. Generally
        only necessary for the root logger.
    :param str logfile:
        Path to a file. If specified, will create a simple file handler and send
        messages to the specified file. The parent dirs to the location will
        be created automatically if they don't already exist.
    """

    # Get numeric level safely
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)
    # Set formatting
    formatter = logging.Formatter('%(asctime)s [%(module)s:%(name)s:%(levelname)s] - %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
    # Create logger
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger()
    if not propagate:
        logger.propagate = False
    logger.setLevel(numeric_level)
    if logfile:
        log_dir, logfile = os.path.split(os.path.expandvars(logfile))
        # Set up file handler
        try:
            os.makedirs(log_dir)
        except OSError:
            # Log dir already exists
            pass
        output_file = os.path.join(log_dir, logfile)
        fhandler = logging.FileHandler(output_file)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    # Create console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def get_combos(conf, keysets):
    """Given a config object and an iterable of the parameters you wish to find
    unique combinations of, return two lists. The first list contains the
    names of all the variable parameters in the config object. The second is a
    list of lists, where each inner list contains a unique combination of
    values of the parameters provided in keysets. Order is preserved such that
    the names in the first list corresponds to the values in the second list.
    For example, the returned lists would look like:

    list1 = [param_name1, param_name2, param_name3]
    list2 = [[val11, val12, val13], [val21, val22, val23], [val31, val32, val33]]
    """

    # log = logging.getLogger()
    # log.info("Constructing dictionary of options and their values ...")
    # Get the list of values from all our variable keysets
    optionValues = OrderedDict()
    for keyset in keysets:
        par = '.'.join(keyset)
        pdict = conf[keyset]
        if pdict['itertype'] == 'numsteps':
            # Force to float in case we did some interpolation in the config
            start, end, step = map(
                float, [pdict['start'], pdict['end'], pdict['step']])
            values = np.linspace(start, end, step)
        elif pdict['itertype'] == 'stepsize':
            # Force to float in case we did some interpolation in the config
            start, end, step = map(
                float, [pdict['start'], pdict['end'], pdict['step']])
            values = np.arange(start, end + step, step)
        elif pdict['itertype'] == 'list':
            values = pdict['value']
        else:
            raise ValueError(
                'Invalid itertype specified at {}'.format(str(keyset)))
        optionValues[par] = values
    # log.debug("Option values dict after processing: %s" % str(optionValues))
    valuelist = list(optionValues.values())
    keys = list(optionValues.keys())
    # Consuming a list of lists/tuples where each inner list/tuple contains all
    # the values for a particular parameter, returns a list of tuples
    # containing all the unique combos for that set of parameters
    combos = list(product(*valuelist))
    # log.debug('The list of parameter combos: %s', str(combos))
    # Gotta map to float cuz yaml writer doesn't like numpy data types
    return keys, combos



def filter_by_param(confs, pars):
    """
    Filters out Config objects from a list of objects given a dict of
    parameters that you wish to keep. Returns a list of the desired Config
    objects

    :param confs: A list of Config objects
    :type confs: list
    :params pars: A dict whose keys are dot separated paths to a config item
    and values are a list of possible values for that parameter. Any simulation
    whose parameter does not match any of the provided values is removed from
    the sims and sim_groups attribute :type pars: dict
    """

    assert(type(pars) == dict)
    for par, vals in pars.items():
        vals = [type(confs[0][par])(v) for v in vals]
        confs = [conf for conf in confs if conf[par] in vals]
    return confs


def make_hash(o, hash_dict=None, hasher=None,
              skip_keys=['General', 'Postprocessing']):
    """
    A recursive function for hasing any python built-in data type. Probably
    won't work on custom objects. It is consistent across runs and handles
    arbitrary nesting. Mainly intended for computing the hash of config
    dictionarys to establish a unique ID. Right now it ignores any settings in
    the General section because those aren't really important when it comes to
    differentiating simulations
    """

    if hasher is None:
        hasher = hashlib.md5()
    # If iterable but not a string, compute the hash of each element (recursing
    # if necessary and updating hasher as we go), and build a tuple containing
    # the hashes of each element.  Then, hash the string representation of that
    # tuple
    if is_iterable(o) and not isinstance(o, dict):
        out = repr(tuple([make_hash(e, hasher=hasher) for e in
                          sorted(o)])).encode('utf-8')
        hasher.update(out)
        return hasher.hexdigest()
    # If not a dict or an iterable, must be a float or string, so just hash it
    # and return
    elif not isinstance(o, dict):
        buf = repr(o).encode('utf-8')
        hasher.update(buf)
        return hasher.hexdigest()
    # If its a dict, recurse through the dictionary and update the hasher as we
    # go
    for k, v in sorted(o.items(), key=lambda tup: tup[0]):
        if k in skip_keys:
            continue
        else:
            ret = make_hash(v, hasher=hasher).encode('utf-8')
            hasher.update(ret)
    return hasher.hexdigest()


def cmp_dicts(d1, d2, skip_keys=[]):
    """
    Recursively compares two configs

    Parameters
    ----------
    d1, d2: dict
       The dictionaries to compare
    skip_keys : list, optional
        A list of forward slash separated strings representing locations in the
        config that should be ignored when doing the comparison. The value of
        this key can differ between `d1` and `d2`, or be present in one dict
        but not the other, and this function will still return True

    Returns
    -------
    bool
        Boolean indicating whether or not the dicts have the same contents and
        keys
    """
    def _cmp_dicts(d1, d2, skip_keys=[], keypath=''):
        for k1 in d1.keys():
            this_path = posixpath.join(keypath, k1)
            if this_path in skip_keys:
                continue
            if k1 not in d2:
                # print('{} not in d2'.format(k1))
                return False
        for k2 in d2.keys():
            this_path = posixpath.join(keypath, k1)
            if this_path in skip_keys:
                continue
            if k2 not in d1:
                # print('{} not in d1'.format(k2))
                return False
        for k1, v1 in d1.items():
            v2 = d2[k1]
            this_path = posixpath.join(keypath, k1)
            if this_path in skip_keys:
                continue
            if isinstance(v1, MutableMapping) and isinstance(v2, MutableMapping):
                ret = _cmp_dicts(v1, v2, skip_keys=skip_keys, keypath=this_path)
                # print("ret = {}".format(ret))
                if not ret:
                    return False
            else:
                if v1 != v2:
                    # print('{} != {}'.format(v1, v2))
                    return False
        return True
    return _cmp_dicts(d1, d2, skip_keys=skip_keys)


def find_keypaths(d, key):
    """
    Find all ocurrences of the key `key` in dictionary `d` recursively

    Parameters
    ----------
    d : dict
        The dictionary in which to search
    key : str
        The key to search for. If `key` exists anywhere in the path to a
        item in the dict, that path is added to the list

    Returns
    -------
    list
        A list of forward slash separated paths containing the key `key` in the
        dictionary
    """

    def _find_paths(d, key, paths=None, keypath=''):
        paths = paths if paths is not None else []
        for k, v in d.items():
            this_path = posixpath.join(keypath, k)
            if key in this_path:
                paths.append(this_path)
            if isinstance(v, MutableMapping):
                _find_paths(v, key, paths=paths, keypath=this_path)
        return paths
    return _find_paths(d, key)


def find_inds(a, b, unique=False):
    """
    Get the indices where we can find the elements of the array a in the array
    b
    """
    return np.where(np.isin(b, a, assume_unique=unique))


def merge_and_sort(a, b, kind='mergesort'):
    """
    Merge arrays a and b, sort them, and return the unique elements
    """
    c = np.concatenate((a, b))
    c.sort(kind=kind)
    flag = np.ones(len(c), dtype=bool)
    np.not_equal(c[1:], c[:-1], out=flag[1:])
    return c[flag]


def cartesian_product(arrays, out=None):
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *accumulate(chain((arr,), repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)


def arithmetic_arange(a, b, dx0, d, endpoint=True):
    """
    A numpy.arange copy whose step sizes increase arithmetically, i.e the step
    sizes follow an arithmetic progression and increase additively by some
    constant `d`.

    :param a: The starting value
    :param b: The ending value
    :param dx0: The size of the very first step
    :param d: The amount to increase each step by.
    :keyword endpoint: Whether or not to include the end point b in the
    returned

    The stepsizes are calculated using the following formula

    :math:`dx_n = dx_{n-1} + n*d
    array.

    .. note::
        The size of the step between the 2nd to last point and the endpoint may
        not follow the arithmetic progression. This is because it may not be
        possible to land exactly on the given start and end point for the
        given initial step size and step increase
    """
    cur_step = dx0
    pts = np.array([a])
    i = 0
    while pts[-1] < b:
        step = cur_step + i*d
        pts = np.append(pts, pts[-1]+step)
        i += 1
        cur_step = step
    if endpoint:
        if pts[-1] > b:
            pts[-1] = b
    if not endpoint and pts[-1] >= b:
        pts = pts[0:-1]
    return pts

def ipv4():
    """
    Get all IPv4 addresses for all interfaces.
    """

    try:
        # to not take into account loopback addresses (no interest here)
        addresses = []
        for interface in interfaces():
            config = ifaddresses(interface)
             #AF_INET is not always present
            if AF_INET in config.keys():
                for link in config[AF_INET]:
                # loopback holds a 'peer' instead of a 'broadcast' address
                    if 'addr' in link.keys() and 'peer' not in link.keys():
                        addresses.append(link['addr'])
        return addresses
    except ImportError:
        return []

def get_public_iface():
    """
    Get the public facing network interface this machine uses to connect to the
    internet. The approach is to look at your default route/default interface
    for routing traffic, then get the interface name from that
    """

    interface = netifaces.gateways()['default'][netifaces.AF_INET][1]
    return interface

def get_public_ip(iface, version=4):
    """
    Get the public facing IP address of the given interface
    """

    if not isinstance(iface, str):
        raise ValueError("Interface must be a string")
    if not isinstance(version, int):
        raise ValueError("Version must be an integer")
    addrs = netifaces.ifaddresses(iface)
    if version == 4:
        ipadd =  addrs[netifaces.AF_INET][0]['addr']
    elif version == 6:
        ipadd = addrs[netifaces.AF_INET6][0]['addr']
    else:
        raise ValueError("version must be either 4 or 6")
    return ipadd


def sorted_dict(adict, reverse=False):
    """
    Returns a sorted version of a dictionary sorted by the 'order' key.
    Used to sort layers, geometric shapes into their proper physical order.
    Can pass in a kwarg to reverse the order if desired
    """

    try:
        sorted_layers = OrderedDict(sorted(adict.items(),
                                           key=lambda tup: tup[1]['order'],
                                           reverse=reverse))
    except KeyError:
        raise KeyError('The dictionary you are attempting to sort must '
                       'itself contain a dictionary that has an "order" key')
    return sorted_layers


def get_numpy_dtype(data, skip_keys=[], keypath=''):
    """
    Given a dict, generate a nested numpy dtype

    :param skip_keys: A list of slash separated paths that should be skipped
    when constructing a nested dtype
    """

    # Normalize all the paths in skip_keys
    skip_keys = [posixpath.normpath(p.strip('/')) for p in skip_keys]
    fields = []
    for (key, value) in data.items():
        newpath = posixpath.join(keypath, key)
        if newpath in skip_keys:
            continue
        if isinstance(value, str):
            value += ' ' * (64 - (len(value) % 64))
            value = value.encode('utf-8')
        if value is None:
            value = bool(value)

        if isinstance(value, dict):
            fields.append((key, get_numpy_dtype(value, keypath=newpath)))
        else:
            value = np.array(value)
            fields.append((key, '%s%s' % (value.shape, value.dtype)))
    return np.dtype(fields)


def get_pytables_desc(data, skip_keys=[], keypath=''):
    """
    Given a dict, generate a dictionary of potentially nested PyTables Column
    objects

    :param skip_keys: A list of slash separated paths that should be skipped
    when constructing the description
    """

    # Normalize all the paths in skip_keys
    skip_keys = [posixpath.normpath(p.strip('/')) for p in skip_keys]
    fields = {}
    for (key, value) in data.items():
        newpath = posixpath.join(keypath, key)
        if newpath in skip_keys:
            continue
        if isinstance(value, str):
            value += ' ' * (64 - (len(value) % 64))
            value = value.encode('utf-8')
        if value is None:
            value = bool(value)

        if isinstance(value, dict):
            fields[key] = get_pytables_desc(value, keypath=newpath)
        else:
            value = np.array(value)
            fields[key] = tb.Col.from_dtype(value.dtype)
    return fields


def _recurse_row(row, base, data):
    for (key, value) in data.items():
        new = base + key
        if isinstance(value, dict):
            _recurse_row(row, new + '/', value)
        else:
            row[new] = value


def add_row(tbl, data):
    """Add a new row to a table based on the contents of a dict.
    """
    row = tbl.row
    for (key, value) in data.items():
        if isinstance(value, MutableMapping):
            _recurse_row(row, key + '/', value)
        else:
            row[key] = value
    row.append()
    tbl.flush()

def view_fields(a, names):
    """
    Create a view of a structured numpy array containing specified fields

    Creates and returns a view of a structured numpy array containing only
    fields in the given collection of field names. Because this function
    returns a _view_, no copying is done and modifying the returned view also
    modifies the original array.

    Parameters
    ----------

    a : np.recarray, np.void
        The original array you wish to construct a view of.
    names : iterable
        An iterable of strings which are the field names to keep.

    Returns
    -------
    type(a)
        A view of the array `a` (not a copy), which has the same type as `a`.
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b


def remove_fields(a, names):
    """
    Create a view of a structured numpy array without specified fields

    Creates and returns a view of a structured numpy array `a` without the
    fields in the `names`. Because this function returns a _view_, no copying
    is done and modifying the returned view also modifies the original array.

    Parameters
    ----------

    a : np.recarray, np.void
        The original array you wish to construct a view of.
    names : iterable
        An iterable of strings which are the field names to remove.

    Returns
    -------
    type(a)
        A view of the array `a` (not a copy), which has the same type as `a`,
        but without the fields contained in `names`
    """
    dt = a.dtype
    keep_names = [name for name in dt.names if name not in names]
    return view_fields(a, keep_names)


def numpy_arr_to_dict(arr):
    """
    Convert (potentially) nested numpy structured array to a dictionary
    containing only builtin Python types
    """
    ret = {}
    for k, v in zip(arr.dtype.names, arr):
        if isinstance(v, np.void):
            ret[k] = numpy_arr_to_dict(v)
        elif isinstance(v, (np.int_, np.int)):
            ret[k] = int(v)
        elif isinstance(v, (np.bool_, np.bool)):
            ret[k] = bool(v)
        elif isinstance(v, (np.float_, np.float)):
            ret[k] = float(v)
        elif isinstance(v, (np.bytes_, np.str_, np.str)):
            ret[k] = v.decode()
        else:
            print('Missed type!!')
            print(type(v))
            print(v)
            ret[k] = v
    return ret


def group_against(confs, key, sort_key=None, skip_keys=None):
    """
    Group configs against particular parameter.

    Group a list of config objects against a particular parameter. Within each
    group, the parameter specified by `key` will vary, and all other parameters
    will remain fixed. Useful for examining the affect of a single parameter on
    simulation outputs, and for generating line plots with the parameter
    specified by `key` on the x-axis.

    Parameters
    ----------
    confs : list
        A list of :py:class:`nanowire.preprocess.config.Config` objects
    key : str
        A key specifying which parameter simulations will be grouped against.
        Individual simulations within each group will be sorted in increasing
        order of this parameter, and all other parameters will remain constant
        within the group. Key can be a slash-separated string pointing to
        nested items in the config.
    sort_key : str, optional
        An optional key used to sort the order of the inner lists within the
        returned outer list of groups. Works because all parameters within each
        internal group are constant (excluding the parameter specified by
        `key`). The outer list of group lists will be sorted in increasing
        order of the specified sort_key.
    skip_keys : list
        A list of keys to skip when comparing two Configs.

    Returns
    -------
    list
        A list of lists. Each inner list is a group, sorted in increasing order
        of the parameter `key`. All other parameters in each group are
        constant. The outer list may optionally be sorted in increasing order
        of an additonal `sort_key`

    Notes
    -----
    The config options in the input list `confs` are not copied, and references
    to the original Config objects are stored in the returned data structure
    """

    # We need only need a shallow copy of the list containing all the Config
    # objects. We don't want to modify the orig list but we wish to share
    # the sim objects the two lists contain
    skip_keys = skip_keys if skip_keys is not None else []
    skip_keys.append(key)
    sim_confs = copy.copy(confs)
    sim_groups = [[sim_confs.pop()]]
    while sim_confs:
        conf = sim_confs.pop()
        # Loop through each group, checking if this sim belongs in the
        # group
        match = False
        for group in sim_groups:
            conf2 = group[0]
            params_same = cmp_dicts(conf, conf2, skip_keys=skip_keys)
            if params_same:
                group.append(conf)
                match = True
                break
        # If we didnt find a matching group, we need to create a new group
        # for this simulation
        if not match:
            sim_groups.append([conf])
    for group in sim_groups:
        # Sort the individual sims within a group in increasing order of
        # the parameter we are grouping against a
        group.sort(key=lambda aconf: aconf[key])
    # Sort the groups in increasing order of the provided sort key
    if sort_key:
        sim_groups.sort(key=lambda agroup: agroup[0][sort_key])
    return sim_groups
