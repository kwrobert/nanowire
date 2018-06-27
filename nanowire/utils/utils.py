import os
import six
import sys
import posixpath
import hashlib
import logging
import netifaces
import traceback
import copy
import warnings
import time
import psutil
import pint
import tempfile as tmp
import numpy as np
import tables as tb
from functools import wraps
from line_profiler import LineProfiler
from contextlib import contextmanager
from itertools import accumulate, repeat, chain, product
from collections import Iterable, OrderedDict, MutableMapping

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

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
                return False
        for k2 in d2.keys():
            this_path = posixpath.join(keypath, k1)
            if this_path in skip_keys:
                continue
            if k2 not in d1:
                return False
        for k1, v1 in d1.items():
            v2 = d2[k1]
            this_path = posixpath.join(keypath, k1)
            if this_path in skip_keys:
                continue
            if isinstance(v1, MutableMapping) and isinstance(v2, MutableMapping):
                ret = _cmp_dicts(v1, v2, skip_keys=skip_keys, keypath=this_path)
                if not ret:
                    return False
            else:
                if v1 != v2:
                    return False
        return True
    return _cmp_dicts(d1, d2, skip_keys=skip_keys)


def find_lists(o, keypath=[], list_locs=None, lists=None):
    """
    Find all lists in a dictionary recursively
    """

    list_locs = list_locs if list_locs is not None else []
    lists = lists if lists is not None else []
    if isinstance(o, dict):
        for key in o.keys():
            loc = keypath + [key]
            val = o[key]
            if isinstance(val, list):
                list_locs.append(loc)
                lists.append(val)
            elif isinstance(val, dict):
                find_lists(val, keypath=loc, list_locs=list_locs, lists=lists)
            else:
                continue
    return list_locs, lists


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


def flatten(d, skip_branch=None, sep='/'):
    """
    Returned a flattened copy of the nested data structure.

    The keys will be the full path to the item's location in the nested
    data structure separated using the optional sep kwarg, and the value
    will be the value at that location

    Parameters
    ----------
    sep : str, default '/'
        The separating character used for all paths
    skip_branch : list, optional
        A list of `sep` separated paths to branches in the config that you
        would like to skip when flattening. This can be leaves in the
        Config or entire branches

    Returns
    -------
    dict
        A dict whose keys are `sep` separated strings representing the full
        path to the value in the originally nested dictionary. The values
        are the same as those from the original dictionary.

    Examples
    --------
    >>> d = {'a': {'b': 1, 'c': 2})
    >>> flatten(d)
    {'a/b': 1, 'a/c': 2}
    >>> flatten(d, sep='_')
    {'a_b': 1, 'a_c': 2}
    >>> flatten(d, sep='_', skip_branch='a_c')
    {'a_b': 1}
    """
    def _flatten(d, flattened=None, keypath='', skip_branch=None, sep='/'):
        """
        Internal, recursive flattening implementation
        """
        flattened = flattened if flattened is not None else {}
        skip_branch = skip_branch if skip_branch is not None else []
        for key, val in d.items():
            newpath = sep.join([keypath, key]).strip(sep)
            if keypath in skip_branch:
                continue
            if isinstance(val, dict):
                _flatten(val, flattened=flattened,
                         keypath=newpath, skip_branch=skip_branch,
                         sep=sep)
            else:
                flattened[newpath] = val
        return flattened
    return _flatten(d, skip_branch=skip_branch, sep=sep)


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
        for interface in netifaces.interfaces():
            config = netifaces.ifaddresses(interface)
             #AF_INET is not always present
            if netifaces.AF_INET in config.keys():
                for link in config[netifaces.AF_INET]:
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
    meta = {}
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
            fields[key], meta[key] = get_pytables_desc(value, keypath=newpath)
        # elif isinstance(value, list):
        #     # Make sure all elements in list have same type
        #     first_type = type(value[0])
        #     if not all(type(el) == first_type for el in value[1:]):
        #         raise TypeError('Can only store homogenous lists in table')
        #     value = np.array(value)
        #     dtype = np.dtype('{}{}'.format(value.shape, value.dtype))
        elif isinstance(value, pint.quantity._Quantity):
            arr = np.array(value.magnitude)
            fields[key] = tb.Col.from_dtype(arr.dtype)
            meta[key] = {'units': str(value.units)}
        else:
            value = np.array(value)
            dtype = np.dtype('{}{}'.format(value.shape, value.dtype))
            # PyTables gets upset by numpy arrays containing unicode strings,
            # but will raise an error if it encounters a unicode string it
            # cannot safely cast to ascii anyway. We should never have any such
            # strings
            # see http://www.pytables.org/MIGRATING_TO_3.x.html
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fields[key] = tb.Col.from_dtype(dtype)
    return fields, meta


def _recurse_row(row, base, data):
    for (key, value) in data.items():
        new = base + key
        if isinstance(value, dict):
            _recurse_row(row, new + '/', value)
        # elif isinstance(value, list) and type(value[0]) == str:
        #     value = [el.encode('utf-8') for el in value]
        #     row[new] = np.array(value)
        elif isinstance(value, pint.quantity._Quantity):
            row[new] = value.magnitude
        else:
            row[new] = value


def add_row(tbl, data):
    """Add a new row to a table based on the contents of a dict.
    """
    row = tbl.row
    for (key, value) in data.items():
        if isinstance(value, MutableMapping):
            _recurse_row(row, key + '/', value)
        # elif isinstance(value, list) and type(value[0]) == str:
        #     value = [el.encode('utf-8') for el in value]
        #     row[key] = np.array(value)
        elif isinstance(value, pint.quantity._Quantity):
            row[key] = value.magnitude
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
        elif isinstance(v, np.ndarray) and isinstance(v[0], np.bytes_):
            ret[k] = [el.decode() for el in v]
        elif isinstance(v, np.ndarray):
            ret[k] = list(v)
        else:
            msg = "numpy_arr_to_dict did not handle value {} of " \
                  "type {}".format(v, type(v))
            warnings.warn(msg, RuntimeWarning)
            ret[k] = v
    return ret




def get_processes_using_file(path):
    """
    Get the process using the file at `path`, if any.

    path : str
        Path to an existing file.

    Notes
    -----

    The information provided by this function is time sensitive. It is possible
    that, by the time the function returns with a list of processes, some of
    those processes might have already been terminated and therefore no longer
    hold an open file descriptor for the file located at `path`. Or, even
    worse, some new process with the same PID has taken it's place.

    Returns
    -------

    :py:class:`psutil.Process` or None
        Returns a :py:class:`psutil.Process` if a process exists and has an
        open file handle to `path`, otherwise returns None.
    """
    procs = []
    for p in psutil.process_iter(attrs=['name', 'open_files']):
        try:
            for f in p.info['open_files'] or []:
                same = os.path.samefile(path, f.path)
                if same:
                    msg = 'Proc {} has file {}'.format(p.name(), path)
                    print(msg)
                    procs.append(p)
                    # return p
        # This catches a race condition where a process ends
        # before we can examine its files
        except psutil.NoSuchProcess as err:
            print("*** Examined process terminated")
    return procs


def wait_until_released(path):
    """
    Wait until there are no processes with a open file descriptor for the file
    at `path`

    path : str
        Path to an existing file.

    Notes
    -----

    This could potentially run forever if there is some other process that
    refuses to release the file
    """
    file_is_open = True
    print('Waiting on path {}'.format(path))
    while file_is_open:
        file_is_open = False
        for p in psutil.process_iter(attrs=['name', 'open_files']):
            open_paths = [f.path for f in p.info['open_files'] or []]
            if path in open_paths:
                print('Path {} open'.format(path))
                file_is_open = True
                break
                # same = os.path.samefile(path, f.path)
                # if same:
                #     file_open = True
        time.sleep(3)
    print('Path {} released!'.format(path))
