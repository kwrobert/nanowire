import sys
import os
import hashlib
import logging
import itertools
import tempfile as tmp
import numpy as np

from collections import OrderedDict
from contextlib import contextmanager
from scipy import interpolate


def setup_sim(sim):
    """
    Convenience function for setting up a simulation for execution. Just calls
    a bunch of methods of the Simulator object
    """

    sim.evaluate_config()
    sim.update_id()
    sim.make_logger()
    sim.configure()
    sim.build_device()
    sim.set_excitation()
    return sim


def get_nk(path, freq):
    """
    Returns n and k, the real and imaginary components of the index of
    refraction at a given frequency :param str path: A path to a text file
    containing the n and k data. The first column should be the frequency
    in Hertz, the second column the n values, and the third column the k
    values. Columns must be delimited by whitespace.  :param float freq:
    The desired frequency in Hertz
    """

    # Get data
    freq_vec, n_vec, k_vec = np.loadtxt(path, unpack=True)
    # Get n and k at specified frequency via interpolation
    f_n = interpolate.interp1d(freq_vec, n_vec, kind='linear',
                               bounds_error=False,
                               fill_value='extrapolate')
    f_k = interpolate.interp1d(freq_vec, k_vec, kind='linear',
                               bounds_error=False,
                               fill_value='extrapolate')
    return f_n(freq), f_k(freq)


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
    bin_size = 0
    for keyset in keysets:
        par = '.'.join(keyset)
        pdict = conf[keyset]
        # Force to float in case we did some interpolation in the config
        start, end, step = map(
            float, [pdict['start'], pdict['end'], pdict['step']])
        if pdict['itertype'] == 'numsteps':
            values = np.linspace(start, end, step)
            # We need to add the size of the bin to each sim config so we can
            # use it to average the total power contained within each bin
            # when computer incident amplitude/power
            if 'frequency' in keyset:
                bin_size = values[1] - values[0]
        elif pdict['itertype'] == 'stepsize':
            values = np.arange(start, end + step, step)
            if 'frequency' in keyset:
                bin_size = float(step)
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
    combos = list(itertools.product(*valuelist))
    # log.debug('The list of parameter combos: %s', str(combos))
    # Gotta map to float cuz yaml writer doesn't like numpy data types
    return keys, combos, float(bin_size)


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

class StreamToLogger(object):
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
   
    # # This will log any uncaught exceptions
    # def handle_exception(exc_type, exc_value, exc_traceback):
    #     if issubclass(exc_type, KeyboardInterrupt):
    #         sys.__excepthook__(exc_type, exc_value, exc_traceback)
    #         return

    #     logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # sys.excepthook = handle_exception
    return logger


def make_hash(o):
    """
    A recursive function for hasing any python built-in data type. Probably
    won't work on custom objects. It is consistent across runs and handles
    arbitrary nesting. Mainly intended for computing the hash of config
    dictionarys to establish a unique ID. Right now it ignores any settings in
    the General section because those aren't really important when it comes to
    differentiating simulations
    """

    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        buf = repr(o).encode('utf-8')
        return hashlib.md5(buf).hexdigest()
    new_o = OrderedDict()
    for k, v in sorted(o.items(),key=lambda tup: tup[0]):
        if k == 'General':
            continue
        else:
            new_o[k] = make_hash(v)
    out = repr(tuple(frozenset(sorted(new_o.items())))).encode('utf-8')
    return hashlib.md5(out).hexdigest()


def cmp_dicts(d1, d2):
    """Recursively compares two dictionaries"""
    # First test the keys
    for k1 in d1.keys():
        if k1 not in d2:
            return False
    for k2 in d2.keys():
        if k2 not in d1:
            return False
    # Now we need to test the contents recursively. We store the results of
    # each recursive comparison in a list and assert that they all must be True
    # at the end
    comps = []
    for k1, v1 in d1.items():
        v2 = d2[k1]
        if isinstance(v1, dict) and isinstance(v2, dict):
            comps.append(cmp_dicts(v1, v2))
        else:
            if v1 != v2:
                return False
    return all(comps)
