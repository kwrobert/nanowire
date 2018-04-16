import sys
import os
import six
import hashlib
import logging
import itertools
import tempfile as tmp
import numpy as np
from itertools import accumulate, repeat, chain, product
from collections import Iterable, OrderedDict
from contextlib import contextmanager
from scipy import interpolate
from scipy import constants
import scipy.integrate as intg


def is_iterable(arg):
    """
    Returns True if object is an iterable and is not a string, false otherwise
    """
    return isinstance(arg, Iterable) and not isinstance(arg, six.string_types)


def setup_sim(sim):
    """
    Convenience function for setting up a simulation for execution. Just calls
    a bunch of methods of the Simulator object
    """

    sim.evaluate_config()
    sim.update_id()
    sim.make_logger()
    sim.make_coord_arrays()
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
    if path is None:
        return (1, 0)
    # Get data
    path = os.path.expandvars(path)
    freq_vec, n_vec, k_vec = np.loadtxt(path, unpack=True)
    # Get n and k at specified frequency via interpolation
    f_n = interpolate.interp1d(freq_vec, n_vec, kind='linear',
                               bounds_error=False,
                               fill_value='extrapolate')
    f_k = interpolate.interp1d(freq_vec, k_vec, kind='linear',
                               bounds_error=False,
                               fill_value='extrapolate')
    return f_n(freq), f_k(freq)


def get_incident_power(sim):
    """
    Returns the incident power per square meter for a simulation depending on
    frequency and the incident polar angle. Uses the spectrum provided in the
    config file for all computations. Power is multiplied by a factor of
    cos(polar_angle)

    Each simulation is conducted at a single frequency :math:`f =  \\omega
    / 2\\pi` which is associated with a frequency "bin" of spectral width
    :math:`\\Delta f`. The solar spectrum is expressed in units of Watts *
    m^-2 * Hz^-1. In order to compute the incident power/area for this
    simulation, we have a few options

    1. Interpolate to find the spectral irradiance at the frequency for
       this simulation, then multiply by the spectral width
    2. Find all available irradiance values contained inside the frequency
       bin, then integrate over the bin using those values as data points.
       The bin extends from :math:`(f - \\Delta f/2, f + \\Delta f/2)`, so
       in summary

       .. math:: \\int_{f - \\Delta f/2}^{f + \\Delta f/2} I(f) df

       where :math:`I` is the incident solar irradiance.

    Method 2 is used in this function, because it is debatably more
    accurate.

    :param sim: The simulation to compute the incident power for
    :type sim: Simulator or Simulation
    :raises ValueError: If the maximum or minimum bin edge extend beyond
                        the data range in the provided spectral data
    """

    freq = sim.conf['Simulation']['params']['frequency']
    polar_angle = sim.conf['Simulation']['params']['polar_angle']
    path = os.path.expandvars(sim.conf['Simulation']['input_power'])
    bin_size = sim.conf['Simulation']['params']['bandwidth']
    # Get NREL AM1.5 data
    freq_vec, p_vec = np.loadtxt(path, unpack=True, delimiter=',')
    # Get all available power values within this bin
    left = freq - bin_size / 2.0
    right = freq + bin_size / 2.0
    inds = np.where((left < freq_vec) & (freq_vec < right))[0]
    # Check for edge cases
    if len(inds) == 0:
        # It is unphysical to claim that an input wave of a single
        # frequency can contain any power. If we're simulating at a single
        # frequency, just assume the wave has the power contained within
        # the NREL bin surrounding that frequency
        sim.log.warning('Your bins are smaller than NRELs! Using NREL'
                        ' bin size')
        closest_ind = np.argmin(np.abs(freq_vec - freq))
        # Is the closest one to the left or the right?
        if freq_vec[closest_ind] > freq:
            other_ind = closest_ind - 1
            left = freq_vec[other_ind]
            left_power = p_vec[other_ind]
            right = freq_vec[closest_ind]
            right_power = p_vec[closest_ind]
        else:
            other_ind = closest_ind + 1
            right = freq_vec[other_ind]
            right_power = p_vec[other_ind]
            left = freq_vec[closest_ind]
            left_power = p_vec[closest_ind]
    elif inds[0] == 0:
        raise ValueError('Your leftmost bin edge lies outside the'
                         ' range provided by NREL')
    elif inds[-1] == len(freq_vec):
        raise ValueError('Your rightmost bin edge lies outside the'
                         ' range provided by NREL')
    else:
        # A simple linear interpolation given two pairs of data points, and the
        # desired x point
        def lin_interp(x1, x2, y1, y2, x):
            return ((y2 - y1) / (x2 - x1)) * (x - x2) + y2
        # If the left or right edge lies between NREL data points, we do a
        # linear interpolation to get the irradiance values at the bin edges.
        # If the left of right edge happens to be directly on an NREL bin edge
        # (unlikely) the linear interpolation will just return the value at the
        # NREL bin. Also the selection of inds above excluded the case of left
        # or right being equal to an NREL bin,
        left_power = lin_interp(freq_vec[inds[0] - 1], freq_vec[inds[0]],
                                p_vec[inds[0] - 1], p_vec[inds[0]], left)
        right_power = lin_interp(freq_vec[inds[-1]], freq_vec[inds[-1] + 1],
                                 p_vec[inds[-1]], p_vec[inds[-1] + 1], right)
    # All the frequency values within the bin and including the bin edges
    freqs = [left]+list(freq_vec[inds])+[right]
    # All the power values
    power_values = [left_power]+list(p_vec[inds])+[right_power]
    sim.log.info('Frequency points in bin: %s', str(freqs))
    sim.log.info('Power values in bin: %s', str(power_values))
    # Just use a trapezoidal method to integrate the spectrum and multiply by
    # angular factor
    power = intg.trapz(power_values, x=freqs)*np.cos(polar_angle)
    sim.log.info('Incident Power/area: %s', str(power))
    return power


def get_incident_amplitude(sim):
    """
    Computes the amplitude of the incident electric field for a simulation.
    This just calls :py:func:`get_incident_power` and then multiplies its
    output by some unitful factors

    This calculation for the amplitude comes from the definition of the
    Poynting vector in free space (taken from Jackson 3rd Ed. pg. 298) :math:`S
    = .5*\\sqrt{\\epsilon_0 / \\mu_0} | E_o |^2` where E_o is the amplitude of
    the plane wave and is not time averaged in any way, and S is the magnitude
    of the Poynting vector.
    """

    power_per_area = get_incident_power(sim)
    period = sim.conf['Simulation']['params']['array_period']
    area = period*period
    power = power_per_area*area
    sim.log.info('Incident Power: %s', str(power))
    # We need to reduce amplitude of the incident wave depending on
    #  incident polar angle
    # E = np.sqrt(2*constants.c*constants.mu_0*intensity)*np.cos(polar_angle)
    E = np.sqrt(2 * constants.c * constants.mu_0 * power)
    sim.log.info('Incident Amplitude: %s', str(E))
    return E


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


def make_hash(o, hash_dict=None, hasher=None):
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
        if k == 'General' or k == 'Postprocessing':
            continue
        else:
            ret = make_hash(v, hasher=hasher).encode('utf-8')
            hasher.update(ret)
    return hasher.hexdigest()


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
