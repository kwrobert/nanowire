import os
import logging
import pint
import numpy as np
import scipy.integrate as intg
from scipy import interpolate
from nanowire.utils.utils import (
    ureg,
    Q_
)
pint.set_application_registry(ureg)


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


@ureg.wraps((None, None), (None, ureg.hertz))
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


@ureg.with_context('spectroscopy')
@ureg.wraps(ureg.watt / ureg.meter**2,
            (None, ureg.hertz, ureg.degrees, None, None))
def get_incident_power(spectrum, freq, polar_angle, bandwidth, logger=None):
    """
    Returns the incident power per area (W/m^2) for a simulation depending on
    frequency, the incident polar angle, the input spectrum, and the bandwidth
    of simulation. Power is multiplied by a factor of cos(polar_angle)

    Each simulation is conducted at a single frequency :math:`f =  \\omega
    / 2\\pi` which is associated with a frequency "bin" of spectral width
    :math:`\\Delta f`. The solar spectrum is expressed in units of Watts *
    m^-2 * nm^-1. To get the spectrum in W*m^-2*Hz^-1, we need to perform a
    quick conversion. We start by noting the the integral of the spectrum in
    either set of units must be the same

    .. math:: \\int I_{\\lambda}(\\lambda)d\\lambda = \\int I_{f}(f)df

    From here, we can see for this to be true the integrands must themselves be
    equal

    .. math:: I_{\\lambda}(\\lambda)d\\lambda = I_{f}(f)df

    Using the relation :math:`c = \\lambda f` we see :math:`d\\lambda =
    \\frac{c}{f^2}df` and we can insert this into the above relation to arrive
    at

    .. math:: I_{f}(f)df = \\frac{c}{f^2} I_{\\lambda}(\\lambda)

    Once we have the spectrum in the correct units, we can compute the incident
    power/area. We have a few options here

    1. Interpolate to find the spectral irradiance at the frequency for
       this simulation, then multiply by the spectral width
    2. Find all available irradiance values contained inside the frequency
       bin, then integrate over the bin using those values as data points.
       The bin extends from :math:`(f - \\Delta f/2, f + \\Delta f/2)`, so
       in summary

       .. math:: \\int_{f - \\Delta f/2}^{f + \\Delta f/2} I(f) df

       where :math:`I` is the incident solar irradiance. If a bin edge falls
       between two available data points, interpolation is used to estimate the
       irradiance value at the bin edge.

    Method 2 is used in this function, because it is debatably more
    accurate.

    :param sim: The simulation to compute the incident power for
    :type sim: Simulator or Simulation
    :raises ValueError: If the maximum or minimum bin edge extend beyond
                        the data range in the provided spectral data
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    # TODO: This feels a bit limiting. Maybe allow user to pass in an
    # arbitrary spectrum somehow
    allowed_spectra = {'am1.5g', 'am1.5d', 'am0'}
    if spectrum not in allowed_spectra:
        raise ValueError('Invalid spectrum. Must be one of '
                         '{}'.format(allowed_spectra))
    index = {'am1.5g': 2, 'am1.5d': 3, 'am0': 1}
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             '../../', 'spectra'))
    data_file = os.path.join(data_dir, 'ASTMG173.csv')
    wvs, power = np.loadtxt(data_file, delimiter=',',
                            usecols=[0, index[spectrum]], unpack=True)
    # Reverse the vectors so when we convert to frequency everything is in the
    # proper order
    wvs = Q_(wvs[::-1], 'nanometers')
    power = Q_(power[::-1], 'W*m^-2*nm^-1')
    freq_vec = wvs.to('hertz')
    # Convert the spectrum using the Laplacian to convert the spectrum from
    # nm^-1 to Hz^-1
    p_vec = power * ureg.speed_of_light / freq_vec ** 2
    freq_vec = freq_vec.magnitude
    p_vec = p_vec.to('watts*meter^-2*hertz^-1').magnitude
    # Get all available power values within this bin
    if isinstance(bandwidth, tuple):
        assert(all(isinstance(el, pint.quantity._Quantity) for el in
                   bandwidth))
        left = bandwidth[0].magnitude
        right = bandwidth[1].magnitude
    else:
        assert(bandwidth.units == 'hertz')
        left = freq - bandwidth / 2.0
        right = freq + bandwidth / 2.0
    inds = np.where((left < freq_vec) & (freq_vec < right))[0]
    # Check for edge cases
    if len(inds) == 0:
        # If we're simulating at a bandwidth narrower than the available NREL
        # spectral resolution just assume the wave has the power contained
        # within the NREL bin surrounding that frequency and log a warning
        logger.warning('Your bins are smaller than NRELs! Using NREL'
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
        # NREL bin.
        left_power = lin_interp(freq_vec[inds[0] - 1], freq_vec[inds[0]],
                                p_vec[inds[0] - 1], p_vec[inds[0]], left)
        right_power = lin_interp(freq_vec[inds[-1]], freq_vec[inds[-1] + 1],
                                 p_vec[inds[-1]], p_vec[inds[-1] + 1], right)
    # All the frequency values within the bin and including the bin edges
    freqs = [left]+list(freq_vec[inds])+[right]
    # All the power values
    power_values = [left_power]+list(p_vec[inds])+[right_power]
    logger.info('Frequency points in bin: %s', str(freqs))
    logger.info('Power values in bin: %s', str(power_values))
    # Just use a trapezoidal method to integrate the spectrum and reduce power
    # of the incident wave depending on incident polar angle
    power = intg.trapz(power_values, x=freqs) * np.cos(polar_angle)
    logger.info('Incident Power/area: %s', str(power))
    return power


# @ureg.wraps(ureg.volt / ureg.meter,
#             (None, ureg.hertz, ureg.degrees, ureg.hertz, None))
def get_incident_amplitude(spectrum, freq, polar_angle, bandwidth,
                           logger=None):
    """
    Computes the amplitude of the incident electric field for a simulation.
    This just calls :py:func:`get_incident_power` and then multiplies its
    output by some unitful factors

    This calculation for the amplitude comes from the definition of the
    Poynting vector in free space (taken from Jackson 3rd Ed. pg. 298) :math:`S
    = .5*\\sqrt{\\epsilon_0 / \\mu_0} | E_o |^2` where E_o is the amplitude of
    the plane wave and is not time averaged in any way, and S is the magnitude
    of the Poynting vector.

    Also see Griffifths, Introduction to Electrodynamics, pg 399 Eq 9.61
    """
    power_per_area = get_incident_power(spectrum, freq, polar_angle, bandwidth,
                                        logger=logger)
    logger.info('Incident Power/Area: %s', str(power_per_area))
    E = np.sqrt(2 * ureg.impedance_of_free_space * power_per_area)
    E = E.to(ureg.volts / ureg.meter)
    logger.info('Incident Amplitude: %s', str(E))
    return E
