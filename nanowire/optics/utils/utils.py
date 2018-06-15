import os
import numpy as np
import scipy.integrate as intg
from scipy import interpolate
from scipy import constants


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
    Returns the incident power per area (W/m^2) for a simulation depending on
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

    freq = sim.conf['Simulation']['frequency']
    polar_angle = sim.conf['Simulation']['polar_angle']
    path = os.path.expandvars(sim.conf['Simulation']['input_power'])
    bin_size = sim.conf['Simulation']['bandwidth']
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
    period = sim.conf['Simulation']['array_period']
    area = period*period
    power = power_per_area*area
    sim.log.info('Incident Power: %s', str(power))
    # We need to reduce amplitude of the incident wave depending on
    #  incident polar angle
    # E = np.sqrt(2*constants.c*constants.mu_0*intensity)*np.cos(polar_angle)
    E = np.sqrt(2 * constants.c * constants.mu_0 * power)
    sim.log.info('Incident Amplitude: %s', str(E))
    return E
