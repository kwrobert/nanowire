from __future__ import print_function
import sys
import os
import time
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as intg
import scipy.constants as constants
import S4
import argparse as ap
#  import profilehooks as ph

from itertools import chain, product
from collections import OrderedDict
from utils import make_hash, configure_logger, get_combos
from config import Config


class Simulator():

    def __init__(self, conf):
        # conf.interpolate()
        # conf.evaluate()
        self.conf = conf
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        period = self.conf['Simulation']['params']['array_period']['value']
        self.id = make_hash(conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'], self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir
        self.s4 = S4.New(Lattice=((period, 0), (0, period)), NumBasis=numbasis)

    def __del__(self):
        """
        Need to make sure we close the file descriptor for the log handler
        attached to each simulator instance so we don't hit the file descriptor
        limit when running massive job campaigns. For some reason this doesn't
        happen automagically
        """
        # Sometimes we hit an error before the log object gets created and
        # assigned as an attribute. Without the try, except we would get an
        # attribute error which makes error messages confusing and useless
        try:
            for handler in self.log.handlers[:]:
                handler.close()
                self.log.removeHandler(handler)
        except AttributeError:
            pass

    def clean_sim(self):
        try:
            for handler in self.log.handlers[:]:
                handler.close()
                self.log.removeHandler(handler)
            del self.log
            del self.s4
        except AttributeError:
            pass

    def update_id(self):
        """Update sim id. Used after changes are made to the config"""
        self.id = make_hash(self.conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'], self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir

    def make_logger(self, log_level='info'):
        """Makes the logger for this simulation"""
        lfile = os.path.join(self.dir, 'sim.log')
        self.log = configure_logger(level=log_level, name=self.id[0:10],
                                    logfile=lfile, propagate=False)

    def configure(self):
        """Configure options for the RCWA solver"""
        self.s4.SetOptions(**self.conf['Solver'])

    def _get_epsilon(self, path):
        """Returns complex dielectric constant for a material by pulling in nk
        text file, interpolating, computing nk values at freq, and
        converting"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        # Get data
        freq_vec, n_vec, k_vec = np.loadtxt(path, unpack=True)
        # Get n and k at specified frequency via interpolation
        f_n = spi.interp1d(freq_vec, n_vec, kind='nearest',
                           bounds_error=False, fill_value='extrapolate')
        f_k = spi.interp1d(freq_vec, k_vec, kind='nearest',
                           bounds_error=False, fill_value='extrapolate')
        n, k = f_n(freq), f_k(freq)
        # Convert to dielectric constant
        # NOTE!!: This assumes the relative magnetic permability (mew) is 1
        epsilon_real = n**2 - k**2
        epsilon_imag = 2 * n * k
        epsilon = complex(epsilon_real, epsilon_imag)
        return epsilon

    def _get_incident_amplitude(self):
        """Returns the incident amplitude of a wave depending on frequency"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        polar_angle = self.conf['Simulation']['params']['polar_angle']['value']
        path = self.conf['Simulation']['input_power']
        bin_size = self.conf['Simulation']['params']['frequency']['bin_size']
        # Get NREL AM1.5 data
        freq_vec, p_vec = np.loadtxt(path, unpack=True, delimiter=',')
        # Get all available power values within this bin
        left = freq - bin_size / 2.0
        right = freq + bin_size / 2.0
        inds = np.where((left < freq_vec) & (freq_vec < right))[0]
        # Check for edge cases
        if len(inds) == 0:
            raise ValueError('Your bins are smaller than NRELs!')
        if inds[0] == 0:
            raise ValueError('Your leftmost bin edge lies outside the'
                             ' range provided by NREL')
        if inds[-1] == len(freq_vec):
            raise ValueError('Your rightmost bin edge lies outside the'
                             ' range provided by NREL')
        # A simple linear interpolation given two pairs of data points, and the
        # desired x point

        def lin_interp(x1, x2, y1, y2, x):
            return ((y2 - y1) / (x2 - x1)) * (x - x2) + y2
        # If the left or right edge lies between NREL data points, we do a
        # linear interpolation to get the irradiance values at the bin edges
        left_power = lin_interp(freq_vec[inds[0] - 1], freq_vec[inds[0]],
                                p_vec[inds[0] - 1], p_vec[inds[0]], left)
        right_power = lin_interp(freq_vec[inds[-1]], freq_vec[inds[-1] + 1],
                                 p_vec[inds[-1]], p_vec[inds[-1] + 1], right)
        # All the frequency values within the bin and including the bin edges
        freqs = [left]+list(freq_vec[inds])+[right]
        # All the power values
        power_values = [left_power]+list(p_vec[inds])+[right_power]
        # Just use a trapezoidal method to integrate the spectrum
        power = intg.trapz(power_values, x=freqs)
        self.log.info('Incident Power: {}'.format(power))
        # We need to reduce total incident power depending on incident polar
        # angle
        # E = np.sqrt(constants.c*constants.mu_0*f_p(freq))*np.cos(polar_angle)
        E = np.sqrt(constants.c * constants.mu_0 * power)
        return E

    def set_excitation(self):
        """Sets the exciting plane wave for the simulation"""
        f_phys = self.conf['Simulation']['params']['frequency']['value']
        self.log.info('Physical Frequency = %E' % f_phys)
        c_conv = constants.c / self.conf['Simulation']['base_unit']
        f_conv = f_phys / c_conv
        self.s4.SetFrequency(f_conv)
        E_mag = self._get_incident_amplitude()
        polar = self.conf['Simulation']['params']['polar_angle']['value']
        azimuth = self.conf['Simulation']['params']['azimuthal_angle']['value']
        # To define circularly polarized light, basically just stick a j
        # (imaginary number) in front of one of your components. The component
        # you choose to stick the j in front of is a matter of convention. In
        # S4, if the incident azimuthal angle is 0, p-polarization is along
        # x-axis. Here, we choose to make the y-component imaginary. The
        # handedness is determined both by the component you stick the j in
        # front of and the sign of the imaginary component. In our convention,
        # minus sign means rhcp, plus sign means lhcp. To be circularly
        # polarized, the magnitudes of the two components must be the same.
        # This means E-field vector rotates clockwise when observed from POV of
        # source. Left handed = counterclockwise.
        # TODO: This might not be properly generalized to handle
        # polarized light if the azimuth angle IS NOT 0. Might need some extra
        # factors of cos/sin of azimuth to gen proper projections onto x/y axes
        polarization = self.conf['Simulation']['polarization']
        if polarization == 'rhcp':
            # Right hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(0, -E_mag),
                                           pAmplitude=complex(E_mag, 0))
        elif polarization == 'lhcp':
            # Left hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(0, E_mag),
                                           pAmplitude=complex(E_mag, 0))
        elif polarization == 'lpx':
            # Linearly polarized along x axis (TM polarixation)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(0, 0),
                                           pAmplitude=complex(E_mag, 0))
        elif polarization == 'lpy':
            # Linearly polarized along y axis (TE polarization)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(E_mag, 0),
                                           pAmplitude=complex(0, 0))
        else:
            raise ValueError('Invalid polarization specification')

    def build_device(self):
        """Build the device geometry"""

        # First define all the materials
        for mat, mat_path in self.conf['Materials'].items():
            eps = self._get_epsilon(mat_path)
            self.s4.SetMaterial(Name=mat, Epsilon=eps)
        self.s4.SetMaterial(Name='vacuum', Epsilon=complex(1, 0))
        # We need to properly sort our layers because order DOES matter. Light
        # will be incident upon the first layer specified
        for layer, ldata in sorted(self.conf['Layers'].items(),
                                   key=lambda tup: tup[1]['order']):
            self.log.info('Building layer: %s' % layer)
            self.log.info('Layer Order %i' % ldata['order'])
            base_mat = ldata['base_material']
            layer_t = ldata['params']['thickness']['value']
            self.s4.AddLayer(Name=layer, Thickness=layer_t,
                             S4_Material=base_mat)
            if 'geometry' in ldata:
                self.log.info('Building geometry in layer: {}'.format(layer))
                for shape, sdata in sorted(ldata['geometry'].items(), key=lambda tup: tup[1]['order']):
                    self.log.info('Building object {} of type {} at order'
                                  ' {}'.format(shape, sdata['type'], sdata['order']))
                    shape_mat = sdata['material']
                    if sdata['type'] == 'circle':
                        rad = sdata['radius']
                        cent = sdata['center']
                        coord = (cent['x'], cent['y'])
                        self.s4.SetRegionCircle(S4_Layer=layer, S4_Material=shape_mat, Center=coord,
                                                Radius=rad)
                    else:
                        raise NotImplementedError(
                            'Shape %s is not yet implemented' % sdata['type'])

    def get_height(self):
        """Get the total height of the device"""

        height = 0
        for layer, ldata in self.conf['Layers'].items():
            layer_t = ldata['params']['thickness']['value']
            height += layer_t
        return height

    def set_lattice(self, period):
        """Updates the S4 simulation object with a new array period"""
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        self.s4 = S4.New(Lattice=((period, 0), (0, period)), NumBasis=numbasis)

    def set_basis(self, numbasis):
        """Updates the S4 simulation object with a new set of basis terms"""
        period = self.conf['Simulation']['params']['array_period']['value']
        self.s4 = S4.New(Lattice=((period, 0), (0, period)), NumBasis=numbasis)

    def update_thicknesses(self):
        """Updates all layer thicknesses without rebuilding the device. This
        allows reuse of any layer eigenmodes already computed and utilizes a
        fundamental efficiency of the RCWA solver"""
        for layer, ldata in self.conf['Layers'].items():
            thickness = ldata['params']['thickness']['value']
            self.s4.SetLayerThickness(S4_Layer=layer, Thickness=thickness)

    #  @ph.timecall
    def get_field(self):
        """Constructs and returns a 2D numpy array of the vector electric
        field. The field components are complex numbers"""
        self.log.info('Computing fields ...')
        x_samp = self.conf['Simulation']['x_samples']
        y_samp = self.conf['Simulation']['y_samples']
        z_samp = self.conf['Simulation']['z_samples']
        max_depth = self.conf['Simulation']['max_depth']
        if max_depth:
            self.log.info('Computing up to depth of {} '
                          'microns'.format(max_depth))
            dz = max_depth / z_samp
            zvec = np.arange(0, max_depth + dz, dz)
        else:
            self.log.info('Computing for entire device')
            height = self.get_height()
            zvec = np.arange(0, height + dz, dz)
        Ex = 0j*np.zeros((z_samp+1, x_samp, y_samp))
        Ey = 0j*np.zeros((z_samp+1, x_samp, y_samp))
        Ez = 0j*np.zeros((z_samp+1, x_samp, y_samp))
        for zcount, z in enumerate(zvec):
            E, H = self.s4.GetFieldsOnGrid(z=z, NumSamples=(x_samp, y_samp),
                                           Format='Array')
            for xcount, xval in enumerate(E):
                for ycount, yval in enumerate(xval):
                    Ex[zcount, xcount, ycount] = yval[0]
                    Ey[zcount, xcount, ycount] = yval[1]
                    Ez[zcount, xcount, ycount] = yval[2]
        self.log.info('Finished computing fields!')
        return Ex, Ey, Ez

    def save_field(self):
        """Saves the fields throughout the device to an npz file"""
        if self.conf['General']['adaptive_convergence']:
            ex, ey, ez, numbasis, conv = self.adaptive_convergence()
            if conv:
                out = os.path.join(self.dir, 'converged_at.txt')
            else:
                out = os.path.join(self.dir, 'not_converged_at.txt')
            self.log.info('Writing convergence file ...')
            with open(out, 'w') as outf:
                outf.write('{}\n'.format(numbasis))
        else:
            ex, ey, ez = self.get_field()
        out = os.path.join(self.dir, self.conf["General"]["base_name"] + '.E')
        # headers = ['x', 'y', 'z', 'Ex', 'Ey', 'Ez']
        headers = [OrderedDict([('Ex_real', 0), ('Ex_imag', 1), ('Ey_real', 2),
                               ('Ey_imag', 3), ('Ez_real', 4), ('Ez_imag', 5)])]
        data = {'Ex': ex, 'Ey': ey, 'Ez': ez}
        if self.conf['General']['save_as'] == 'text':
            np.savetxt(out, efield, header=','.join(headers))
        elif self.conf['General']['save_as'] == 'npz':
            # Compression adds a small amount of time. The time cost is
            # nonlinear in the file size, meaning the penalty gets larger as the
            # field grid gets finer. However, the storage gains are enormous!
            # Compression brought the file from 1.1G to 3.9M in a test case.
            # I think the compression ratio is so high because npz is a binary
            # format, and all compression algs benefit from large sections of
            # repeated characters
            np.savez_compressed(out, **data)
            #  np.savez(out,headers=headers,data=efield)
        else:
            raise ValueError('Invalid file type specified in config')

    def get_fluxes(self):
        """
        Get the fluxes at the top and bottom of each layer. This is a surface
        integral of the component of the Poynting flux perpendicular to this
        x-y plane of the interface, and have forward and backward componenets.
        Returns a dict where the keys are the layer name and the values are a
        length 2 tuple with the forward component first and the backward
        component second. The components are complex numbers
        """
        self.log.info('Computing fluxes ...')
        flux_dict = {}
        for layer, ldata in self.conf['Layers'].items():
            self.log.info('Computing fluxes through layer: %s' % layer)
            # This gets flux at top of layer
            forw, back = self.s4.GetPowerFlux(S4_Layer=layer)
            flux_dict[layer] = (forw, back)
            # This gets flux at the bottom
            offset = ldata['params']['thickness']['value']
            forw, back = self.s4.GetPowerFlux(S4_Layer=layer, zOffset=offset)
            key = layer + '_bottom'
            flux_dict[key] = (forw, back)
        self.log.info('Finished computing fluxes!')
        return flux_dict

    def save_fluxes(self):
        """Saves the fluxes at layer interfaces to a file"""
        path = os.path.join(self.dir, 'fluxes.dat')
        fluxes = self.get_fluxes()
        with open(path, 'w') as out:
            out.write(
                '# Layer,ForwardReal,BackwardReal,ForwardImag,BackwardImag\n')
            for layer, ldata in fluxes.items():
                forw, back = ldata
                row = '%s,%s,%s,%s,%s\n' % (layer, forw.real, back.real,
                                            forw.imag, back.imag)
                out.write(row)

    def calc_diff(self, fields1, fields2, exclude=False):
        """Calculate the percent difference between two vector fields"""
        # This list contains three 3D arrays corresponding to the x,y,z
        # componenets of the e field. Within each 3D array is the complex
        # magnitude of the difference between the two field arrays at each
        # spatial point squared
        diffs_sq = [np.absolute(arr1 - arr2)**2 for arr1, arr2 in zip(fields1, fields2)]
        # Sum the squared differences of each component
        mag_diffs = sum(diffs_sq)
        # Now compute the norm(E)^2 of the comparison sim at each sampling
        # point
        normsq = sum([np.absolute(field)**2 for field in fields1])
        # We define the percent difference as the ratio of the sums of the
        # difference vector magnitudes to the comparison field magnitudes,
        # squared rooted.
        # TODO: This seems like a somewhat shitty metric that washes out any
        # large localized deviations. Should test other metrics
        diff = np.sqrt(np.sum(mag_diffs) / np.sum(normsq))
        self.log.info('Percent difference = {}'.format(diff))
        return diff

    def adaptive_convergence(self):
        """Performs adaptive convergence by checking the error between vector
        fields for simulations with two different numbers of basis terms.
        Returns the field array, last number of basis terms simulated, and a
        boolean representing whether or not the simulation is converged"""
        self.log.info('Beginning adaptive convergence procedure')
        start_basis = self.conf['Simulation']['params']['numbasis']['value']
        basis_step = self.conf['General']['basis_step']
        ex, ey, ez = self.get_field()
        max_diff = self.conf['General']['max_diff']
        max_iter = self.conf['General']['max_iter']
        percent_diff = 100
        iter_count = 0
        while percent_diff > max_diff and iter_count < max_iter:
            new_basis = start_basis + basis_step
            self.log.info('Checking error between {} and {} basis'
                          ' terms'.format(start_basis, new_basis))
            self.set_basis(new_basis)
            self.build_device()
            self.set_excitation()
            ex2, ey2, ez2 = self.get_field()
            percent_diff = self.calc_diff([ex, ey, ex], [ex2, ey2, ez2])
            start_basis = new_basis
            ex, ey, ez = ex2, ey2, ez2
            iter_count += 1
        if percent_diff > max_diff:
            self.log.warning('Exceeded maximum number of iterations')
            return ex2, ey2, ez2, new_basis, False
        else:
            self.log.info('Converged at {} basis terms'.format(new_basis))
            return ex2, ey2, ez2, new_basis, True

    def mode_solve(self, update=False):
        """Find modes of the system. Supposedly you can get the resonant modes
        of the system from the poles of the S matrix determinant, but I
        currently can't really make any sense of this output"""
        if not update:
            self.configure()
            self.build_device()
        else:
            self.update_thicknesses()
        mant, base, expo = self.s4.GetSMatrixDeterminant()
        self.log.info('Matissa: %s'%str(mant))
        self.log.info('Base: %s'%str(base))
        self.log.info('Exponent: %s'%str(expo))
        res = mant*base**expo
        self.log.info('Result: %s'%str(res))

    def get_data(self, update=False):
        """Gets all the data for this similation by calling the relevant class
        methods. Basically just a convenient wrapper to execute all the
        functions defined above"""
        self.conf.write(os.path.join(self.dir, 'sim_conf.yml'))
        start = time.time()
        if not update:
            self.configure()
            self.build_device()
            self.set_excitation()
        else:
            self.update_thicknesses()
        self.save_field()
        self.save_fluxes()
        end = time.time()
        runtime = end - start
        time_file = os.path.join(self.dir, 'time.dat')
        with open(time_file, 'w') as out:
            out.write('{}\n'.format(runtime))
        self.log.info('Simulation {} completed in {:.2}'
                      ' seconds!'.format(self.id[0:10], runtime))
        return

def main():

    parser = ap.ArgumentParser(description="""Runs a single simulation given
                               the config file for a single simulation""")
    parser.add_argument('config_file', type=str, help="""Absolute path to the
    YAML file specifying the parameters for this simulation""")
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['debug', 'info',
                                 'warning', 'error', 'critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()

    # Load the configuration object
    if os.path.isfile(args.config_file):
        conf = Config(path=os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        sys.exit(1)
    # If we're using gc3 for execution, we need to change the sim_dir to
    # reflect whatever random directory gc3 is running this simulation in so
    # all the output files end up in the correct location
    if conf['General']['execution'] == 'gc3':
        pwd = os.getcwd()
        conf['General']['base_dir'] = pwd
        conf['General']['treebase'] = pwd
    # Instantiate the simulator using this config object
    sim = Simulator(conf)
    try:
        os.makedirs(sim.dir)
    except OSError:
        print('exception raised')
        pass

    if not sim.conf.variable_thickness:
        sim.make_logger()
        sim.log.info('Executing sim %s' % sim.id[0:10])
        sim.get_data()
    else:
        sim.log.info('Computing a thickness sweep at %s' % sim.id[0:10])
        orig_id = sim.id[0:10]
        # Get all combinations of layer thicknesses
        keys, combos, bin_size = get_combos(sim.conf, sim.conf.variable_thickness)
        # Update base directory to new sub directory
        sim.conf['General']['base_dir'] = sim.dir
        # Set things up for the first combo
        combo = combos.pop()
        # First update all the thicknesses in the config. We make a copy of the
        # list because it gets continually updated in the config object
        var_thickness = sim.conf.variable_thickness
        for i in range(len(combo)):
            keyseq = var_thickness[i]
            sim.conf[keyseq] = {'type': 'fixed', 'value': float(combo[i])}
        # With all the params updated we can now make the subdir from the
        # sim id and get the data
        sim.update_id()
        os.makedirs(sim.dir)
        subpath = os.path.join(orig_id, sim.id[0:10])
        sim.make_logger()
        sim.log.info('Computing initial thickness at %s' % subpath)
        sim.get_data()
        # Now we can repeat the same exact process, but instead of rebuilding
        # the device we just update the thicknesses
        for combo in combos:
            for i in range(len(combo)):
                keyseq = var_thickness[i]
                sim.conf[keyseq] = {'type': 'fixed', 'value': float(combo[i])}
            sim.update_id()
            subpath = os.path.join(orig_id, sim.id[0:10])
            os.makedirs(sim.dir)
            sim.make_logger()
            sim.log.info('Computing additional thickness at %s' % subpath)
            sim.get_data(update=True)

if __name__ == '__main__':
    main()
