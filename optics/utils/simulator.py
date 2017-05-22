from __future__ import print_function
import os
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as intg
import scipy.constants as constants
import S4
#  import profilehooks as ph

from itertools import chain, product
from utils import make_hash, configure_logger


class Simulator():

    def __init__(self, conf, log_level='info'):
        conf.interpolate()
        conf.evaluate()
        self.conf = conf
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        period = self.conf['Simulation']['params']['array_period']['value']
        self.id = make_hash(conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'], self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir
        lfile = os.path.join(sim_dir, 'sim.log')
        self.log = configure_logger(level=log_level, name=self.id[0:10],
                                    logfile=lfile, propagate=False)
        self.s4 = S4.New(Lattice=((period, 0), (0, period)), NumBasis=numbasis)

    def update_id(self):
        """Update sim id. Used after changes are made to the config"""
        self.id = make_hash(self.conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'], self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir

    def configure(self):
        """Configure options for the RCWA solver"""
        self.s4.SetOptions(**self.conf['Solver'])

    def _get_epsilon(self, path):
        """Returns complex dielectric constant for a material by pulling in nk text file, interpolating,
        computing nk values at freq, and converting"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        # Get data
        freq_vec, n_vec, k_vec = np.loadtxt(path, skiprows=1, unpack=True)
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
        avg = self.conf['Simulation']['average_bins']
        bin_size = self.conf['Simulation']['params']['frequency']['bin_size']
        # Get data
        freq_vec, p_vec = np.loadtxt(path, unpack=True, delimiter=',')
        # Get all available power values within this bin
        left = freq - bin_size / 2.0
        right = freq + bin_size / 2.0
        power_inds = np.where((left < freq_vec) & (freq_vec < right))
        freqs = freq_vec[power_inds]
        power_values = p_vec[power_inds[0]]
        # Calculate the average irradiance of this bin and multiply by the bin
        # width to get a power value. This tends to give a smoother result than
        # directly integrating
        if avg:
            power = np.average(power_values) * bin_size
        # Integrate the spectrum within this bin to get a power value
        else:
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
        for layer, ldata in sorted(self.conf['Layers'].items(), key=lambda tup: tup[1]['order']):
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
        arr = np.zeros((x_samp * y_samp * len(zvec), 9))
        count = 0
        for z in zvec:
            E, H = self.s4.GetFieldsOnGrid(
                z=z, NumSamples=(x_samp, y_samp), Format='Array')
            xcount = 0
            for xval in E:
                ycount = 0
                for yval in xval:
                    fixed = [val for c in yval for val in
                             (c.real, c.imag)]
                    row = [xcount, ycount, z] + fixed
                    arr[count, :] = row
                    count += 1
                    ycount += 1
                xcount += 1
        self.log.info('Finished computing fields!')
        return arr

    def row_iter(self, data, xsamp, ysamp, z):
        inds = product(range(xsamp), range(ysamp))
        for raw_row in chain(*data):
            fixed = [val for c in raw_row for val in
                     (c.real, c.imag)]
            ind = inds.next()
            final = [ind[0], ind[1], z] + fixed
            yield final

    #  @ph.timecall
    def get_field2(self):
        """Constructs and returns a 2D numpy array of the vector electric
        field. The field components are complex numbers. This is about 1-2
        seconds slower than the current get_field method, but I'm keeping it
        around because the implementation feels nice"""
        self.log.info('Computing fields ...')
        x_samp = self.conf['Simulation']['x_samples']
        y_samp = self.conf['Simulation']['y_samples']
        z_samp = self.conf['Simulation']['z_samples']
        height = self.get_height()
        dz = height / z_samp
        zvec = np.arange(0, height + dz, dz)
        arr = np.zeros((x_samp * y_samp * len(zvec), 9))
        count = 0
        for z in zvec:
            E, H = self.s4.GetFieldsOnGrid(
                z=z, NumSamples=(x_samp, y_samp), Format='Array')
            for row in self.row_iter(E, x_samp, y_samp, z):
                arr[count, :] = row
                count += 1
        self.log.info('Finished computing fields!')
        return arr

    def save_field(self):
        """Saves the fields throughout the device to an npz file"""
        if self.conf['General']['adaptive_convergence']:
            efield, numbasis, conv = self.adaptive_convergence()
            if conv:
                out = os.path.join(self.dir, 'converged_at.txt')
            else:
                out = os.path.join(self.dir, 'not_converged_at.txt')
            self.log.info('Writing convergence file ...')
            with open(out, 'w') as outf:
                outf.write('{}\n'.format(numbasis))
        else:
            efield = self.get_field()
        out = os.path.join(self.dir, self.conf["General"]["base_name"] + '.E')
        headers = ['x', 'y', 'z', 'Ex', 'Ey', 'Ez']
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
            np.savez_compressed(out, headers=headers, data=efield)
            #  np.savez(out,headers=headers,data=efield)
        else:
            raise ValueError('Invalid file type specified in config')

    def get_fluxes(self):
        """Get the fluxes at the top and bottom of each layer. Returns a dict
        where the keys are the layer name and the values are a length 2 tuple
        with the forward component first and the backward component second. The
        components are complex numbers"""
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

    def calc_diff(self, d1, d2, exclude=False):
        """Calculate the percent difference between two vector fields"""
        # This is a 2D table where each row contains the differences between
        # each electric field component at the corresponding point in space
        # squared. This is the magnitude squared of the difference vector
        # between the two datasets at each sampling point
        diffs_sq = (d1[:, 3:] - d2[:, 3:])**2
        # Sum along the columns to get the sum of the squares of the components
        # of the difference vector. This should be a 1D array
        mag_diffs = np.sum(diffs_sq, axis=1)
        # Now compute the norm(E)^2 of the comparison sim at each sampling
        # point
        normsq = np.sum(d1**2, axis=1)
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
        field1 = self.get_field()
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
            field2 = self.get_field()
            percent_diff = self.calc_diff(field1, field2)
            start_basis = new_basis
            field1 = field2
            iter_count += 1
        if percent_diff > max_diff:
            self.log.warning('Exceeded maximum number of iterations')
            return field2, new_basis, False
        else:
            self.log.info('Converged at {} basis terms'.format(new_basis))
            return field2, new_basis, True
