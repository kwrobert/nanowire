import numpy as np
import os
import time
import glob
import logging
import tables as tb
import scipy.constants as c
from scipy import interpolate
from collections import MutableMapping
from .utils import open_atomic


class TransmissionData(tb.IsDescription):
    layer = tb.StringCol(60, pos=0)
    reflection = tb.Float32Col(pos=1)
    transmission = tb.Float32Col(pos=2)
    absorption = tb.Float32Col(pos=3)


class DataManager(MutableMapping):

    def __init__(self, conf, log):
        self._data = {}
        self._avgs = {}
        self._updated = {}
        self.conf = conf
        self.log = log

    def _update_keys(self):
        raise NotImplementedError

    def _load_data(self, key):
        self._data[key] = None
        self._update[key] = False

    def __getitem__(self, key):
        """
        Here is where the fancy lazy loading is implemented
        """
        if self._data[key] is None:
            self._load_data(key)
            return self._data[key]
        else:
            return self._data[key]

    def __setitem__(self, key, value):
        """
        Check for equality of the existing item in the dict and the value
        passed in. If they are the same, don't bother updating the dict. If
        they are different, replace the existing item and register that this
        key has been updated in the _updated dict so we know to write it later
        on
        """
        # np.array_equal is necessary in case we are dealing with numpy arrays
        # Elementwise comparison of arrays of different shape throws a
        # deprecation warning, and array_equal works on dicts and lists
        try:
            unchanged = np.array_equal(self._data[key], value)
        except KeyError:
            unchanged = False
        if unchanged:
            self.log.info('Data in %s unchanged, not updating', key)
        else:
            self.log.info('Updating %s', key)
            self._data[key] = value
            self._updated[key] = True

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        '''returns simple dict representation of the mapping'''
        return str(self._data)

    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return '{}, D({})'.format(super(HDF5DataManager, self).__repr__(),
                                  self._data)

    def __del__(self):
        """
        Closes data file before being destroyed
        """
        self.log.debug('Closing data file')
        self._dfile.close()


class HDF5DataManager(DataManager):

    """
    We don't want to use the object dict (i.e __dict__) to store the simulation
    data because I dont want to worry about having keys for certain pieces of
    data conflict with some attributes I might want to set on this object
    """

    def __init__(self, conf, log):
        super(HDF5DataManager, self).__init__(conf, log)
        path = os.path.join(self.conf['General']['base_dir'], 'data.hdf5')
        self._dfile = tb.open_file(path, 'a')
        ID = os.path.basename(self.conf['General']['sim_dir'])
        self.gpath = '/sim_{}'.format(ID)
        self.gobj = self._dfile.get_node(self.gpath, classname='Group')
        self._update_keys()

    def _update_keys(self, clear=False):
        """
        Used to pull in keys for all the possible data items this simulation
        could have, without loading the actual items
        """
        for child in self.gobj._f_iter_nodes():
            if clear:
                self._data[child._v_name] = None
                self._updated[child._v_name] = False
            else:
                if child._v_name not in self._data:
                    self._data[child._v_name] = None
                    self._updated[child._v_name] = False

    def _load_data(self, key):
        """
        Implements logic for loading data when the user asks for it by
        accessing the revelant key. This is only called from __getitem__
        """

        nodepath = os.path.join(self.gpath, key)
        try:
            node = self._dfile.get_node(nodepath)
        except tb.NoSuchNodeError:
            # Maybe we just haven't computed transmission data yet
            if key == 'transmission_data':
                return
            else:
                raise tb.NoSuchNodeError
        if isinstance(node, tb.Array):
            self._data[node.name] = node.read()
        elif isinstance(node, tb.Table):
            if key == 'fluxes':
                self._data[key] = {tup[0].decode('utf-8'): (tup[1], tup[2])
                                  for tup in node.read()}
            elif key == 'transmission_data':
                try:
                    self._data['transmission_data'] = {tup[0].decode('utf-8'):
                                                       (tup[1], tup[2], tup[3])
                                                       for tup in node.read()}
                except tb.NoSuchNodeError:
                    pass

    def write_data(self):
        """
        Writes all necessary data out to the HDF5 file
        """

        self.log.info('Beginning HDF5 data writing procedure')
        # Filter out the original data so we don't resave it
        black_list = ('fluxes', 'Ex', 'Ey', 'Ez', 'transmission_data')
        for key, arr in self._data.items():
            if key not in black_list and self._updated[key]:
                self.log.info('Writing data for %s', key)
                try:
                    existing_arr = self._dfile.get_node(self.gpath, name=key)
                    existing_arr[...] = arr
                except tb.NoSuchNodeError:
                    if self.conf['General']['compression']:
                        filt = tb.Filters(complevel=4, complib='blosc')
                        self._dfile.create_carray(self.gpath, key, obj=arr,
                                                filters=filt,
                                                atom=tb.Atom.from_dtype(arr.dtype))
                    else:
                        self._dfile.create_array(self.gpath, key, arr)
            else:
                self.log.info('Data for %s unchanged, not writing', key)
            num_rows = len(list(self.conf['Layers'].keys()))*2
        # We need to handle transmission_data separately because it gets
        # saved into a table
        if self._updated['transmission_data']:
            self.log.info('Writing transmission data')
            try:
                tb_path = self.gpath + '/transmission_data'
                table = self._dfile.get_node(tb_path, classname='Table')
                # If the table exists, clear it out
                table.remove_rows(start=0)
            except tb.NoSuchNodeError:
                table = self._dfile.create_table(self.gpath, 'transmission_data',
                                               description=TransmissionData,
                                               expectedrows=num_rows)
            for port, tup in self._data['transmission_data'].items():
                row = table.row
                row['layer'] = port
                row['reflection'] = tup[0]
                row['transmission'] = tup[1]
                row['absorption'] = tup[2]
                row.append()
            table.flush()
        else:
            self.log.info('Data for transmission_data unchanged, not writing')
            


class NPZDataManager(MutableMapping):

    def __init__(self, conf, log):
        super(NPZDataManager, self).__init__(conf, log)
        self._update_keys()
        path = os.path.join(self.conf['General']['sim_dir'],
                            'field_data.npz')
        self._dfile = np.load(path)

    def _update_keys(self, clear=False):
        for key in self._dfile.files:
            if clear:
                self._data[key] = None
            else:
                if key not in self._data:
                    self._data[key] = None

    def _load_data(self, key):
        """
        Actually pulls data from disk out of the _dfile NPZ archive for the
        requested key and puts it in the self._data disk
        """

        if key == 'fluxes' or key == 'transmission_data':
            if key in self._dfile:
                # We have do so some weird stuff here to unpack the
                # dictionaries because np.savez sticks them in a 0D array for
                # some reason
                self._data[key] == self._dfile[key][()]
        else:
            self._data[key] = self._dfile[key]

    def write_data(self):
        """
        Writes all the data in the _data dict to disk. Unfortunately numpy npz
        archives don't support setting individual items in the NPZArchive
        object (i.e _dfile) and only writing the changes, so if any data key
        has been updated we need to write the entire dict for now
        """
        # TODO: Stop using .npz archives and make my own wrapper around a bunch
        # of individual npy files

        # Get the current path
        base = self.conf['General']['sim_dir']
        self.log.info('Writing data for %s', base)
        fname = os.path.join(base, self.conf['General']['base_name'])
        # Save the headers and the data
        with open_atomic(fname, 'w') as out:
            np.savez_compressed(out, **self.data)
        # Save any local averages we have computed
        dpath = os.path.join(base, 'all.avg')
        with open_atomic(dpath, 'w') as out:
            np.savez_compressed(out, **self.avgs)


class Simulation:
    """
    An object that represents a simulation. It contains the data for the
    sim, the data file headers, and its configuration object as attributes.
    This object contains all the information about a simulation, but DOES NOT
    perform any actions on that data. That job is relegated to the various
    processor objects
    """

    def __init__(self, conf):
        self.conf = conf
        self.log = logging.getLogger('postprocess')
        self.data = self._get_data_manager()
        self.failed = False
        self.avgs = {}
        # Compute and store dx, dy, dz at attributes
        self.z_samples = int(conf['Simulation']['z_samples'])
        self.x_samples = int(conf['Simulation']['x_samples'])
        self.y_samples = int(conf['Simulation']['y_samples'])
        max_depth = conf['Simulation']['max_depth']
        if max_depth:
            self.height = max_depth
            self.dz = max_depth / self.z_samples
        else:
            self.height = self.conf.get_height()
            self.dz = self.height / self.z_samples
        self.period = conf['Simulation']['params']['array_period']['value']
        self.dx = self.period / self.x_samples
        self.dy = self.period / self.y_samples
        self.id = os.path.basename(conf['General']['sim_dir'])

    def _get_data_manager(self):
        """
        Factory function that instantiates the correct data manager object
        depending on the specified file type
        """
        ftype = self.conf['General']['save_as']
        if ftype == 'npz':
            return NPZDataManager(self.conf, self.log)
        elif ftype == 'hdf5':
            return HDF5DataManager(self.conf, self.log)
        else:
            raise ValueError('Invalid file type in config')

    def write_data(self):
        """Writes the data. All writes have been wrapped with an atomic context
        manager than ensures all writes are atomic and do not corrupt data
        files if they are interrupted for any reason"""
        start = time.time()
        self.data.write_data()
        end = time.time()
        self.log.info('Write time: %.2f seconds', end - start)

    def get_scalar_quantity(self, quantity):
        """
        Retrieves the entire 3D matrix for some scalar quantity

        :param str quantity: The quantity you would like to retrive (ex: 'Ex'
                             or 'normE')
        :return: A 3D numpy array of the specified quantity
        :raises KeyError: If the specified quantity does not exist in the data
                          dict

        """

        self.log.debug('Retrieving scalar quantity %s', str(quantity))
        try:
            return self.data[quantity]
        except KeyError:
            self.log.error('You attempted to retrieve a quantity that does not'
                           ' exist in the data dict')
            raise

    def clear_data(self):
        """Clears all the data attributes to free up memory"""
        self.data._update_keys(clear=True)

    def extend_data(self, quantity, new_data):
        """
        Adds a new column of data to the field arrays, or updates it if it
        already exists
        """
        if quantity in self.data:
            self.log.debug("Quantity %s exists in matrix, updating", quantity)
            self.data[quantity] = new_data
        else:
            self.log.debug('Adding %s to data dict', str(quantity))
            self.data[quantity] = new_data

    def normE(self):
        """
        Calculates the norm of E. Adds it to the data dict for the simulation
        and also returns a 3D array
        :return: A 3D numpy array containing normE
        """

        # Get the magnitude of E and add it to our data
        E_mag = np.zeros_like(self.data['Ex'], dtype=np.float64)
        for comp in ('Ex', 'Ey', 'Ez'):
            E_mag += np.absolute(self.data[comp])
        self.extend_data('normE', E_mag)
        return E_mag

    def normEsquared(self):
        """Calculates and returns normE squared"""

        # Get the magnitude of E and add it to our data
        E_magsq = np.zeros_like(self.data['Ex'], dtype=np.float64)
        for comp in ('Ex', 'Ey', 'Ez'):
            E_magsq += np.absolute(self.data[comp])**2
        self.extend_data('normEsquared', E_magsq)
        return E_magsq

    def normH(self):
        """Calculate and returns the norm of H"""

        H_mag = np.zeros_like(self.data['Hx'], dtype=np.float64)
        for comp in ('Hx', 'Hy', 'Hz'):
            H_mag += np.absolute(self.data[comp])
        self.extend_data('normH', H_mag)
        return H_mag

    def normHsquared(self):
        """Calculates and returns the norm of H squared"""

        H_magsq = np.zeros_like(self.data['Hx'], dtype=np.float64)
        for comp in ('Hx', 'Hy', 'Hz'):
            H_magsq += np.absolute(self.data[comp])**2
        self.extend_data('normHsquared', H_magsq)
        return H_magsq

    def get_nk(self, path, freq):
        """Returns functions to compute index of refraction components n and k at a given
        frequency"""
        # Get data
        freq_vec, n_vec, k_vec = np.loadtxt(path, unpack=True)
        # Get n and k at specified frequency via interpolation
        f_n = interpolate.interp1d(freq_vec, n_vec, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        f_k = interpolate.interp1d(freq_vec, k_vec, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        return f_n(freq), f_k(freq)

    def _get_circle_nk(self, shape, sdata, nk, samps, steps):
        """Returns a 2D matrix containing the N,K values at each point in space
        for a circular in-plane geometry"""
        # Set up the parameters
        cx, cy = sdata['center']['x'], sdata['center']['y']
        rad = sdata['radius']
        rad_sq = rad * rad
        mat = sdata['material']
        dx = steps[0]
        dy = steps[1]
        # Build the matrix
        nk_mat = np.full((samps[1], samps[0]), np.nan)
        for xi in range(samps[0]):
            for yi in range(samps[1]):
                dist = ((xi * dx) - cx)**2 + ((yi * dy) - cy)**2
                if dist <= rad_sq:
                    nk_mat[yi, xi] = nk[mat][0] * nk[mat][1]
        return nk_mat

    def _genrate_nk_geometry(self, lname, nk, samps, steps):
        """Computes the nk profile in a layer with a nontrivial internal
        geometry. Returns a 2D matrix containing the product of n and k at each
        point
        lname: Name of the layer as a string
        ldata: This dict containing all the data for this layer
        nk: A dictionary with the material name as the key an a tuple
        containing (n,k) as the value
        samps: A tuple/list containing the number of sampling points in each
        spatial direction in (x,y,z) order
        steps: Same as samps but instead contains the step sizes in each
        direction"""
        # Initialize the matrix with values for the base material
        base_mat = self.conf['Layers'][lname]['base_material']
        nk_mat = nk[base_mat][0] * nk[base_mat][1] * \
            np.ones((samps[1], samps[0]))
        # Get the shapes sorted in increasing order
        shapes = self.conf.sorted_dict(self.conf['Layers'][lname]['geometry'])
        # Loop through the shapes. We want them in increasing order so the
        # smallest shape, which is contained within all the other shapes and
        # should override their nk values, goes last
        for shape, sdata in shapes.items():
            if sdata['type'] == 'circle':
                update = self._get_circle_nk(shape, sdata, nk, samps, steps)
            else:
                raise NotImplementedError('Computing generation rate for layers'
                                          ' with %s shapes is not currently supported' % sdata['type'])
            # Update the matrix with the values from this new shape. The update
            # array will contain nonzero values within the shape, and nan
            # everywhere else. This line updates the nk_mat with only the
            # not nans from the update matrix, and leaves all other elements
            # untouched
            # nk_mat = np.where(update != 0, update, nk_mat)
            nk_mat = np.where(np.isnan(update), nk_mat, update)
        return nk_mat

    def genRate(self):
        # We need to compute normEsquared before we can compute the generation
        # rate
        try:
            normEsq = self.get_scalar_quantity('normEsquared')
        except KeyError:
            normEsq = self.normEsquared()
            self.extend_data('normEsquared', normEsq)
            # Make sure we don't compute it twice
            try:
                self.conf['Postprocessing']['Cruncher'][
                    'normEsquared']['compute'] = False
            except KeyError:
                pass
        # Prefactor for generation rate. Note we gotta convert from m^3 to cm^3,
        # hence 1e6 factor
        fact = c.epsilon_0 / (c.hbar * 1e6)
        # Get the indices of refraction at this frequency
        freq = self.conf['Simulation']['params']['frequency']['value']
        nk = {mat: (self.get_nk(matpath, freq)) for mat, matpath in
              self.conf['Materials'].items()}
        nk['vacuum'] = (1, 0)
        self.log.debug(nk)
        # Get spatial discretization
        samps = (self.x_samples, self.y_samples, self.z_samples)
        gvec = np.zeros_like(normEsq)
        steps = (self.dx, self.dy, self.dz)
        # Main loop to compute generation in each layer
        boundaries = []
        count = 0
        ordered_layers = self.conf.sorted_dict(self.conf['Layers'])
        for layer, ldata in ordered_layers.items():
            # Get boundaries between layers and their starting and ending
            # indices
            layer_t = ldata['params']['thickness']['value']
            self.log.debug('LAYER: %s', layer)
            self.log.debug('LAYER T: %f', layer_t)
            if count == 0:
                start = 0
                end = int(layer_t / self.dz) + 1
                boundaries.append((layer_t, start, end))
            else:
                prev_tup = boundaries[count - 1]
                dist = prev_tup[0] + layer_t
                start = prev_tup[2]
                end = int(dist / self.dz) + 1
                boundaries.append((dist, start, end))
            self.log.debug('START: %i', start)
            self.log.debug('END: %i', end)
            if 'geometry' in ldata:
                self.log.debug('HAS GEOMETRY')
                # This function returns the N,K profile in that layer as a 2D
                # matrix. Each element contains the product of n and k at that
                # point, using the NK values for the appropriate material
                nk_mat = self._genrate_nk_geometry(layer, nk, samps, steps)
                gvec[start:end, :, :] = fact * \
                    nk_mat * normEsq[start:end, :, :]
            else:
                # Its just a simple slab
                self.log.debug('NO GEOMETRY')
                lmat = ldata['base_material']
                self.log.debug('LAYER MATERIAL: %s', lmat)
                self.log.debug('MATERIAL n: %s', str(nk[lmat][0]))
                self.log.debug('MATERIAL k: %s', str(nk[lmat][1]))
                region = fact * nk[lmat][0] * \
                    nk[lmat][1] * normEsq[start:end, :, :]
                self.log.debug('REGION SHAPE: %s', str(region.shape))
                self.log.debug('REGION: ')
                self.log.debug(str(region))
                gvec[start:end, :, :] = region
            self.log.debug('GEN RATE MATRIX: ')
            self.log.debug(str(gvec))
            count += 1
        self.extend_data('genRate', gvec)
        return gvec

    def angularAvg(self, quantity):
        """Perform an angular average of some quantity for either the E or H field"""
        try:
            quant = self.get_scalar_quantity(quantity)
        except KeyError:
            getattr(self, quantity)()
            quant = self.get_scalar_quantity(quantity)
            # Make sure we don't compute it twice
            try:
                self.conf['Postprocessing']['Cruncher'][
                    quantity]['compute'] = False
            except KeyError:
                pass
        # Get spatial discretization
        rsamp = self.conf['Simulation']['r_samples']
        thsamp = self.conf['Simulation']['theta_samples']
        period = self.conf['Simulation']['params']['array_period']['value']
        x = np.linspace(0, period, self.x_samples)
        y = np.linspace(0, period, self.y_samples)
        # Maximum r value such that circle and square unit cell have equal area
        rmax = period / np.sqrt(np.pi)
        # Diff between rmax and unit cell boundary at point of maximum
        # difference
        delta = rmax - period / 2.0
        # Extra indices we need to expand layers by
        x_inds = int(np.ceil(delta / self.dx))
        y_inds = int(np.ceil(delta / self.dy))
        # Use periodic BCs to extend the data in the x-y plane
        ext_vals = np.zeros((quant.shape[0], quant.shape[1] +
                             2 * x_inds, quant.shape[2] + 2 * y_inds),
                            dtype=quant.dtype)
        # Left-Right extensions. This indexing madness extracts the slice we
        # want, flips it along the correct dimension then sticks in the correct
        # spot in the extended array
        ext_vals[:, x_inds:-x_inds, 0:y_inds] = quant[:, :, 0:y_inds][:, :, ::-1]
        ext_vals[:, x_inds:-x_inds, -
                 y_inds:] = quant[:, :, -y_inds:][:, :, ::-1]
        # Top-Bottom extensions
        ext_vals[:, 0:x_inds, y_inds:-
                 y_inds] = quant[:, 0:x_inds, :][:, ::-1, :]
        ext_vals[:, -x_inds:, y_inds:-
                 y_inds] = quant[:, -x_inds:, :][:, ::-1, :]
        # Corners, slightly trickier
        # Top left
        ext_vals[:, 0:x_inds, 0:y_inds] = ext_vals[
            :, x_inds:2 * x_inds, 0:y_inds][:, ::-1, :]
        # Bottom left
        ext_vals[:, -x_inds:, 0:y_inds] = ext_vals[:, -
                                                   2 * x_inds:-x_inds, 0:y_inds][:, ::-1, :]
        # Top right
        ext_vals[:, 0:x_inds, -y_inds:] = ext_vals[:,
                                                   0:x_inds, -2 * y_inds:-y_inds][:, :, ::-1]
        # Bottom right
        ext_vals[:, -x_inds:, -y_inds:] = ext_vals[:, -
                                                   x_inds:, -2 * y_inds:-y_inds][:, :, ::-1]
        # Now the center
        ext_vals[:, x_inds:-x_inds, y_inds:-y_inds] = quant[:, :, :]
        # Extend the points arrays to include these new regions
        x = np.concatenate((np.array([self.dx * i for i in
                                      range(-x_inds, 0)]), x, np.array([x[-1] + self.dx * i for i in range(1, x_inds + 1)])))
        y = np.concatenate((np.array([self.dy * i for i in
                                      range(-y_inds, 0)]), y, np.array([y[-1] + self.dy * i for i in range(1, y_inds + 1)])))
        # The points on which we have data
        points = (x, y)
        # The points corresponding to "rings" in cylindrical coordinates. Note we construct these
        # rings around the origin so we have to shift them to actually correspond to the center of
        # the nanowire
        rvec = np.linspace(0, rmax, rsamp)
        thvec = np.linspace(0, 2 * np.pi, thsamp)
        cyl_coords = np.zeros((len(rvec) * len(thvec), 2))
        start = 0
        for r in rvec:
            xring = r * np.cos(thvec)
            yring = r * np.sin(thvec)
            cyl_coords[start:start + len(thvec), 0] = xring
            cyl_coords[start:start + len(thvec), 1] = yring
            start += len(thvec)
        cyl_coords += period / 2.0
        # For every z layer in the 3D matrix of our quantity
        avgs = np.zeros((ext_vals.shape[0], len(rvec)))
        i = 0
        for layer in ext_vals:
            interp_vals = interpolate.interpn(
                points, layer, cyl_coords, method='linear')
            rings = interp_vals.reshape((len(rvec), len(thvec)))
            avg = np.average(rings, axis=1)
            avgs[i, :] = avg
            i += 1
        avgs = avgs[:, ::-1]
        # Save to avgs dict for this sim
        key = quantity + '_angularAvg'
        self.data[key] = avgs
        return avgs

    def transmissionData(self, port='Substrate'):
        """
        Computes reflection, transmission, and absorbance

        :param sim: :py:class:`utils.simulation.Simulation`
        :param str port: Name of the location at which you would like to place the
                         transmission port (i.e where you would like to compute
                         transmission). This must correspond to one of the keys placed in
                         the fluxes.dat file
        """

        data = self.data['fluxes']
        # sorted_layers is an OrderedDict, and thus has the popitem method
        sorted_layers = self.conf.sorted_dict(self.conf['Layers'])
        # self.log.info('SORTED LAYERS: %s', str(sorted_layers))
        first_layer = sorted_layers.popitem(last=False)
        self.log.info('FIRST LAYER: %s', str(first_layer))
        # An ordered dict is actually just a list of tuples so we can access
        # the key directly like so
        first_name = first_layer[0]
        self.log.info('FIRST LAYER NAME: %s', str(first_name))
        # p_inc = data[first_name][0]
        # p_ref = np.abs(data[first_name][1])
        # p_trans = data[last_name][0]
        p_inc = np.absolute(data[first_name][0])
        p_ref = np.absolute(data[first_name][1])
        p_trans = np.absolute(data[port][0])
        reflectance = p_ref / p_inc
        transmission = p_trans / p_inc
        absorbance = 1 - reflectance - transmission
        tot = reflectance + transmission + absorbance
        delta = np.abs(tot - 1)
        self.log.info('Reflectance %f' % reflectance)
        self.log.info('Transmission %f' % transmission)
        self.log.info('Absorbance %f' % absorbance)
        self.log.info('Total = %f' % tot)
        assert(reflectance >= 0)
        assert(transmission >= 0)
        # assert(absorbance >= 0)
        # assert(delta < .00001)
        if 'transmission_data' in self.data:
            self.data['transmission_data'].update({port: (reflectance,
                                                          transmission,
                                                          absorbance)})
        else:
            self.data['transmission_data'] = {port: (reflectance,
                                                     transmission,
                                                     absorbance)}
        # ftype = self.conf['General']['save_as']
        # if ftype == 'npz':
        #     outpath = os.path.join(base, 'ref_trans_abs.dat')
        #     self.log.info('Writing transmission file')
        #     if os.path.isfile(outpath):
        #         with open(outpath, 'a') as out:
        #             out.write('%s,%f,%f,%f\n' % (port, reflectance, transmission, absorbance))
        #     else:
        #         with open(outpath, 'w') as out:
        #             out.write('# Port, Reflectance,Transmission,Absorbance\n')
        #             out.write('%s,%f,%f,%f\n' % (port, reflectance, transmission, absorbance))
        # elif ftype == 'hdf5':
        #     group = '/sim_{}'.format(self.id)
        #     num_rows = len(list(self.conf['Layers'].keys()))*2
        #     try:
        #         tb_path = group + '/transmission_data'
        #         table = self.hdf5.get_node(tb_path, classname='Table')
        #     except tb.NoSuchNodeError:
        #         table = self.hdf5.create_table(group, 'transmission_data',
        #                                        description=TransmissionData,
        #                                        expectedrows=num_rows)
        #     row = table.row
        #     row['layer'] = port
        #     row['reflection'] = reflectance
        #     row['transmission'] = transmission
        #     row['absorption'] = absorbance
        #     row.append()
        #     table.flush()
        return reflectance, transmission, absorbance

    def integrated_absorbtion(self):
        """Computes the absorption of a layer by using the volume integral of
        the product of the imaginary part of the relative permittivity and the
        norm squared of the E field"""

        raise NotImplementedError('There are some bugs in S4 and other reasons'
                                  ' that this function doesnt work yet')
        base = self.conf['General']['sim_dir']
        path = os.path.join(base, 'integrated_absorption.dat')
        inpath = os.path.join(base, 'energy_densities.dat')
        freq = self.conf['Simulation']['params']['frequency']['value']
        # TODO: Assuming incident amplitude and therefore incident power is
        # just 1 for now
        fact = -.5 * freq * c.epsilon_0
        with open(inpath, 'r') as inf:
            lines = inf.readlines()
            # Remove header line
            lines.pop(0)
            # Dict where key is layer name and value is list of length 2 containing real and
            # imaginary parts of energy density integral
            data = {line.strip().split(',')[0]: line.strip().split(',')[
                1:] for line in lines}
        self.log.info('Energy densities: %s' % str(data))
        with open(path, 'w') as outf:
            outf.write('# Layer, Absorption\n')
            for layer, vals in data.items():
                absorb = fact * float(vals[1])
                outf.write('%s,%s\n' % (layer, absorb))

