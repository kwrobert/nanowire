import os
import matplotlib
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use(['ggplot', 'paper'])
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import scipy.constants as c
from scipy import interpolate
import scipy.integrate as intg
import tables as tb
import numpy as np
from collections import MutableMapping
from .utils import open_atomic
import copy
import time
import glob
import logging
import itertools


class TransmissionData(tb.IsDescription):
    layer = tb.StringCol(60, pos=0)
    reflection = tb.Float32Col(pos=1)
    transmission = tb.Float32Col(pos=2)
    absorption = tb.Float32Col(pos=3)


class DataManager(MutableMapping):
    """
    The base class for all *DataManager objects. This purpose of this object is
    to manage retrieving data from some on-disk format and storing it in a
    dictionary for later retrieval (the _data dict). This object behaves like a
    dict, and overrides all the key dictionary special methods.

    Lazy-loading logic is implemented. When initialized, this object should
    populate the keys of _data with all data items available on disk without
    loading them, instead storing None as the value. An item is only actually
    retrieved from the on-disk format when it is requested.

    Lazy writing is implemented as well. The value corresponding to a given key
    is updated if and only if the attempted assignment value differs from the
    existing value. If so, the _updated dict stores True to indicate that data
    value has been updated. Upon writing, only update values are written to
    disk. DISCLAIMER: This doesn't actually work for the NPZ backend because
    IDK how to modify individual arrays within the archive.

    We don't use the object dict (i.e __dict__) to store the simulation
    data because I dont want to worry about having keys for certain pieces of
    data conflict with some attributes I might want to set on this object. It's
    slightly less memory efficient but not in a significant way.
    """

    def __init__(self, conf, log):
        self._data = {}
        self._avgs = {}
        self._updated = {}
        self.conf = conf
        self.log = log
        self._dfile = None

    def _update_keys(self):
        raise NotImplementedError

    def _load_data(self, key):
        self._data[key] = None
        self._updated[key] = False

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
        return '{}, D({})'.format(super(DataManager, self).__repr__(),
                                  self._data)

    def __del__(self):
        """
        Closes data file before being destroyed
        """
        self.log.debug('Closing data file')
        self._dfile.close()


class HDF5DataManager(DataManager):
    """
    Data manager class for the HDF5 storage backend
    """

    def __init__(self, conf, log):
        """
        :param :class:`~utils.config.Config`: Config object for the simulation
        that this DataManager will be managing data for
        :param log: A logger object
        """

        super(HDF5DataManager, self).__init__(conf, log)
        path = os.path.join(self.conf['General']['sim_dir'], 'sim.hdf5')
        self._dfile = tb.open_file(path, 'a')
        ID = os.path.basename(self.conf['General']['sim_dir'])
        self.gpath = '/sim_{}'.format(ID)
        self.gobj = self._dfile.get_node(self.gpath, classname='Group')
        self._update_keys()

    def _update_keys(self, clear=False):
        """
        Used to pull in keys for all the data items this simulation has stored
        on disk, without loading the actual items
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


class NPZDataManager(DataManager):

    def __init__(self, conf, log):
        """
        :param :class:`~utils.config.Config`: Config object for the simulation
        that this DataManager will be managing data for
        :param log: A logger object
        """

        super(NPZDataManager, self).__init__(conf, log)
        self._update_keys()
        path = os.path.join(self.conf['General']['sim_dir'],
                            'field_data.npz')
        self._dfile = np.load(path)

    def _update_keys(self, clear=False):
        """
        Used to pull in keys for all the data items this simulation has stored
        on disk, without loading the actual items
        """
        for key in self._dfile.files:
            if clear:
                self._data[key] = None
            else:
                if key not in self._data:
                    self._data[key] = None

    def _load_data(self, key):
        """
        Actually pulls data from disk out of the _dfile NPZ archive for the
        requested key and puts it in the self._data dict for later retrieval
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
            np.savez_compressed(out, **self._data)
        # Save any local averages we have computed
        dpath = os.path.join(base, 'all.avg')
        with open_atomic(dpath, 'w') as out:
            np.savez_compressed(out, **self._avgs)


class Simulation:
    """
    An object that represents a simulation. It stores a DataManager object for
    managing the reading and writing of all data as an attribute. It also
    stores a Config object, which is a dict-like object representing the
    configuration for the simulation, as an attribute. Many of the methods are
    for performing calculations on the data.
    """

    def __init__(self, conf):
        """
        :param :class:`~utils.config.Config`: Config object for this simulation
        """

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
        depending on the file type specified in the config
        """

        ftype = self.conf['General']['save_as']
        if ftype == 'npz':
            return NPZDataManager(self.conf, self.log)
        elif ftype == 'hdf5':
            return HDF5DataManager(self.conf, self.log)
        else:
            raise ValueError('Invalid file type in config')

    def write_data(self):
        """
        Writes the data. This is a simple wrapper around the write_data()
        method of the DataManager object, with some code to compute the time it
        took to perform the write operation
        """

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
        Adds a new key, value pair to the DataManager object
        """
        if quantity in self.data:
            self.log.debug("Quantity %s exists in matrix, updating", quantity)
            self.data[quantity] = new_data
        else:
            self.log.debug('Adding %s to data dict', str(quantity))
            self.data[quantity] = new_data

    def get_line(self, quantity, line_dir, c1, c2):
        """
        Gets data along a line through the 3D data array for the given quantity
        along a given direction

        :param str line_dir: Any of 'x', 'y', or 'z'. Determines the direction
        along which the line cut is taken, the other two coordinates remain
        fixed and are specified by c1 and c2.
        :param int c1: The integer index for the first fixed coordinate.
        Indexes are in x,y, z order so if line_dir='z' then c1 corresponds to x
        :param int c2: The integer index for the second coordinate.
        :param str quantity: The quantity whose data array you wish to take a
        line cut through
        """

        scalar = self.get_scalar_quantity(quantity)
        if line_dir == 'x':
            # z along rows, y along columns
            return scalar[c2, :, c1]
        elif line_dir == 'y':
            # x along columns, z along rows
            return scalar[c2, c1, :]
        elif line_dir == 'z':
            # x along rows, y along columns
            return scalar[:, c1, c2]

    def get_plane(self, quantity, plane, pval):
        """
        Gets data along a 2D plane/slice through the 3D data array for a given
        quantity

        :param str plane: Any of 'xy', 'yz', or 'xz'. Determines the plane
        along which the slice is taken
        :param int pval: The index along the final unspecified direction. If
        plane='xy' then index would index along the z direction.
        :param str quantity: The quantity whose data array you wish to take a
        line cut through
        """

        self.log.info('Retrieving plane for %s', quantity)
        scalar = self.get_scalar_quantity(quantity)
        if plane == 'yz' or plane == 'zy':
            # z along rows, y along columns
            return scalar[:, pval, :]
        elif plane == 'xz' or plane == 'zx':
            # x along columns, z along rows
            return scalar[:, :, pval]
        elif plane == 'xy' or plane == 'yx':
            # x along rows, y along columns
            return scalar[pval, :, :]

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
        """
        Calculates and returns normE squared. Adds it to the data dict for
        the simulation and also returns a 3D array
        :return: A 3D numpy array containing normE squared
        """

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
        """
        Returns n and k, the real and imaginary components of the index of refraction at a given
        frequency
        :param str path: A path to a text file containing the n and k data. The
        first column should be the frequency in Hertz, the second column the n
        values, and the third column the k values. Columns must be delimited
        by whitespace.
        :param float freq: The desired frequency in Hertz
        """

        # Get data
        freq_vec, n_vec, k_vec = np.loadtxt(path, unpack=True)
        # Get n and k at specified frequency via interpolation
        f_n = interpolate.interp1d(freq_vec, n_vec, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        f_k = interpolate.interp1d(freq_vec, k_vec, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        return f_n(freq), f_k(freq)

    def _get_circle_nk(self, shape, sdata, nk, samps, steps):
        """
        Returns a 2D matrix containing the N,K values at each point in space
        for a circular in-plane geometry
        :param dict sdata: The dictionary containing the parameters and info
        about this circle
        :param dict nk: The dict containing the n and k values for all the
        materials used in the sim. Keys are material names, values are tuples
        containing (n, k)
        :param tup samps: The number of samples taken in the (x, y) directions
        :param tup steps: The length of the spatial discretization step in the
        (x, y) directions
        """

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
        """
        Computes the nk profile in a layer with a nontrivial internal
        geometry. Returns a 2D matrix containing the product of n and k at each
        point
        :param str lname: Name of the layer as a string
        :param dict ldata: This dict containing all the data for this layer
        :param dict nk: A dictionary with the material name as the key an a tuple
        containing (n,k) as the value
        :param tup samps: A tuple/list containing the number of sampling points in each
        spatial direction in (x,y,z) order
        :param tup steps: Same as samps but instead contains the step sizes in each
        direction
        """

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
        """
        Computes and returns the 3D matrix containing the generation rate.
        Returns in units of cm^-3
        """

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
        """
        Perform an angular average of some quantity for either the E or H field
        """

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

        :param str port: Name of the location at which you would like to place the
                         transmission port (i.e where you would like to compute
                         transmission). This must correspond to one of the keys placed in
                         the fluxes dict located at self.data['fluxes']
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
        assert(absorbance >= 0)
        # assert(delta < .00001)
        if 'transmission_data' in self.data:
            new = copy.deepcopy(self.data['transmission_data'])
            new.update({port: (reflectance, transmission, absorbance)})
            self.data['transmission_data'] = new
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
        """
        Computes the absorption of a layer by using the volume integral of
        the product of the imaginary part of the relative permittivity and the
        norm squared of the E field
        """

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

    def _draw_layer_circle(self, ldata, shape_key, start, end, plane, pval, ax_hand):
        """Draws the circle within a layer"""
        shape_data = ldata['geometry'][shape_key]
        center = shape_data['center']
        radius = shape_data['radius']
        if plane == 'xy':
            circle = mpatches.Circle((center['x'], center['y']), radius=radius,
                                     fill=False)
            ax_hand.add_artist(circle)
        if plane in ["xz", "zx", "yz", "zy"]:
            plane_x = pval*self.dx
            plane_to_center = np.abs(center['x'] - plane_x)
            self.log.debug('DIST: {}'.format(plane_to_center))
            # Only draw if the observation plane actually intersects with the
            # circle
            if not plane_to_center >= radius:
                # Check if the plane intersects with the center of the circle
                if plane_to_center > 0:
                    intersect_angle = np.arccos(plane_to_center/radius)
                    self.log.debug('ANGLE: {}'.format(intersect_angle))
                    half_width = plane_to_center*np.tan(intersect_angle)
                else:
                    half_width = radius
                self.log.debug('HALF WIDTH: {}'.format(half_width))
                # Vertical lines should span height of the layer
                z = [self.height - start * self.dz, self.height - end * self.dz]
                # The upper edge
                x = [center['y'] + half_width, center['y'] + half_width]
                line = mlines.Line2D(x, z, linestyle='solid', linewidth=2.0,
                                     color='grey')
                ax_hand.add_line(line)
                # Now the lower edge
                x = [center['y'] - half_width, center['y'] - half_width]
                line = mlines.Line2D(x, z, linestyle='solid', linewidth=2.0,
                                     color='grey')
                ax_hand.add_line(line)
        return ax_hand

    def _draw_layer_geometry(self, ldata, start, end, plane, pval, ax_hand):
        """Given a dictionary with the data containing the geometry for a
        layer, draw the internal geometry of the layer for a given plane type
        and plane value"""
        for shape, data in ldata['geometry'].items():
            if data['type'] == 'circle':
                ax = self._draw_layer_circle(ldata, shape, start, end,
                                             plane, pval, ax_hand)
            else:
                self.log.warning('Drawing of shape {} not '
                                 'supported'.format(data['type']))
        return ax

    def draw_geometry_2d(self, plane, pval, ax_hand):
        """This function draws the layer boundaries and in-plane geometry on 2D
        heatmaps"""
        # Get the layers in order
        ordered_layers = self.conf.sorted_dict(self.conf['Layers'])
        period = self.conf['Simulation']['params']['array_period']['value']
        # Loop through them
        boundaries = []
        count = 0
        for layer, ldata in ordered_layers.items():
            # If x or y, draw bottom edge and text label now. Layer geometry
            # is handled in its own function
            if plane in ["xz", "zx", "yz", "zy"]:
                # Get boundaries between layers and their starting and ending
                # indices
                layer_t = ldata['params']['thickness']['value']
                if count == 0:
                    start = 0
                    end = int(layer_t / self.dz) + 1
                    boundaries.append((layer_t, start, end, layer))
                else:
                    prev_tup = boundaries[count - 1]
                    dist = prev_tup[0] + layer_t
                    start = prev_tup[2]
                    end = int(dist / self.dz) + 1
                    boundaries.append((dist, start, end))
                if layer_t > 0:
                    x = [0, period]
                    y = [self.height - start * self.dz,
                         self.height - start * self.dz]
                    label_y = y[0] - 0.25
                    label_x = x[-1] - .01
                    plt.text(label_x, label_y, layer, ha='right',
                             family='sans-serif', size=16, color='grey')
                    line = mlines.Line2D(x, y, linestyle='solid', linewidth=2.0,
                                         color='grey')
                    ax_hand.add_line(line)
                    count += 1
            else:
                # If look at a fixed z pval, the start and end values are
                # nonsensical but we must pass a value in
                start, end = None, None
            # If we have some internal geometry for this layer, draw it
            if 'geometry' in ldata:
                ax = self._draw_layer_geometry(ldata, start, end, plane, pval, ax_hand)
        return ax

    def heatmap2d(self, x, y, cs, labels, ptype, pval, save_path=None,
                  show=False, draw=False, fixed=None, colorsMap='jet'):
        """A general utility method for plotting a 2D heat map"""
        cm = plt.get_cmap(colorsMap)
        if np.iscomplexobj(cs):
            self.log.warning('Plotting only real part of %s in heatmap',
                             labels[2])
            cs = cs.real
        if fixed:
            if 'dielectric_profile' in save_path:
                cNorm = matplotlib.colors.Normalize(
                    vmin=np.amin(0), vmax=np.amax(16))
            else:
                cNorm = matplotlib.colors.Normalize(
                    vmin=np.amin(0), vmax=np.amax(2.5))
        else:
            cNorm = matplotlib.colors.Normalize(
                vmin=np.amin(cs), vmax=np.amax(cs))
            # cNorm = matplotlib.colors.LogNorm(vmin=np.amin(cs)+.001, vmax=np.amax(cs))
            # cNorm = matplotlib.colors.LogNorm(vmin=1e13, vmax=np.amax(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(cs,cmap=cm,norm=cNorm,extent=[x.min(),x.max(),y.min(),y.max()],aspect='auto')
        ax.grid(False)
        scalarMap.set_array(cs)
        # div = make_axes_locatable(ax)
        # zoom_ax = div.append_axes("right",size='100%', pad=.5)
        # zoom_ax.imshow(cs[75:100,:], extent=[x.min(), x.max(), .8, 1.4])
        # zoom_ax.grid(False)
        # cax = div.append_axes("right",size="100%",pad=.05)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[2])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if draw:
            self.log.info('Beginning geometry drawing routines ...')
            ax = self.draw_geometry_2d(ptype, pval, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def plane_2d(self, quantity, plane, pval, draw=False, fixed=None):
        """Plots a heatmap of a fixed 2D plane"""
        self.log.info('Plotting plane')
        pval = int(pval)
        x = np.arange(0, self.period, self.dx)
        y = np.arange(0, self.period, self.dy)
        z = np.arange(0, self.height + self.dz, self.dz)
        # Get the scalar values
        freq = self.conf['Simulation']['params']['frequency']['value']
        wvlgth = (c.c / freq) * 1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(
            freq, wvlgth)
        # Get the plane we wish to plot
        cs = self.get_plane(quantity, plane, pval)
        self.log.info('DATA SHAPE: %s' % str(cs.shape))
        show = self.conf['General']['show_plots']
        p = False
        if plane == 'yz' or plane == 'zy':
            labels = ('y [um]', 'z [um]', quantity, title)
            if self.conf['General']['save_plots']:
                p = os.path.join(self.conf['General']['sim_dir'],
                                 '%s_plane_2d_yz_pval%s.pdf' % (quantity,
                                                               str(pval)))
            self.heatmap2d(y, z, cs, labels, plane, pval,
                           save_path=p, show=show, draw=draw, fixed=fixed)
        elif plane == 'xz' or plane == 'zx':
            labels = ('x [um]', 'z [um]', quantity, title)
            if self.conf['General']['save_plots']:
                p = os.path.join(self.conf['General']['sim_dir'],
                                 '%s_plane_2d_xz_pval%s.pdf' % (quantity,
                                                               str(pval)))
            self.heatmap2d(x, z, cs, labels, plane, pval,
                           save_path=p, show=show, draw=draw, fixed=fixed)
        elif plane == 'xy' or plane == 'yx':
            labels = ('y [um]', 'x [um]', quantity, title)
            if self.conf['General']['save_plots']:
                p = os.path.join(self.conf['General']['sim_dir'],
                                 '%s_plane_2d_xy_pval%s.pdf' % (quantity,
                                                               str(pval)))
            self.heatmap2d(x, y, cs, labels, plane, pval,
                           save_path=p, show=show, draw=draw, fixed=fixed)

    def scatter3d(self, x, y, z, cs, labels, ptype, colorsMap='jet'):
        """A general utility method for scatter plots in 3D"""
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9, 7))

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), edgecolor='none')
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[3])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        fig.suptitle(os.path.basename(self.conf['General']['sim_dir']))
        if self.conf['General']['save_plots']:
            name = labels[-1] + '_' + ptype + '.pdf'
            path = os.path.join(self.conf['General']['sim_dir'], name)
            fig.savefig(path)
        if self.conf['General']['show_plots']:
            plt.show()
        plt.close(fig)

    def full_3d(self, quantity):
        """Generates a full 3D plot of a specified scalar quantity"""
        # The data just tells you what integer grid point you are on. Not what actual x,y coordinate you
        # are at
        x = np.arange(0, self.period, self.dx)
        y = np.arange(0, self.period, self.dy)
        z = np.arange(0, self.height + self.dz, self.dz)
        points = np.array(list(itertools.product(z, x, y)))
        # Get the scalar
        scalar = self.get_scalar_quantity(quantity)
        labels = ('X [um]', 'Y [um]', 'Z [um]', quantity)
        # Now plot!
        self.scatter3d(points[:, 1], points[:, 2], points[
                       :, 0], scalar.flatten(), labels, 'full_3d')

    def planes_3d(self, quantity, xplane, yplane):
        """Plots some scalar quantity in 3D but only along specified x-z and y-z planes"""
        xplane = int(xplane)
        yplane = int(yplane)
        # Get the scalar values
        x = np.linspace(0, self.period, self.x_samples)
        y = np.linspace(0, self.period, self.y_samples)
        z = np.linspace(0, self.height, self.z_samples)
        # Get the data on the plane with a fixed x value. These means we'll
        # have changing (y, z) points
        xdata = self.get_plane(quantity, 'yz', xplane)
        # z first cuz we want y to be changing before z to correspond with the
        # way numpy flattens arrays. Note this means y points will be in the
        # 2nd column
        xplanepoints = np.array(list(itertools.product(z, y)))
        xdata = xdata.flatten()
        xplanexval = np.array(list(itertools.repeat(x[xplane], len(xdata))))
        xplanedata = np.zeros((xplanepoints.shape[0], 4))
        xplanedata[:, 0] = xplanexval
        xplanedata[:, 1] = xplanepoints[:, 1]
        xplanedata[:, 2] = xplanepoints[:, 0]
        xplanedata[:, 3] = xdata
        # Same procedure for fixed y plane
        ydata = self.get_plane(quantity, 'xz', yplane)
        yplanepoints = np.array(list(itertools.product(z, x)))
        ydata = ydata.flatten()
        yplaneyval = np.array(list(itertools.repeat(y[yplane], len(ydata))))
        yplanedata = np.zeros((yplanepoints.shape[0], 4))
        yplanedata[:, 0] = yplanepoints[:, 1]
        yplanedata[:, 1] = yplaneyval
        yplanedata[:, 2] = yplanepoints[:, 0]
        yplanedata[:, 3] = ydata
        labels = ('X [um]', 'Y [um]', 'Z [um]', quantity)
        # Now stack them vertically and plot!
        all_data = np.vstack((xplanedata, yplanedata))
        self.scatter3d(all_data[:, 0], all_data[:, 1], all_data[:, 2],
                       all_data[:, 3], labels, 'planes_3d')

    def line_plot(self, x, y, ptype, labels):
        """Make a simple line plot"""
        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(labels[2])
        if self.conf['General']['save_plots']:
            name = labels[1] + '_' + ptype + '.pdf'
            path = os.path.join(self.conf['General']['sim_dir'], name)
            fig.savefig(path)
        if self.conf['General']['show_plots']:
            plt.show()
        plt.close(fig)

    def fixed_line(self, quantity, direction, coord1, coord2):
        """
        Plot a scalar quantity on a line along a given direction at some pair
        of coordinates in the plane perpendicular to that direction. The
        remaining coordinates are specified in x, y, z order. So for example if
        direction='z' then coord1 corresponds to x and coord2 corresponds to y.
        If direction='y' then coord1 corresponds to x and coord2 corresponds to
        z.
        :param str direction: The direction along which to plot the line. Must
        be one of 'x', 'y', or 'z'. 
        :param str direction: The direction along which you wish to plot a
        line. Must be one of 'x', 'y', or 'z'. The other two coordinates remain
        fixed and are specified by coord1 and coord2.
        :param int coord1: The integer index for the first fixed coordinate.
        Indexes are in x,y, z order so if line_dir='z' then c1 corresponds to x
        :param int coord2: The integer index for the second coordinate.
        :param str quantity: The quantity whose data array you wish to take a
        line cut through
        """

        coord1 = int(coord1)
        coord2 = int(coord2)
        # Get the scalar values
        # Filter out any undesired data that isn't on the planes
        data = self.get_line(quantity, direction, coord1, coord2)
        x = np.linspace(0, self.period, self.x_samples)
        y = np.linspace(0, self.period, self.y_samples)
        z = np.linspace(0, self.height, self.z_samples)
        if direction == 'x':
            # z along rows, y along columns
            pos_data = x
        elif direction == 'y':
            # x along columns, z along rows
            pos_data = y
        elif direction == 'z':
            # x along rows, y along columns
            pos_data = z
        freq = self.conf['Simulation']['params']['frequency']['value']
        wvlgth = (c.c / freq) * 1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(
            freq, wvlgth)
        labels = ('Z [um]', quantity, title)
        ptype = "%s_line_plot_%i_%i" % (direction, coord1, coord2)
        if np.iscomplexobj(data):
            self.log.warning('Plotting only real component of %s in line plot',
                             quantity)
            data = data.real
        self.line_plot(pos_data, data, ptype, labels)

class SimulationGroup:

    """
    A group of simulations. Takes a list of Simulation objects as an argument.
    This class is not responsible for grouping Simulation objects by some
    criteria, it just expects a list of already grouped Simulation objects.

    Provides methods for calculating things on a group of simulations. The
    results of these methods might not be sensible for some groupings. For
    example, calculating the convergence of a group that is group against
    frequency doesn't make any sense. Similiarly, calculating Jsc of a group
    that is grouped against number of basis terms also doesn't make sense.
    """

    def __init__(self, sims):
        self.sims = sims
        self.log = logging.getLogger('postprocess')
        self.num_sims = len(sims)


    def get_plane(self, scalar, plane, pval):
        """
        Gets data along a 2D plane/slice through the 3D data array for a given
        quantity

        :param str plane: Any of 'xy', 'yz', or 'xz'. Determines the plane
        along which the slice is taken
        :param int pval: The index along the final unspecified direction. If
        plane='xy' then index would index along the z direction.
        :param str quantity: The quantity whose data array you wish to take a
        line cut through
        """

        if plane == 'yz' or plane == 'zy':
            # z along rows, y along columns
            return scalar[:, pval, :]
        elif plane == 'xz' or plane == 'zx':
            # x along columns, z along rows
            return scalar[:, :, pval]
        elif plane == 'xy' or plane == 'yx':
            # x along rows, y along columns
            return scalar[pval, :, :]

    def diff_sq(self, x, y):
        """Returns the magnitude of the difference vector squared between two vector fields at each
        point in space"""
        # Calculate the magnitude of the difference vector SQUARED at each point in space
        # This is mag(vec(x) - vec(y))^2 at each point in space. This should be a 1D array
        # with # of elements = # sampling points
        mag_diff_vec = sum([np.absolute(v1 - v2)**2 for v1, v2 in zip(x, y)])
        return mag_diff_vec

    def get_slice(self, sim):
        """Returns indices for data that strip out air and substrate regions"""
        # TODO: This function is definitely not general. We need to get a list
        # of layers to exclude from the user. For now, just assume we want to
        # exclude the top and bottom regions
        # sorted_layers is an OrderedDict, and thus has the popitem method
        sorted_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        first_layer = sorted_layers.popitem(last=False)
        last_layer = sorted_layers.popitem()
        # We can get the starting and ending planes from their heights
        start_plane = int(round(first_layer[1]['params'][
                          'thickness']['value'] / sim.dz))
        end_plane = int(round(last_layer[1]['params'][
                        'thickness']['value'] / sim.dz))
        return start_plane, end_plane

    def get_comp_vec(self, sim, field, start, end):
        """Returns the comparison vector"""
        # Compare all other sims to our best estimate, which is sim with highest number of
        # basis terms (last in list cuz sorting)

        # Get the proper file extension depending on the field.
        norm = 'norm'+field
        # Get the comparison vector
        vecs = [sim.data[field+comp][start:end] for comp in ('x', 'y', 'z')]
        normvec = sim.get_scalar_quantity('normE')
        normvec = normvec[start:end]**2
        return vecs, normvec

    def local_error(self, field, exclude=False):
        """Computes the average of the local error between the vector fields of two simulations at
        each point in space"""
        self.log.info('Running the local error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        if exclude:
            start, end = self.get_slice(self.sims[0])
            excluded = '_excluded'
        else:
            start = 0
            end = None
            excluded = ''
        base = self.sims[0].conf['General']['results_dir']
        errpath = os.path.join(base, 'localerror_%s%s.dat' % (field, excluded))
        with open(errpath, 'w') as errfile:
            self.log.info('Computing local error for sweep %s', base)
            # Set the reference sim
            ref_sim = self.sims[-1]
            # Get the comparison vector
            vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
            # For all other sims in the groups, compare to best estimate
            # and write to error file
            for i in range(0, self.num_sims - 1):
                sim2 = self.sims[i]
                vecs2, normvec2 = self.get_comp_vec(sim2, field, start, end)
                self.log.info("Computing local error between numbasis %i and numbasis %i",
                              ref_sim.conf['Simulation'][ 'params']['numbasis']['value'],
                              sim2.conf['Simulation']['params']['numbasis']['value'])
                # Get the array containing the magnitude of the difference vector at each point
                # in space
                mag_diff_vec = self.diff_sq(vecs1, vecs2)
                # Normalize the magnitude squared of the difference vector by the magnitude squared of
                # the local electric field of the comparison simulation at
                # each point in space
                if len(mag_diff_vec) != len(normvec):
                    self.log.error("The normalization vector has an incorrect number of elements!!!")
                    raise ValueError
                norm_mag_diff = mag_diff_vec / normvec
                # Compute the average of the normalized magnitude of all
                # the difference vectors
                avg_diffvec_mag = np.sum(norm_mag_diff) / norm_mag_diff.size
                errfile.write('%i,%f\n' % (sim2.conf['Simulation']['params'][
                              'numbasis']['value'], avg_diffvec_mag))
                sim2.clear_data()
            ref_sim.clear_data()

    def global_error(self, field, exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell"""

        self.log.info('Running the global error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        if exclude:
            start, end = self.get_slice(self.sims[0])
            excluded = '_excluded'
        else:
            start = 0
            end = None
            excluded = ''
        # base = self.sims[0].conf['General']['base_dir']
        base = self.sims[0].conf['General']['results_dir']
        errpath = os.path.join(base, 'globalerror_%s%s.dat' % (field, excluded))
        with open(errpath, 'w') as errfile:
            self.log.info('Computing global error for sweep %s', base)
            # Set reference sim
            ref_sim = self.sims[-1]
            # Get the comparison vector
            vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
            # For all other sims in the groups, compare to best estimate
            # and write to error file
            for i in range(0, self.num_sims - 1):
                sim2 = self.sims[i]
                vecs2, normvec2 = self.get_comp_vec(sim2, field, start, end)
                self.log.info("Computing global error between numbasis %i and numbasis %i",
                              ref_sim.conf['Simulation'][ 'params']['numbasis']['value'],
                              sim2.conf['Simulation']['params']['numbasis']['value'])
                # Get the array containing the magnitude of the difference vector at each point
                # in space
                mag_diff_vec = self.diff_sq(vecs1, vecs2)
                # Check for equal lengths between norm array and diff mag
                # array
                if len(mag_diff_vec) != len(normvec):
                    self.log.error( "The normalization vector has an incorrect number of elements!!!")
                    raise ValueError
                # Error as a percentage should be the square root of the ratio of sum of mag diff vec
                # squared to mag efield squared
                error = np.sqrt(np.sum(mag_diff_vec) / np.sum(normvec))
                errfile.write('%i,%f\n' % (sim2.conf['Simulation']['params']['numbasis']['value'], error))
                sim2.clear_data()
            ref_sim.clear_data()

    def adjacent_error(self, field, exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell.
        This computes error between adjacent sims in a sweep through basis terms."""

        self.log.info('Running the adjacent error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        if exclude:
            start, end = self.get_slice(self.sims[0])
            excluded = '_excluded'
        else:
            start = 0
            end = None
            excluded = ''
        base = self.sims[0].conf['General']['results_dir']
        errpath = os.path.join(base, 'adjacenterror_%s%s.dat' % (field, excluded))
        with open(errpath, 'w') as errfile:
            self.log.info('Computing adjacent error for sweep %s', base)
            # For all other sims in the groups, compare to best estimate
            # and write to error file
            for i in range(1, self.num_sims):
                # Set reference sim
                ref_sim = self.sims[i]
                # Get the comparison vector
                vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                sim2 = self.sims[i - 1]
                vecs2, normvec2 = self.get_comp_vec(sim2, field, start, end)
                self.log.info("Computing adjacent error between numbasis %i and numbasis %i",
                              ref_sim.conf['Simulation'][ 'params']['numbasis']['value'],
                              sim2.conf['Simulation']['params']['numbasis']['value'])
                # Get the array containing the magnitude of the difference vector at each point
                # in space
                mag_diff_vec = self.diff_sq(vecs1, vecs2)
                # Check for equal lengths between norm array and diff mag
                # array
                if len(mag_diff_vec) != len(normvec):
                    self.log.error("The normalization vector has an incorrect number of elements!!!")
                    raise ValueError
                # Error as a percentage should be thkkk square root of the ratio of sum of mag diff vec
                # squared to mag efield squared
                error = np.sqrt(np.sum(mag_diff_vec) / np.sum(normvec))
                # self.log.info(str(error))
                errfile.write('%i,%f\n' % (sim2.conf['Simulation']['params']['numbasis']['value'], error))
                sim2.clear_data()
                ref_sim.clear_data()

    def scalar_reduce(self, quantity, avg=False):
        """
        Combine a scalar quantity across all simulations in each group. If
        avg=False then a direct sum is computed, otherwise an average is
        computed
        """

        base = self.sims[0].conf['General']['results_dir']
        self.log.info('Performing scalar reduction for group at %s' % base)
        self.log.debug('QUANTITY: %s'%quantity)
        group_comb = self.sims[0].get_scalar_quantity(quantity)
        self.log.debug(group_comb.dtype)
        self.sims[0].clear_data()
        # This approach is more memory efficient then building a 2D array
        # of all the data from each group and summing along an axis
        for sim in self.sims[1:]:
            self.log.debug(sim.id)
            quant = sim.get_scalar_quantity(quantity)
            self.log.debug(quant.dtype)
            group_comb += quant
            sim.clear_data()
        if avg:
            group_comb = group_comb / self.num_sims
            fname = 'scalar_reduce_avg_%s' % quantity
        else:
            fname = 'scalar_reduce_%s' % quantity
            path = os.path.join(base, fname)
        ftype = self.sims[0].conf['General']['save_as']
        if ftype == 'npz':
            np.save(path, group_comb)
        elif ftype == 'hdf5':
            self.log.warning('FIX HDF5 SCALAR REDUCE SAVING')
            np.save(path, group_comb)
        else:
            raise ValueError('Invalid file type in config')

    def fractional_absorbtion(self, port='Substrate'):
        """
        Computes the fraction of the incident spectrum that is absorbed by
        the device. This is a unitless number, and its interpretation somewhat
        depends on the units you express the incident spectrum in. If you
        expressed your incident spectrum in photon number, this can be
        interpreted as the fraction of incident photons that were absorbed. If
        you expressed your incident spectrum in terms of power per unit area,
        then this can be interpreted as the fraction of incident power per unit
        area that gets absorbed. In summary, its the fraction of whatever you
        put in that is being absorbed by the device.
        """

        base = self.sims[0].conf['General']['results_dir']
        self.log.info('Computing fractional absorbtion for group at %s' % base)
        vals = np.zeros(self.num_sims)
        freqs = np.zeros(self.num_sims)
        wvlgths = np.zeros(self.num_sims)
        spectra = np.zeros(self.num_sims)
        path = self.sims[0].conf['Simulation']['input_power_wv']
        wv_vec, p_vec = np.loadtxt(path, usecols=(0, 2), unpack=True, delimiter=',')
        p_wv = interpolate.interp1d(wv_vec, p_vec, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        # Assuming the sims have been grouped by frequency, sum over all of
        # them
        for i, sim in enumerate(self.sims):
            # Unpack data for the port we passed in as an argument
            ref, trans, absorb = sim.data['transmission_data'][port]
            freq = sim.conf['Simulation']['params']['frequency']['value']
            wvlgth = c.c / freq
            wvlgth_nm = wvlgth * 1e9
            freqs[i] = freq
            wvlgths[i] = wvlgth
            # Get solar power from chosen spectrum
            # Get p at wvlength by interpolation
            sun_pow = p_wv(wvlgth_nm)
            spectra[i] = sun_pow * wvlgth_nm
            vals[i] = absorb * sun_pow * wvlgth_nm
            sim.clear_data()
        # Use Trapezoid rule to perform the integration. Note all the
        # necessary factors of the wavelength have already been included
        # above
        wvlgths = wvlgths[::-1]
        vals = vals[::-1]
        spectra = spectra[::-1]
        integrated_absorbtion = intg.trapz(vals, x=wvlgths * 1e9)
        power = intg.trapz(spectra, x=wvlgths * 1e9)
        # factor of 1/10 to convert A*m^-2 to mA*cm^-2
        #wv_fact = c.e/(c.c*c.h*10)
        #wv_fact = .1
        #Jsc = (Jsc*wv_fact)/power
        frac_absorb = integrated_absorbtion / power
        outf = os.path.join(base, 'fractional_absorbtion.dat')
        with open(outf, 'w') as out:
            out.write('%f\n' % frac_absorb)
        self.log.info('Fractional Absorbtion = %f' % frac_absorb)
        return frac_absorb

    def Jsc(self, port='Substrate'):
        """Computes photocurrent density. This is just the integrated
        absorption scaled by a unitful factor. Assuming perfect carrier
        collection, meaning every incident photon gets converted to 1 collected
        electron, this factor is q/(hbar*c) which converts to a current per
        unit area"""
        base = self.sims[0].conf['General']['results_dir']
        self.log.info('Computing photocurrent density for group at %s' % base)
        vals = np.zeros(self.num_sims)
        freqs = np.zeros(self.num_sims)
        wvlgths = np.zeros(self.num_sims)
        spectra = np.zeros(self.num_sims)
        wv_fact = c.e / (c.c * c.h * 10)
        # Assuming the sims have been grouped by frequency, sum over all of
        # them
        for i, sim in enumerate(self.sims):
            # Unpack data for the port we passed in as an argument
            ref, trans, absorb = sim.data['transmission_data'][port]
            freq = sim.conf['Simulation']['params']['frequency']['value']
            wvlgth = c.c / freq
            wvlgth_nm = wvlgth * 1e9
            freqs[i] = freq
            wvlgths[i] = wvlgth
            # Get solar power from chosen spectrum
            path = sim.conf['Simulation']['input_power_wv']
            wv_vec, p_vec = np.loadtxt(path, usecols=(0, 2), unpack=True, delimiter=',')
            # Get p at wvlength by interpolation
            p_wv = interpolate.interp1d(wv_vec, p_vec, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
            sun_pow = p_wv(wvlgth_nm)
            spectra[i] = sun_pow * wvlgth_nm
            # This is our integrand
            vals[i] = absorb * sun_pow * wvlgth_nm
            # test = absorb * sun_pow * wvlgth_nm * wv_fact * delta_wv
            # self.log.info('Sim %s Jsc Integrand: %f', sim.id, test)
            sim.clear_data()
        # Use Trapezoid rule to perform the integration. Note all the
        # necessary factors of the wavelength have already been included
        # above
        wvlgths = wvlgths[::-1]
        vals = vals[::-1]
        spectra = spectra[::-1]
        integrated_absorbtion = intg.trapz(vals, x=wvlgths)
        # integrated_absorbtion = intg.simps(vals, x=wvlgths)
        # factor of 1/10 to convert A*m^-2 to mA*cm^-2
        Jsc = wv_fact * integrated_absorbtion
        outf = os.path.join(base, 'jsc.dat')
        with open(outf, 'w') as out:
            out.write('%f\n' % Jsc)
        self.log.info('Jsc = %f', Jsc)
        return Jsc

    def Jsc_integrated_persim(self):
        for i, sim in enumerate(self.sims):
            try:
                genRate = sim.data['genRate']
            except FileNotFoundError:
                genRate = sim.genRate()
            # Gen rate in cm^-3. Gotta convert lengths here from um to cm
            z_vals = np.linspace(0, sim.height*1e-4, sim.z_samples)
            x_vals = np.linspace(0, sim.period*1e-4, sim.x_samples)
            y_vals = np.linspace(0, sim.period*1e-4, sim.y_samples)
            z_integral = intg.trapz(genRate, x=z_vals, axis=0)
            x_integral = intg.trapz(z_integral, x=x_vals, axis=0)
            y_integral = intg.trapz(x_integral, x=y_vals, axis=0)
            # z_integral = intg.simps(genRate, x=z_vals, axis=0)
            # x_integral = intg.simps(z_integral, x=x_vals, axis=0)
            # y_integral = intg.simps(x_integral, x=y_vals, axis=0)
            # Convert period to cm and current to mA
            Jsc = 1000*(c.e/(sim.period*1e-4)**2)*y_integral
            self.log.info('Sim %s Jsc Integrate Value: %f', sim.id, Jsc)


    def Jsc_integrated(self):
        """
        Compute te photocurrent density by performing a volume integral of the
        generation rate
        """
        fname = 'scalar_reduce_genRate.npy'
        base = self.sims[0].conf['General']['results_dir']
        self.log.info('Computing integrated Jsc for group at %s', base)
        path = os.path.join(base, fname)
        try:
            genRate = np.load(path)
        except FileNotFoundError:
            self.scalar_reduce('genRate')
            genRate = np.load(path)
        # Gen rate in cm^-3. Gotta convert lengths here from um to cm
        z_vals = np.linspace(0, self.sims[0].height*1e-4, self.sims[0].z_samples)
        x_vals = np.linspace(0, self.sims[0].period*1e-4, self.sims[0].x_samples)
        y_vals = np.linspace(0, self.sims[0].period*1e-4, self.sims[0].y_samples)
        z_integral = intg.trapz(genRate, x=z_vals, axis=0)
        x_integral = intg.trapz(z_integral, x=x_vals, axis=0)
        y_integral = intg.trapz(x_integral, x=y_vals, axis=0)
        # z_integral = intg.simps(genRate, x=z_vals, axis=0)
        # x_integral = intg.simps(z_integral, x=x_vals, axis=0)
        # y_integral = intg.simps(x_integral, x=y_vals, axis=0)
        # Convert period to cm and current to mA
        Jsc = 1000*(c.e/(self.sims[0].period*1e-4)**2)*y_integral
        outf = os.path.join(base, 'jsc_integrated.dat')
        with open(outf, 'w') as out:
            out.write('%f\n' % Jsc)
        self.log.info('Jsc_integrated = %f', Jsc)
        return Jsc

    def weighted_transmissionData(self, port='Substrate'):
        """Computes spectrally weighted absorption,transmission, and reflection"""

        base = self.sims[0].conf['General']['results_dir']
        self.log.info('Computing spectrally weighted transmission data for group at %s' % base)
        abs_vals = np.zeros(self.num_sims)
        ref_vals = np.zeros(self.num_sims)
        trans_vals = np.zeros(self.num_sims)
        freqs = np.zeros(self.num_sims)
        wvlgths = np.zeros(self.num_sims)
        spectra = np.zeros(self.num_sims)
        # Get solar power from chosen spectrum
        path = self.sims[0].conf['Simulation']['input_power_wv']
        wv_vec, p_vec = np.loadtxt(path, usecols=(0, 2), unpack=True, delimiter=',')
        # Get interpolating function for power
        p_wv = interpolate.interp1d(wv_vec, p_vec, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        # Assuming the leaves contain frequency values, sum over all of them
        for i, sim in enumerate(self.sims):
            ref, trans, absorb = sim.data['transmission_data'][port]
            freq = sim.conf['Simulation']['params']['frequency']['value']
            wvlgth = c.c / freq
            wvlgth_nm = wvlgth * 1e9
            freqs[i] = freq
            wvlgths[i] = wvlgth
            sun_pow = p_wv(wvlgth_nm)
            spectra[i] = sun_pow * wvlgth_nm
            abs_vals[i] = sun_pow * absorb * wvlgth_nm
            ref_vals[i] = sun_pow * ref * wvlgth_nm
            trans_vals[i] = sun_pow * trans * wvlgth_nm
        # Now integrate all the weighted spectra and divide by the power of
        # the spectra
        wvlgths = wvlgths[::-1]
        abs_vals = abs_vals[::-1]
        ref_vals = ref_vals[::-1]
        trans_vals = trans_vals[::-1]
        spectra = spectra[::-1]
        power = intg.trapz(spectra, x=wvlgths * 1e9)
        wght_ref = intg.trapz(ref_vals, x=wvlgths * 1e9) / power
        wght_abs = intg.trapz(abs_vals, x=wvlgths * 1e9) / power
        wght_trans = intg.trapz(trans_vals, x=wvlgths) / power
        out = os.path.join(base, 'weighted_transmission_data.dat')
        with open(out, 'w') as outf:
            outf.write('# Reflection, Transmission, Absorbtion\n')
            outf.write('%f,%f,%f' % (wght_ref, wght_trans, wght_abs))
        return wght_ref, wght_trans, wght_abs

    def convergence(self, quantity, err_type='global', scale='linear'):
        """Plots the convergence of a field across all available simulations"""

        self.log.info('Plotting convergence')
        base = self.sims[0].conf['General']['base_dir']
        if err_type == 'local':
            fglob = os.path.join(base, 'localerror_%s*.dat' % quantity)
        elif err_type == 'global':
            fglob = os.path.join(base, 'globalerror_%s*.dat' % quantity)
        elif err_type == 'adjacent':
            fglob = os.path.join(base, 'adjacenterror_%s*.dat' % quantity)
        else:
            self.log.error('Attempting to plot an unsupported error type')
            raise ValueError
        paths = glob.glob(fglob)
        for path in paths:
            labels = []
            errors = []
            with open(path, 'r') as datf:
                for line in datf.readlines():
                    lab, err = line.split(',')
                    labels.append(lab)
                    errors.append(err)
            fig = plt.figure(figsize=(9, 7))
            plt.ylabel('M.S.E of %s' % quantity)
            plt.xlabel('Number of Fourier Terms')
            plt.plot(labels, errors)
            plt.yscale(scale)
            # plt.xticks(x,labels,rotation='vertical')
            plt.tight_layout()
            plt.title(os.path.basename(base))
            if self.gconf['General']['save_plots']:
                if '_excluded' in path:
                    excluded = '_excluded'
                else:
                    excluded = ''
                name = '%s_%sconvergence_%s%s.pdf' % (
                    os.path.basename(base), err_type, quantity, excluded)
                path = os.path.join(base, name)
                fig.savefig(path)
            if self.gconf['General']['show_plots']:
                plt.show()
            plt.close(fig)

    def plot_scalar_reduce(self, quantity, plane, pval, draw=False, fixed=None):
        """Plot the result of a particular scalar reduction for each group"""

        sim = self.sims[0]
        base = sim.conf['General']['results_dir']
        self.log.info('Plotting scalar reduction of %s for quantity %s' % (base, quantity))
        cm = plt.get_cmap('jet')
        max_depth = sim.conf['Simulation']['max_depth']
        period = sim.conf['Simulation']['params']['array_period']['value']
        x = np.arange(0, period, sim.dx)
        y = np.arange(0, period, sim.dy)
        z = np.arange(0, max_depth + sim.dz, sim.dz)
        ftype = sim.conf['General']['save_as']
        if ftype == 'npz':
            globstr = os.path.join(base, 'scalar_reduce*_%s.npy' % quantity)
            files = glob.glob(globstr)
        elif ftype == 'hdf5':
            self.log.warning('FIX LOAD IN GLOBAL SCALAR REDUCE')
            globstr = os.path.join(base, 'scalar_reduce*_%s.npy' % quantity)
            files = glob.glob(globstr)
        else:
            raise ValueError('Incorrect file type in config')
        title = 'Reduction of %s' % quantity
        for datfile in files:
            p = False
            if ftype == 'npz':
                scalar = np.load(datfile)
            elif ftype == 'hdf5':
                self.log.warning('FIX LOAD IN GLOBAL SCALAR REDUCE')
                scalar = np.load(datfile)
            else:
                raise ValueError('Incorrect file type in config')
            cs = self.get_plane(scalar, plane, pval)
            if plane == 'yz' or plane == 'zy':
                labels = ('y [um]', 'z [um]', quantity, title)
                if sim.conf['General']['save_plots']:
                    fname = 'scalar_reduce_%s_plane_2d_yz.pdf' % quantity
                    p = os.path.join(base, fname)
                show = sim.conf['General']['show_plots']
                self.sims[0].heatmap2d(y, z, cs, labels, plane, pval,
                               save_path=p, show=show, draw=draw, fixed=fixed)
            elif plane == 'xz' or plane == 'zx':
                labels = ('x [um]', 'z [um]', quantity, title)
                if sim.conf['General']['save_plots']:
                    fname = 'scalar_reduce_%s_plane_2d_xz.pdf' % quantity
                    p = os.path.join(base, fname)
                show = sim.conf['General']['show_plots']
                self.sims[0].heatmap2d(sim, x, z, cs, labels, plane, pval,
                               save_path=p, show=show, draw=draw, fixed=fixed)
            elif plane == 'xy' or plane == 'yx':
                labels = ('y [um]', 'x [um]', quantity, title)
                if sim.conf['General']['save_plots']:
                    fname = 'scalar_reduce_%s_plane_2d_xy.pdf' % quantity
                    p = os.path.join(base, fname)
                self.sims[0].heatmap2d(sim, x, y, cs, labels, plane, pval,
                               save_path=p, show=show, draw=draw, fixed=fixed)

    def transmission_data(self, absorbance, reflectance, transmission, port='Substrate'):
        """Plot transmissions, absorption, and reflectance assuming leaves are frequency"""

        base = self.sims[0].conf['General']['results_dir']
        self.log.info('Plotting transmission data for group at %s' % base)
        # Assuming the leaves contain frequency values, sum over all of them
        freqs = np.zeros(self.num_sims)
        refl_l = np.zeros(self.num_sims)
        trans_l = np.zeros(self.num_sims)
        absorb_l = np.zeros(self.num_sims)
        for i, sim in enumerate(self.sims):
            # Unpack data for the port we passed in as an argument
            ref, trans, absorb = sim.data['transmission_data'][port]
            freq = sim.conf['Simulation']['params']['frequency']['value']
            freqs[i] = freq
            trans_l[i] = trans
            refl_l[i] = ref
            absorb_l[i] = absorb
        freqs = (c.c / freqs[::-1]) * 1e9
        refl_l = refl_l[::-1]
        absorb_l = absorb_l[::-1]
        trans_l = trans_l[::-1]
        plt.figure()
        if absorbance:
            self.log.info('Plotting absorbance')
            plt.plot(freqs, absorb_l, '-o', label='Absorption')
        if reflectance:
            plt.plot(freqs, refl_l, '-o', label='Reflection')
        if transmission:
            plt.plot(freqs, trans_l, '-o', label='Transmission')
        plt.legend(loc='best')
        figp = os.path.join(base, 'transmission_plots_port%s.pdf'%port)
        plt.xlabel('Wavelength (nm)')
        plt.ylim((0, 1.0))
        plt.savefig(figp)
        plt.close()
