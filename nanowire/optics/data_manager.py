import os
import numpy as np
import tables as tb
from collections import MutableMapping
from .utils.utils import open_atomic


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