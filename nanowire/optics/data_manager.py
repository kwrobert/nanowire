import os
import posixpath
import numpy as np
import tables as tb
from collections import MutableMapping
from abc import ABCMeta, abstractmethod
from .utils.utils import open_atomic


class TransmissionData(tb.IsDescription):
    layer = tb.StringCol(60, pos=0)
    reflection = tb.Float32Col(pos=1)
    transmission = tb.Float32Col(pos=2)
    absorption = tb.Float32Col(pos=3)


class DataManager(MutableMapping, metaclass=ABCMeta):
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

    def __init__(self, conf, log=None):
        self._data = {}
        self._avgs = {}
        self._updated = {}
        self.conf = conf
        if log is None:
            self.log = print
        else:
            self.log = log
        self._dstore = None
        self._blacklist = set()

    def add_to_blacklist(self, key):
        """
        Add a key to the blacklist so it never gets written to disk
        """
        assert(isinstance(key, str))
        self._blacklist.add(key)

    def remove_from_blacklist(self, key):
        """
        Remove a key to the blacklist so it will get written to disk on next
        write_data call
        """
        assert(isinstance(key, str))
        self._blacklist.remove(key)


    @abstractmethod
    def write_data(self):
        """
        Write in-memory data out to the underlying data store
        """

    @abstractmethod
    def close(self):
        """
        Close the underlying data store object and prevent further access to it
        """

    @abstractmethod
    def _load_data(self, key):
        """
        Because this class provides a dict-like interface, clients should never
        have to call this method directly
        """

    @abstractmethod
    def _update_keys(self):
        """
        Pull in all the keys for the data items existing in the underlying data
        store, without actually pulling the data items into memory
        """

    def _check_equal(self, key, value):
        """
        Check the equality of a given value and the object located at
        self._data[key]. Returns True if equal, False if not.

        This is used to determine if the object located at self._data[key]
        needs to be updated and written to disk when the write_data() is
        called.  It is intended to be overwridden

        :param key str: The key for the object in self._data you would like to
        check for equality
        :param value: The value you wish to compare to. Could have arbitrary
        type depending on the subclass
        :rtype: bool
        """
        return self._data[key] == value

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

        try:
            unchanged = self._check_equal(key, value)
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
        base_dir = os.path.expandvars(self.conf['General']['base_dir'])
        sim_dir = os.path.expandvars(self.conf['General']['sim_dir'])
        path = os.path.join(base_dir, sim_dir, 'sim.hdf5')
        # path = os.path.join(sim_dir, 'sim.hdf5')
        self._dstore = tb.open_file(path, 'a')
        ID = os.path.basename(self.conf['General']['sim_dir'])
        self.gpath = '/sim_{}'.format(ID)
        try:
            self.gobj = self._dstore.get_node(self.gpath, classname='Group')
        except tb.NoSuchNodeError:
            self.gobj = self._dstore.create_group('/', self.gpath[1:])
        self._update_keys()
        self._updated = {key:False for key in self._data.keys()}

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

    def _check_equal(self, key, value):
        # np.array_equal is necessary in case we are dealing with numpy arrays
        # Elementwise comparison of arrays of different shape throws a
        # deprecation warning, and array_equal works on dicts and lists
        return np.array_equal(self._data[key], value)

    def _load_data(self, key):
        """
        Implements logic for loading data when the user asks for it by
        accessing the revelant key. This is only called from __getitem__
        """

        nodepath = os.path.join(self.gpath, key)
        try:
            node = self._dstore.get_node(nodepath)
        except tb.NoSuchNodeError:
            self.log.error('The node you requested does not exist in the'
                           ' HDF5 file')
            raise
        if isinstance(node, tb.Table):
            self._data[key] = node.read().view(np.recarray)
        else:
            self._data[key] = node.read()

    def write_data(self, blacklist=('normE', 'normEsquared'), clear=False):
        """
        Writes all necessary data out to the HDF5 file

        :param blacklist: A tuple of strings indicating keys that should not be
        saved to disk
        :type blacklist: tuple
        :param clear: Clear data after writing to free memory. Default: False
        :type clear: bool
        """

        blacklist = set(blacklist).union(self._blacklist)
        self.log.info('Beginning HDF5 data writing procedure')
        # Filter out the original data so we don't resave it
        keys = [key for key in self._data.keys() if key not in blacklist and
                self._updated[key]]
        self.log.info(keys)
        for key in keys:
            obj = self._data[key]
            # Check for recarry first cuz it is a subclass of ndarray
            if isinstance(obj, np.recarray):
                self.log.info('Writing data for recarray %s', key)
                num_rows = obj.shape[0]
                try:
                    # If the table exists, clear it out
                    self._dstore.remove_node(self.gpath, name=key)
                    table = self._dstore.create_table(self.gpath, key,
                                                     description=obj.dtype,
                                                     expectedrows=num_rows)
                except tb.NoSuchNodeError:
                    table = self._dstore.create_table(self.gpath, key,
                                                     description=obj.dtype,
                                                     expectedrows=num_rows)
                row = table.row
                fields = obj.dtype.names
                for record in obj:
                    for (i, el) in enumerate(record):
                        row[fields[i]] = el
                    row.append()
                table.flush()
            elif isinstance(obj, np.ndarray):
                self.log.info('Writing data for array %s', key)
                try:
                    existing_arr = self._dstore.get_node(self.gpath, name=key)
                    if existing_arr.shape == obj.shape:
                        existing_arr[...] = obj
                    else:
                        self._dstore.remove_node(self.gpath, name=key)
                        if self.conf['General']['compression']:
                            filt = tb.Filters(complevel=8, complib='zlib')
                            self._dstore.create_carray(self.gpath, key, obj=obj,
                                                      filters=filt,
                                                      atom=tb.Atom.from_dtype(obj.dtype))
                        else:
                            self._dstore.create_array(self.gpath, key, obj)
                except tb.NoSuchNodeError:
                    if self.conf['General']['compression']:
                        filt = tb.Filters(complevel=8, complib='zlib')
                        self._dstore.create_carray(self.gpath, key, obj=obj,
                                                  filters=filt,
                                                  atom=tb.Atom.from_dtype(obj.dtype))
                    else:
                        self._dstore.create_array(self.gpath, key, obj)
            if clear:
                del obj
                del self._data[key]
        self._dstore.flush()

    def close(self):
        self._update_keys(clear=True)
        self._dstore.close()

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
        self._dstore = np.load(path)

    def _update_keys(self, clear=False):
        """
        Used to pull in keys for all the data items this simulation has stored
        on disk, without loading the actual items
        """
        for key in self._dstore.files:
            if clear:
                self._data[key] = None
            else:
                if key not in self._data:
                    self._data[key] = None

    def _load_data(self, key):
        """
        Actually pulls data from disk out of the _dstore NPZ archive for the
        requested key and puts it in the self._data dict for later retrieval
        """

        if key == 'fluxes' or key == 'transmission_data':
            if key in self._dstore:
                # We have do so some weird stuff here to unpack the
                # dictionaries because np.savez sticks them in a 0D array for
                # some reason
                self._data[key] == self._dstore[key][()]
        else:
            self._data[key] = self._dstore[key]

    def write_data(self):
        """
        Writes all the data in the _data dict to disk. Unfortunately numpy npz
        archives don't support setting individual items in the NPZArchive
        object (i.e _dstore) and only writing the changes, so if any data key
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
