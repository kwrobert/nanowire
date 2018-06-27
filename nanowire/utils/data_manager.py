import os
import warnings
import posixpath
import logging
import pint
import numpy as np
import tables as tb
from tables.file import _checkfilters
from tables.parameters import EXPECTED_ROWS_TABLE
from tables.utils import lazyattr
from tables import tableextension
from collections import MutableMapping
from abc import ABCMeta, abstractmethod
from nanowire.utils.utils import (
    open_atomic,
    ureg,
    Q_,
    add_row
)


class UnitRowWrapper:
    """
    A wrapper around a PyTables :py:class:`tableextension.Row` object that
    applies units wherever necessary and delegates any functions it hasn't
    overridden to the wrapped tableextension.Row object. The reason we can't
    just subclass a Row object is because there are many places in the Table
    code that have hardcoded calls to tableextension.Row, which would all need
    to be entirely rewritten (not just overridden and wrapped, like with the
    read methods) to use the new subclass. Its easier to just override all
    functions that return a row in a Table subclass, and everywhere a
    tableextension.Row would be returned return this object instead.
    """
    def __init__(self, row, units, unit_dtype):
        self._row = row
        self.units = units
        self.unit_dtype = unit_dtype

    def __getattr__(self, attr):
        return getattr(self._row, attr)

    def __getitem__(self, key):
        item = self._row.__getitem__(key)
        if key in self.units:
            return Q_(item, self.units[key])
        else:
            return item

    def __setitem__(self, key, val):
        if isinstance(val, pint.quantity._Quantity):
            print("Setting a pint quantity in UnitRowWrapper")
            if key not in self.units:
                msg = 'Attempting to store a pint Quantity not in the ' \
                      'self.units dict'
                raise ValueError(msg)
            self._row.__setitem__(key, val.magnitude)
        else:
            self._row.__setitem__(key, val)

    def fetch_all_fields(self):
        fields = self._row.fetch_all_fields()
        print('Fields: {}'.format(fields))
        print('type Fields: {}'.format(type(fields)))
        print('dtype Fields: {}'.format(fields.dtype))
        fields = fields.astype(self.unit_dtype)
        print('new dtype Fields: {}'.format(fields.dtype))
        for k, unit in self.units.items():
            print('k: ', k)
            print('unit: ', unit)
            q = Q_(fields[k], unit)
            print('Quantity: {}'.format(q))
            fields[k] = q
        return fields


class UnitTable(tb.Table):
    """
    A PyTables table that has an attribute for storing unit metadata. Any
    column (may NOT be nested) with units will automatically be deserialized to
    a pint Quantity object
    """
    _c_classid = 'UnitTable'

    # def __init__(self, parentNode, name, description=None,
    #              title="", filters=None,
    #              expectedrows=EXPECTED_ROWS_TABLE,
    #              chunkshape=None, byteorder=None, _log=True, units=None):
    def __init__(self, *args, sample_record=None, units=None,
                 type_mapping=None, description=None, **kwargs):
        # super(UnitTable, self).__init__(parentNode, name,
        #                                 description=description, title=title,
        #                                 filters=filters,
        #                                 expectedrows=expectedrows,
        #                                 chunkshape=chunkshape, byteorder=byteorder,
        #                                 _log=_log)

        print('Received descr: {}'.format(description))
        type_mapping = type_mapping if type_mapping is not None else {}
        # The user is instantiating a table from scratch (i.e not loading from
        # disk) and passed in a sample record from which we can derive the
        # units and a converted description without object types
        if sample_record is not None:
            units, descr, had_objects = self._f_get_descr_from_sample(sample_record)
        # We received a description either by loading it from disk or because
        # the user passed it to us
        else:
            # If we have a valid object-free description this function will
            # pass it through unchanged. This means any dtype loaded from disk
            # will not be modified, and had_objects will be falsey (an empty
            # set). This will raise an error if we find any object types in
            # description that we cannot map to a writable type using
            # type_mapping
            descr, had_objects = self._f_convert_descr(description,
                                                       type_mapping)
        # At the point we have a converted, object-free description which we
        # can pass to a normal Table.__init__
        print('Descr to super __init__: {}'.format(descr))
        print('type(Descr) to super __init__: {}'.format(type(descr)))

        super(UnitTable, self).__init__(*args, description=descr,
                                        **kwargs)
        # This should be totally possible to handle, but when writing this
        # I didn't have time to write all the recursion and validation code
        if self.description._v_is_nested:
            raise ValueError('Cannot make a nested UnitTable')
        if units is None:
            try:
                units = self.attrs['units']
            except KeyError:
                raise ValueError('Must specify a dictionary of units')
        # had_object should only be true if the user passed in a data type with
        # objects when instantiating a table from scratch. Object types should
        # never exist in a dtype loaded from disk. Therefore, we need to make
        # sure the units dict specifies a unit for each object type that exists
        # in the original description
        if had_objects:
            unit_keys_set = set(units.keys())
            # NOTE: It matters which object you choose to call the difference
            # method on. If all fields we found objects in are accounted for by
            # the keys in units, all is well. If we have more units than we
            # have objects, thats fine. Maybe the user just wants to assign
            # units to some simple types
            diff = had_objects.difference(unit_keys_set)
            if diff:
                msg = 'Found objects in fields {} without corresponding ' \
                      'units'.format(diff)
                raise ValueError(msg)
        print('After super')
        self.units = units
        self.attrs['units'] = units

        print('UnitTable attrs: ', self.attrs)
        print('UnitTable attr names: ', self.attrs._v_attrnames)
        print('UnitTable units: ', self.units)
        # print('UnitTable units: ', self.attrs['units'])
        self.unit_dtype = self._f_build_unit_dtype()
        self.vector_Q = np.vectorize(Q_, otypes=[np.object])

    def _f_get_descr_from_sample(self, sample):
        """
        Get a valid PyTables description from a single sample record
        """
        new_dtype_struct = []
        units = {}
        had_objects = set()
        print('Orig Sample dtype: {}'.format(sample.dtype))
        for field, descr in sample.dtype.fields.items():
            el = sample[field]
            el_dtype = descr[0]
            if el_dtype == np.object_ or el_dtype == np.object:
                if not isinstance(el, pint.quantity._Quantity):
                    msg = 'Cannot save arbitrary objects. All objects must ' \
                          'be pint Quantities'
                    raise ValueError(msg)
                units[field] = str(el.units)
                new_dtype_struct.append((field, np.dtype(type(el.magnitude))))
                had_objects.add(field)
            else:
                new_dtype_struct.append((field, el_dtype))
        new_dtype = np.dtype(new_dtype_struct)
        print('New Sample dtype: {}'.format(new_dtype))
        print('Sample units: {}'.format(units))
        return units, new_dtype, had_objects

    def _f_convert_descr(self, descr, type_mapping):
        """
        Return a new numpy dtype with all fields of object type converted to
        the type stored in type_mapping[field]

        Parameters
        ----------

        dtype : np.dtype, tables.IsDescription, or dict
            Either a numpy dtype or something that can be converted to a numpy
            dtype using table.description.dtype_from_descr that may or may not
            contain object types
        type_mapping : dict
            A dict that describes how to replace any object types we find.  The
            keys are the fields names as strings present in `dtype`. The values
            are the writable data types those fields should map to.

        Returns
        -------
        np.dtype
            A new numpy dtype with all fields containing objects replaced with
            the type in type_mapping[field]
        had_objects
            A set containing the names of all the fields that were formerly
            objects
        """
        if descr is None and not type_mapping:
            print('Got description is none!')
            return descr, False
        if not isinstance(descr, np.dtype):
            dtype = tb.description.dtype_from_descr(descr)
        else:
            dtype = descr
        new_dtype = []
        had_objects = set()
        for field, descr in dtype.fields.items():
            fdtype = descr[0]
            if fdtype.dtype == np.object_ or fdtype.dtype == np.object:
                if field not in type_mapping:
                    msg = 'Found object type in field {} without a type to ' \
                          'map to'.format(field)
                    raise ValueError(msg)
                new_dtype.append((field, type_mapping[field]))
                had_objects.add(field)
            else:
                new_dtype.append((field, fdtype))
        return new_dtype, had_objects

    def _f_build_unit_dtype(self):
        """
        Builds a new dtype for any records that converts fields with units to
        numpy object types so we can store pint Quantities wherever we have
        units
        """
        new_descr = []
        for field, dtype in self.dtype.descr:
            unit = self.units.get(field, None)
            if unit is not None:
                new_descr.append((field, 'O'))
            else:
                new_descr.append((field, dtype))
        return np.dtype(new_descr)

    def _apply_units(self, data):
        print("UnitTable data: ", data)
        print("UnitTable units: ", self.units)
        print(type(data))
        units_data = data.astype(self.unit_dtype)
        for field, unit in self.units.items():
            units_data[field] = self.vector_Q(units_data[field], unit)
        print('Unit data: ', units_data)
        return units_data

    @lazyattr
    def row(self):
        row = tableextension.Row(self)
        return UnitRowWrapper(row, self.units, self.unit_dtype)

    def read(self, *args, **kwargs):
        data = tb.Table.read(self, *args, **kwargs)
        return self._apply_units(data)

    def read_coordinates(self, *args, **kwargs):
        data = tb.Table.read_coordinates(self, *args, **kwargs)
        return self._apply_units(data)

    def read_sorted(self, *args, **kwargs):
        data = tb.Table.read_sorted(self, *args, **kwargs)
        return self._apply_units(data)

    def read_where(self, *args, **kwargs):
        data = tb.Table.read_sorted(self, *args, **kwargs)
        return self._apply_units(data)

    def iterrows(self, *args, **kwargs):
        for row in tb.Table.iterrows(self, *args, **kwargs):
            print('Orig row: {}'.format(row))
            print('Type of orig row: {}'.format(type(row)))
            wrapped_row = UnitRowWrapper(row, self.units, self.unit_dtype)
            print('Wrapped row: {}'.format(wrapped_row))
            print('Type of wrapped row: {}'.format(type(wrapped_row)))
            yield wrapped_row

    def itersequence(self, *args, **kwargs):
        for row in tb.Table.itersequence(self, *args, **kwargs):
            print('Orig row: {}'.format(row))
            print('Type of orig row: {}'.format(type(row)))
            wrapped_row = UnitRowWrapper(row, self.units, self.unit_dtype)
            print('Wrapped row: {}'.format(wrapped_row))
            print('Type of wrapped row: {}'.format(type(wrapped_row)))
            yield wrapped_row

    def itersorted(self, *args, **kwargs):
        for row in tb.Table.itersorted(self, *args, **kwargs):
            print('Orig row: {}'.format(row))
            print('Type of orig row: {}'.format(type(row)))
            wrapped_row = UnitRowWrapper(row, self.units, self.unit_dtype)
            print('Wrapped row: {}'.format(wrapped_row))
            print('Type of wrapped row: {}'.format(type(wrapped_row)))
            yield wrapped_row


def create_unit_table(self, where, name, title="",
                      filters=None, expectedrows=10000,
                      chunkshape=None, byteorder=None,
                      createparents=False, obj=None, description=None,
                      units=None, sample_record=None, type_mapping=None):
    parentNode = self._get_or_create_path(where, createparents)

    _checkfilters(filters)
    return UnitTable(parentNode, name, sample_record=sample_record,
                     units=units, type_mapping=type_mapping,
                     description=description,
                     title=title, filters=filters,
                     expectedrows=expectedrows,
                     chunkshape=chunkshape, byteorder=byteorder)


tb.File.create_unit_table = create_unit_table

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

    def __init__(self, store_path, logger=None):
        self._data = {}
        self._avgs = {}
        self._updated = {}
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger
        self.store_path = store_path
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
    def open_dstore(self):
        """
        Open the data store located at the path self.store_path for reading and
        writing. Subclasses must implement their own logic for opening their
        particular data store
        """

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
    # DEFAULT_FILTER = tb.Filters(complevel=8, complib='zlib')
    DEFAULT_FILTER = None

    def __init__(self, store_path, group_path='/', filt=None, logger=None):
        """
        :param :class:`~utils.config.Config`: Config object for the simulation
        that this DataManager will be managing data for
        :param logger: A logger object
        """

        super(HDF5DataManager, self).__init__(store_path, logger=logger)
        self.open_dstore()
        self.gpath = group_path
        self.filt = filt if filt is not None else self.DEFAULT_FILTER
        try:
            self.gobj = self._dstore.get_node(self.gpath, classname='Group')
        except tb.NoSuchNodeError:
            self.gobj = self._dstore.create_group('/', self.gpath[1:],
                                                  filters=self.filt)
        self._update_keys()
        self._updated = {key: False for key in self._data.keys()}
        self.writable = {np.ndarray: self.write_numpy_array,
                         str: self.write_string,
                         float: self.write_float,
                         pint.quantity._Quantity: self.write_pint_quantity}

    def _get_writer(self, obj):
        for klass, writer in self.writable.items():
            # Structured arrays need special handling. An object of type
            # np.ndarray can still be a structured array. A homogenous array
            # will always have dtype.field == None
            if isinstance(obj, klass):
                return writer
        else:
            msg = 'No writer available for class {}'.format(obj.__class__)
            raise ValueError(msg)

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
        # Returns only user set attributes, and none of the special system
        # attributes used internally by pytables
        for k in self.gobj._v_attrs._f_list(attrset='user'):
            if clear:
                self._data[k] = None
                self._updated[k] = False
            else:
                if k not in self._data:
                    self._data[k] = None
                    self._updated[k] = False

    def _check_equal(self, key, value):
        # np.array_equal is necessary in case we are dealing with numpy arrays
        # Elementwise comparison of arrays of different shape throws a
        # deprecation warning, and array_equal works on dicts and lists
        return np.array_equal(self._data[key], value)

    def _load_data(self, key):
        """
        Implements logic for loading data when the user asks for it by
        accessing the relevant key. This is only called from __getitem__
        """

        nodepath = posixpath.join(self.gpath, key)
        try:
            node = self._dstore.get_node(nodepath)
            if isinstance(node, (tb.Table, UnitTable)):
                # return node.read().view(np.recarray)
                # return node.read()
                self._data[key] = node.read()
                return
            else:
                data = node.read()
                # self._data[key] = node.read()
                try:
                    units = node.attrs['units']
                    self._data[key] = Q_(data, units)
                except KeyError:
                    self._data[key] = data
                return
        except tb.NoSuchNodeError:
            pass
        # Check the attributes of the group
        try:
            self.log.info('The node you requested does not exist in the '
                          'HDF5 file, checking group attributes')
            self._data[key] = self.gobj._v_attrs[key]
        except KeyError as e:
            msg = 'The nodes or attributes exist with name {}'.format(key)
            self.log.error(msg)
            e.args = e.args + (msg,)
            raise

    def open_dstore(self):
        self._dstore = tb.open_file(self.store_path, 'a')

    def write_struct_array(self, recarr, name):
        """
        Write a structured numpy array to the underlying HDF5 datastore

        Write a record array to a PyTables Table leaf node under group
        `self.gpath` with name `name`. If a Table node does not already exist
        directly beneath `self.gpath` , it is created automatically. If a Table
        node does already exist, it is removed and a new table is written in
        its place.
        """

        # Saving homogenous arrays to Tables is silly but not an error
        if recarr.dtype.fields is None:
            self.log.warning('Saving homogenous array to PyTables Table')
            warnings.warn('Use write_ndarray to save homogenous arrays!')
        self.log.info('Writing data for structured array %s', name)
        num_rows = recarr.shape[0]
        has_quants = any([isinstance(el, pint.quantity._Quantity) for el in
                          recarr[0]])
        try:
            # If the table exists, clear it out
            self._dstore.remove_node(self.gpath, name=name)
        except tb.NoSuchNodeError:
            pass
        if has_quants:
            table = self._dstore.create_unit_table(self.gpath, name,
                                                   sample_record=recarr[0],
                                                   expectedrows=num_rows,
                                                   filters=self.filt)
        else:
            table = self._dstore.create_table(self.gpath, name,
                                              description=recarr.dtype,
                                              expectedrows=num_rows,
                                              filters=self.filt)
        row = table.row
        fields = recarr.dtype.names
        for record in recarr:
            for (i, el) in enumerate(record):
                row[fields[i]] = el
            row.append()
        table.flush()
        self._update_keys()
        return table

    def write_ndarray(self, arr, name):
        """
        Write an ndarray of homogenous datatype to the underlying HDF5
        datastore

        Write an array to a PyTables CArray leaf node under group `self.gpath`
        with name `name`. If a CArray node does not already exist directly
        beneath `self.gpath`, it is created automatically. If a CArray node
        does already exist and has the same shape as `arr`, its data is
        overwritten with the data from `arr`. If a CArray exists but it's shape
        is not compatiable with `arr`, it is removed and a new table is written
        in its place.
        """
        if arr.dtype.fields is not None:
            raise ValueError('Can only save arrays of homogenous dtype using '
                             'this function. Try write_struct_array instead')
        self.log.info('Writing data for array %s', name)
        try:
            # If compatible array exists, try overwriting its data
            node = self._dstore.get_node(self.gpath, name=name)
            if node.shape == arr.shape:
                node[...] = arr
                return node
            # Otherwise remove it and write a new one
            self._dstore.remove_node(self.gpath, name=name)
            node = self._dstore.create_carray(self.gpath, name, obj=arr,
                                              filters=self.filt,
                                              atom=tb.Atom.from_dtype(arr.dtype))
        except tb.NoSuchNodeError:
            node = self._dstore.create_carray(self.gpath, name, obj=arr,
                                              filters=self.filt,
                                              atom=tb.Atom.from_dtype(arr.dtype))
        self._update_keys()
        return node

    def write_numpy_array(self, arr, name):
        """
        Convenience function for saving arbitrary numpy arrays, be they
        structured or unstructured. Just calls the appropriate method depending
        on the value of arr.dtype.fields
        """

        if arr.dtype.fields is None:
            return self.write_ndarray(arr, name)
        else:
            return self.write_struct_array(arr, name)

    def write_string(self, s, key):
        """
        Save a single string to the HDF5 file

        Saves a string `s` to an attribute of `self.gpath`
        """

        node = self._dstore.get_node(self.gpath)
        node._v_attrs[key] = s
        self._update_keys()
        return node

    def write_float(self, f, key):
        """
        Save a single float to the HDF5 file

        Saves a float `f` to an attribute of `self.gpath`
        """

        node = self._dstore.get_node(self.gpath)
        node._v_attrs[key] = f
        self._update_keys()
        return node

    def write_pint_quantity(self, q, key):
        """
        Write a pint Quantity object to the HDF5 file. A Quantity object can
        wrap literally *any* python type, so we have to check for the type of
        magnitude and call the appropriate writer here
        """
        mag = q.magnitude
        # We write arrays out as usual, then just set an attribute on the node
        # that identifies its units
        if isinstance(mag, np.ndarray):
            node = self.write_numpy_array(q.magnitude, key)
            # Only leaves have the .attrs attribute
            node.attrs['units'] = str(q.units)
        else:
            # PyTables can save arbitrary python objects as attributes and
            # reload them transparently, as long as they are pickleable
            node = self._dstore.get_node(self.gpath)
            node._v_attrs[key] = q
        self._update_keys()

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
            writer = self._get_writer(obj)
            writer(obj, key)
            # Flush to disk after every write operation, HDF5 has no journaling
            self._dstore.flush()
            if clear:
                del obj
                del self._data[key]

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

        super(NPZDataManager, self).__init__(conf, logger)
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
