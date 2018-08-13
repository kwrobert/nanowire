import os
import posixpath
import copy
import re
import logging
import json
import pint
import yaml
import pickle
import warnings
import unqlite
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import tables as tb
from boltons.iterutils import remap, default_enter
from dicthash import generate_hash_from_dict
from collections import MutableMapping, OrderedDict, ItemsView
from nanowire.utils.utils import (
    remove_fields,
    numpy_arr_to_dict,
    Q_,
    cmp_dicts,
    find_keypaths,
    decode_bytes,
    make_filter,
)
from nanowire.utils.logging import add_logger

log = logging.getLogger(__name__)

def get_env_variable(match):
    """Given re module match object, return environment variable matching the
    string that was matched"""
    try:
        # Group 1 excludes the dollar sign
        res = os.environ[match.group(1)]
    except KeyError:
        raise KeyError('Environment variable %s does not exist' %
                       match.group(1))
    return res


def splitall(path):
    """
    Get each individual component in a path separated by forward slashes.
    Relative paths are normalized first.

    :param path: A slash separated path /like/this/../here
    :type path: str

    :return: A list of all the parts in the path
    :rtype: list
    """

    path = posixpath.normpath(path.strip('/'))
    allparts = []
    while True:
        parts = posixpath.split(path)
        if parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


class Config(MutableMapping):
    def __init__(self, *args, skip_keys=None, **kwargs):
        self._d = dict(*args, **kwargs)
        self.skip_keys = skip_keys if skip_keys is not None else []
        self.ID = self.gen_id()

    def __getitem__(self, k):
        """
        This override allows us to get a nested value in a dict using a forward
        slash separated string
        """

        parts = self._get_parts(k)
        val = self.getfromseq(parts)
        return val

    def __setitem__(self, k, v):
        """
        This setup allows setting a value inside a nested dictionary using
        either a slash separated string or a list/tuple with the usual []
        operator.
        """

        parts = self._get_parts(k)
        self.setfromseq(parts, v)
        self._update_id()

    def __delitem__(self, k):
        parts = self._get_parts(k)
        self.delfromseq(parts)
        self._update_id()

    def _get_parts(self, k):
        if isinstance(k, str):
            parts = splitall(k)

        elif isinstance(k, (tuple, list)):
            parts = k
        else:
            raise ValueError('Key must be a slash separated string, list, '
                             'or tuple')
        return parts

    def getfromseq(self, keyset):
        """
        A convenience method to get the section of the config file located
        at the end of the sequence of keys
        """

        try:
            ret = self._d
            for key in keyset:
                ret = ret[key]
        except KeyError as ex:
            key = ex.args[0]
            msg = 'Key {} missing from path {}'.format(key, keyset)
            ex.args = (msg,)
            raise
        return ret

    def setfromseq(self, keyset, value):
        """
        A convenience method to set the a value in the config given a sequence
        of keys
        """
        sect = self._d
        for key in keyset[:-1]:
            if key in sect:
                sect = sect[key]
            else:
                sect[key] = {}
                sect = sect[key]
        sect[keyset[-1]] = value

    def delfromseq(self, keyset):
        """
        Deletes the section of the config located at the end of a sequence
        of keys
        """
        del self.getfromseq(keyset[:-1])[keyset[-1]]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __str__(self):
        return str(self._d)

    def __repr__(self):
        return repr(self._d)

    def gen_id(self):
        # Just return some simple string for an empty dict
        if not self._d:
            return 'empty'
        if self.skip_keys:
            return generate_hash_from_dict(self._d, blacklist=self.skip_keys)
        else:
            return generate_hash_from_dict(self._d)

    def _update_id(self):
        self.ID = self.gen_id()

    def flatten(self, skip_branch=None, sep='/'):
        """
        Returned a flattened copy of the nested data structure.

        The keys will be the full path to the item's location in the nested
        data structure separated using the optional sep kwarg, and the value
        will be the value at that location

        Parameters
        ----------
        sep : str, default '/'
            The separating character used for all paths
        skip_branch : list, optional
            A list of `sep` separated paths to branches in the config that you
            would like to skip when flattening. This can be leaves in the
            Config or entire branches

        Returns
        -------
        dict
            A dict whose keys are `sep` separated strings representing the full
            path to the value in the originally nested dictionary. The values
            are the same as those from the original dictionary.

        Examples
        --------
        >>> c = Config({'a': {'b': 1, 'c': 2})
        >>> c.flatten()
        {'a/b': 1, 'a/c': 2}
        >>> c.flatten(sep='_')
        {'a_b': 1, 'a_c': 2}
        >>> c.flatten(sep='_', skip_branch='a_c')
        {'a_b': 1}
        """
        return self._flatten(skip_branch=skip_branch, sep=sep)

    def _flatten(self, branch=None, flattened=None, keypath='',
                 skip_branch=None, sep='/'):
        """
        Internal, recursive flattening implementation
        """
        flattened = flattened if flattened is not None else {}
        skip_branch = skip_branch if skip_branch is not None else []
        if keypath:
            branch = branch if branch is not None else self._d[keypath]
        else:
            branch = branch if branch is not None else self._d
        if isinstance(branch, dict):
            for key, val in branch.items():
                newpath = sep.join([keypath, key]).strip(sep)
                if keypath in skip_branch:
                    continue
                if isinstance(val, dict):
                    self._flatten(branch=val, flattened=flattened,
                                  keypath=newpath, skip_branch=skip_branch,
                                  sep=sep)
                else:
                    flattened[newpath] = val
        return flattened

    @classmethod
    def _to_yaml(cls, dumper, conf):
        return dumper.represent_dict({'ID': conf.ID,
                                      'skip_keys': conf.skip_keys,
                                      '_d': conf._d})

    def update_in_db(self, db, col):
        """
        Update the document in collection `col` contained in this config to the collection `col` in the
        UnQLite database `db`
        """
        # ID = self.ID.encode('utf-8')
        idkey = '{}_docid'.format(self.ID)
        doc_id = db[idkey]
        write_dict = {'ID': self.ID, 'skip_keys': self.skip_keys, '_d': self._d}
        write_dict = prepare_for_db(write_dict)
        with db.transaction():
            success = col.update(doc_id, write_dict)
            print("DB Return Code: {}".format(success))
            if success is False:
                for i in range(3):
                    success = col.update(doc_id, write_dict)
                    if success:
                        break
                else:
                    raise Exception("Unable to update collection")
            db[self.ID] = pickle.dumps(self)
        return success

    def store_in_db(self, db, col):
        # ID = self.ID.encode('utf-8')
        write_dict = {'ID': self.ID, 'skip_keys': self.skip_keys, '_d': self._d}
        write_dict = prepare_for_db(write_dict)
        with db.transaction():
            success = col.store(write_dict)
            print("DB Return Code: {}".format(success))
            if success is False:
                for i in range(3):
                    success = col.update(doc_id, write_dict)
                    if success:
                        break
                else:
                    raise Exception("Unable to update collection")
            db[self.ID] = pickle.dumps(self)
            db['{}_docid'.format(self.ID)] = success
        return success

    @classmethod
    def fromYAML(cls, stream, skip_keys=None, **kwargs):
        """
        Return an instance of a Config object given a raw YAML string or a
        file-like object containing valid YAML syntax. Handles arbitrary YAML
        and YAML dumped by another Config object
        """
        # Use the _much_ faster C based loader imported above unless the user
        # says otherwise
        if 'Loader' not in kwargs:
            kwargs['Loader'] = Loader
        # Make sure we decode any byte-strings
        try:
            stream = stream.decode()
        except AttributeError:
            pass
        data = yaml.load(stream, **kwargs)
        skip_keys = skip_keys if skip_keys is not None else []
        if 'skip_keys' in data:
            for item in data['skip_keys']:
                if isinstance(item, list):
                    skip_keys.append(tuple(item))
                else:
                    skip_keys.append(item)
            del data['skip_keys']
        if 'ID' in data and '_d' in data:
            return cls(data['_d'], skip_keys=skip_keys)
        else:
            return cls(data, skip_keys=skip_keys)

    @staticmethod
    def fromJSON(stream, **kwargs):
        """
        Load config from a JSON stream (either string or file-like object)
        """
        return json.loads(stream)

    @classmethod
    def fromFile(cls, path, syntax='yaml', **kwargs):
        """
        Load config from a file given a path. File must be in YAML or JSON
        syntax
        """
        syntax = syntax.lower()
        if syntax not in ('yaml', 'json'):
            raise ValueError('Can only load from yaml or JSON files')
        path = os.path.expandvars(path)
        if not os.path.isfile(path):
            raise ValueError("Path {} is not a regular file".format(path))
        d = {'yaml': cls.fromYAML, 'json': cls.fromJSON}
        with open(path, 'r') as stream:
            inst = d[syntax](stream, **kwargs)
        return inst

    # @classmethod
    # def fromDocument(cls, doc, skip_keys=None):
    #     """
    #     Load config from a document retrieved from an UnQLite DB
    #     """
    #     # schema = doc['__schema']
    #     data = prepare_from_db(schema, doc)
    #     skip_keys = skip_keys if skip_keys is not None else []
    #     if 'skip_keys' in data:
    #         for item in data['skip_keys']:
    #             if isinstance(item, list):
    #                 skip_keys.append(tuple(item))
    #             else:
    #                 skip_keys.append(item)
    #         del data['skip_keys']
    #     if 'ID' in data and '_d' in data:
    #         inst = cls(data['_d'], skip_keys=skip_keys)
    #     else:
    #         inst = cls(data, skip_keys=skip_keys)
    #     if doc['ID'].decode('utf-8') != inst.ID:
    #         raise ValueError("Loaded config does not have same ID as DB entry")
    #     return inst

    @classmethod
    def from_array(cls, array, skip_fields=[], **kwargs):
        if skip_fields:
            array = remove_fields(array, skip_fields)
        d = numpy_arr_to_dict(array)
        return cls(d, **kwargs)

    def write(self, f):
        """
        Dumps this config object to its YAML representation given a path to a
        file or an open file handle
        """
        if isinstance(f, str):
            f = os.path.expandvars(f)
            f = open(f, 'w')
        yaml.dump(self, stream=f, default_flow_style=False, Dumper=Dumper)
        f.close()

    def dump(self):
        """
        Returns YAML representation of this particular config
        """
        return yaml.dump(self, default_flow_style=False, Dumper=Dumper)


def represent_odict(dumper, data):
    return dumper.represent_dict(data)


def represent_pint_quantity(dumper, data):
    qty = data
    d = {'magnitude': qty.magnitude,
         'units': str(qty.units),
         'base_units': str((1.0*qty.units).to_base_units())}
    return dumper.represent_mapping('!pintq', d)


def construct_pint_quantity(loader, node):
    d = loader.construct_mapping(node)
    return Q_(d['magnitude'], d['units'])


yaml.add_representer(OrderedDict, represent_odict, Dumper=Dumper)
yaml.add_representer(Config, Config._to_yaml, Dumper=Dumper)
yaml.add_multi_representer(pint.quantity._Quantity, represent_pint_quantity,
                           Dumper=Dumper)
yaml.add_constructor('!pintq', construct_pint_quantity, Loader=Loader)
# Add to default, non-C loader just in case
yaml.add_representer(OrderedDict, represent_odict)
yaml.add_representer(Config, Config._to_yaml)
yaml.add_multi_representer(pint.quantity._Quantity, represent_pint_quantity)
yaml.add_constructor('!pintq', construct_pint_quantity)


def group_against(confs, key, sort_key=None, skip_keys=None):
    """
    Group configs against particular parameter.

    Group a list of config objects against a particular parameter. Within each
    group, the parameter specified by `key` will vary, and all other parameters
    will remain fixed. Useful for examining the affect of a single parameter on
    simulation outputs, and for generating line plots with the parameter
    specified by `key` on the x-axis.

    Parameters
    ----------
    confs : list
        A list of :py:class:`nanowire.preprocess.config.Config` objects
    key : str
        A key specifying which parameter simulations will be grouped against.
        Individual simulations within each group will be sorted in increasing
        order of this parameter, and all other parameters will remain constant
        within the group. Key can be a slash-separated string pointing to
        nested items in the config.
    sort_key : str, optional
        An optional key used to sort the order of the inner lists within the
        returned outer list of groups. Works because all parameters within each
        internal group are constant (excluding the parameter specified by
        `key`). The outer list of group lists will be sorted in increasing
        order of the specified sort_key.
    skip_keys : list
        A list of keys to skip when comparing two Configs.

    Returns
    -------
    list
        A list of lists. Each inner list is a group, sorted in increasing order
        of the parameter `key`. All other parameters in each group are
        constant. The outer list may optionally be sorted in increasing order
        of an additonal `sort_key`

    Notes
    -----
    The config objects in the input list `confs` are not copied, and references
    to the original Config objects are stored in the returned data structure
    """

    # We need only need a shallow copy of the list containing all the Config
    # objects. We don't want to modify the orig list but we wish to share
    # the sim objects the two lists contain
    skip_keys = skip_keys if skip_keys is not None else []
    skip_keys.append(key)
    sim_confs = copy.copy(confs)
    sim_groups = [[sim_confs.pop()]]
    while sim_confs:
        conf = sim_confs.pop()
        # Loop through each group, checking if this sim belongs in the
        # group
        match = False
        for group in sim_groups:
            conf2 = group[0]
            params_same = cmp_dicts(conf, conf2, skip_keys=skip_keys)
            if params_same:
                group.append(conf)
                match = True
                break
        # If we didnt find a matching group, we need to create a new group
        # for this simulation
        if not match:
            sim_groups.append([conf])
    for group in sim_groups:
        # Sort the individual sims within a group in increasing order of
        # the parameter we are grouping against a
        if isinstance(group[0][key], pint.quantity._Quantity):
            group.sort(key=lambda aconf: aconf[key].magnitude)
        else:
            group.sort(key=lambda aconf: aconf[key])
    # Sort the groups in increasing order of the provided sort key
    if sort_key:
        sim_groups.sort(key=lambda agroup: agroup[0][sort_key])
    return sim_groups


def group_by(confs, key, sort_key=None):
    """
    Groups simulations by the parameter associated with `key` in the
    config. Within each group, the parameter associated with `key` will
    remain fixed, and all other parameters may vary.

    Parameters
    ----------
    conf : list, tuple
        A list or tuple of :py:class:`nanowire.preprocess.config.Config`
        objects
    key : str
        The key for the parameter you wish to group by. Must be an
        forward-slash separate path-like string.
    sort_key : callable, str, optional
        An optional key by which to sort the parameters within each group. For
        example, group by parameter A but sort each group in increasing order
        of parameter B. If a callable object is provided, that callable will be
        applied to each individual simulation in the group to generate the sort
        key. If an string is provided, it will be interpreted as a key in the
        config and the parameter associated with that key will be used.

    Returns
    -------
    list
        A singly nested list of lists. Each inner list contains a group of
        simulations.
    """

    # This works by storing the different values of the specifed parameter
    # as keys, and a list of sims whose value matches the key as the value
    pdict = {}
    for conf in confs:
        if conf[key] in pdict:
            pdict[conf[key]].append(conf)
        else:
            pdict[conf[key]] = [conf]
    # Now all the sims with matching values for the provided key are just
    # the lists located at each key. We sort the groups in increasing order
    # of the provided key
    groups = sorted(pdict.values(), key=lambda group: group[0][key])
    # If specified, sort the sims within each group in increasing order of
    # the provided sorting key
    if sort_key:
        if callable(sort_key):
            for group in groups:
                group.sort(key=sort_key)
        else:
            for group in groups:
                group.sort(key=lambda sim: conf[sort_key])
    return groups

# @add_logger(logging.getLogger(__name__))
def load_confs(db, base_dir='', query='', table_name='simulations', IDs=None):
    """
    Load configs from disk

    Collect all the simulations contained in the HDF5 database file located
    at filesystem path `db` satisfying the query `query`.

    Parameters
    ----------
    db : str, :py:class:`tb.file.File`
        Either a string path to an HDF5 file containing the database of
        simulation configs, or a PyTables file handle
    base_dir : str, optional
        The base directory of the directory tree that all simulations will
        dump their output data into. If not specified, defaults to the
        directory of the database file.
    query : str, optional
        A query string conforming to PyTables table.where syntax specifying
        which configs should be loaded from the HDF5 database. If not
        specified, all configs in the database are loaded
    table_path : str, optional
        String specifying path inside the HDF5 database to the group
        containing the HDF5 simulations table. Default: '/'
    table_path : str, optional
        String specifying the name of the HDF5 simulations table.
        Default: 'simulations'

    Returns
    -------
    confs : dict
        A dict whose keys are Config IDs and values are a tuple containing a
        :py:class:`nanowire.preprocess.config.Config` object as the first
        element and the output path for the Config object as the second element
    t_sweeps : dict
        A dict describing which simulations are thickness sweeps. The key:value
        pairs are pairs of simulation IDs. The keys are sim IDs whose solution
        files are hard links to the solution of the simulation ID value
    db : :py:class:`tb.file.File`
        An open PyTables file handle. If you passed in an open file handle,
        this is the same file handle you passed in
    """

    if IDs and not isinstance(IDs, set):
        IDs = set(IDs)
    if isinstance(db, str):
        db = unqlite.UnQLite(db)
    else:
        db.open()
    col = db.collection(table_name)
    # Only creates a new collection if one does not already exist
    col.create()
    if not base_dir:
        base_dir = os.path.dirname(db.filename)
    confs = {}
    confs_list = []
    if query:
        filter_func = make_filter(query)
        for doc in col.filter(filter_func):
            ID = doc['ID'].decode()
            short_id = ID[0:10]
            conf_path = os.path.join(base_dir, short_id, 'sim_conf.yml')
            if IDs and ID not in IDs:
                continue
            log.info('Loading config: %s', conf_path)
            # conf = Config.fromDocument(doc)
            conf = pickle.loads(db[ID])
            if ID != conf.ID:
                raise ValueError('ID in database and ID of loaded '
                                 'config do not match')
            confs[conf.ID] = (conf, conf_path)
            confs_list.append(conf)
    else:
        for doc in col.all():
            ID = doc['ID'].decode()
            short_id = ID[0:10]
            pkl_data = db[ID]
            conf_path = os.path.join(base_dir, short_id, 'sim_conf.yml')
            if IDs and ID not in IDs:
                continue
            log.info('Loading config: %s', conf_path)
            # conf = Config.fromDocument(doc)
            conf = pickle.loads(pkl_data)
            if ID != conf.ID:
                raise ValueError('ID in database and ID of loaded '
                                 'config do not match')
            confs[conf.ID] = (conf, conf_path)
            confs_list.append(conf)
        log.info("Confs loaded!")
    # We need to handle the case of thickness sweeps to take advantage of
    # a core efficiency of RCWA, which is that thickness sweeps should come
    # for free. t_sweeps is a dict whose keys and values are both
    # simulation IDs. The keys are ID's of simulations who have a different
    # thickness from the corresponding value, but otherwise have identical
    # parameters. The values in the dict will be the simulations we
    # actually run, and the keys are simulations whose solution will just
    # point to the corresponding value. The keys are all unique (by
    # definition in a dict), but there may be duplicate values
    thickness_paths = set(p for c in confs_list
                          for p in find_keypaths(c, 'thickness'))
    t_sweeps = {}
    # Speed things up by only checking paths that have multiple values
    vals_check = {path: set() for path in thickness_paths}
    for path in thickness_paths:
        for conf in confs_list:
            val = conf.get(path, None)
            if val is not None:
                if isinstance(val, pint.quantity._Quantity):
                    vals_check[path].add(val.to_base_units().magnitude)
                else:
                    vals_check[path].add(val)
    # If, after checking all the configs, there is only 1 value for the
    # thickness, we didn't sweep thru it and we need not worry
    for path, val_set in vals_check.items():
        if len(val_set) == 1:
            thickness_paths.remove(path)
    print("Swept through following thicknesses: {}".format(thickness_paths))
    for path in thickness_paths:
        # Skip max_depth when comparing because it depends on layer
        # thicknesses
        groups = group_against(confs_list, path,
                               skip_keys=['General/max_depth'])
        for group in groups:
            keep_id = group[0].ID
            for c in group[1:]:
                t_sweeps[c.ID] = keep_id
    # self.t_sweeps = t_sweeps
    # for root, dirs, files in os.walk(base_dir):
    #     if os.path.basename(root) in short_ids and 'sim_conf.yml' in files:
    #         conf_path = osp.join(root, 'sim_conf.yml')
    #         self.log.info('Loading config: %s', conf_path)
    #         conf_obj = Config.fromFile(conf_path)
    #         confs.append((conf_path, conf_obj))
    # self.sim_confs = confs
    # if not confs:
    #     log.error('Unable to find any configs')
    #     raise RuntimeError('Unable to find any configs')
    return confs, t_sweeps, db


def dump_configs(db, table_name='simulations',
                 outdir='', fname=None, IDs=None):
    """
    Dump all the Configs in an HDF5 database to YAML files

    Parameters
    ----------

    db : str, :py:class:`tb.file.File`
        Either a string path to an HDF5 file containing the database of
        simulation configs, or a PyTables file handle
    table_path : str, optional
        Path to the group containing the database table. Default: '/'
    table_name : str, optional
        Name of the database table. Default: 'simulations'
    outdir : str, optional
        Directory to dump the config files into. Defaults to the same directory
        as the HDF5 file. If you pass in a path that does not exist, it is
        created.
    fname : callable, optional
        A callable that determines the location of the outputted files beneath
        outdir. The callable must accept a dict as the only argument, and
        return a string representing the path beneath `outdir` (including the
        filename) that the config file will be written to. If not specified
        files are written directly beneath `outdir` named by config ID with a
        '.yml' extension.
    IDs : list/tuple/set, optional
        A container of IDs. Only the IDs in the container will be written

    Returns
    -------

    paths : list
        List of absolute paths to the dumped files
    db : :py:class:`tb.file.File`
        An open PyTables file handle. If you passed in an open file handle,
        this is the same file handle you passed in
    """

    if not os.path.isfile(db):
        raise ValueError('Arg {} is not a regular file'.format(db))
    print("IDS: ", IDs)
    if IDs and not isinstance(IDs, set):
        IDs = set(IDs)
    outdir = outdir if outdir else os.path.dirname(db)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    db = unqlite.UnQLite(db)
    col = db.collection(table_name)
    col.create()
    paths = []
    for doc in col.all():
        ID = doc['ID'].decode()
        if IDs and ID not in IDs:
            continue
        if fname is not None:
            outname = fname(doc)
        else:
            outname = '{}.yml'.format(ID)
        outpath = os.path.join(outdir, outname)
        filedir = os.path.dirname(outpath)
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        conf = Config.fromDocument(doc)
        conf.write(outpath)
        paths.append(outpath)
    return paths, db

def prepare_for_db(d):
    # schema = {}
    good_types = {float, int, list, dict, bool}

    def visit(path, key, value):
        fullpath = posixpath.join(*[str(el) for el in path], str(key))
        if type(value) in good_types or value is None:
            return key, value
        elif isinstance(value, str):
            # schema[fullpath] = 'str'
            return key, value
        elif isinstance(value, pint.quantity._Quantity):
            # if isinstance(value.magnitude, int):
            #     schema[fullpath] = 'pintq int {}'.format(value.units)
            # else:
            #     schema[fullpath] = 'pintq float {}'.format(value.units)
            return key, value.magnitude
        else:
            msg = "Cannot serialize type {}, will store None".format(type(value))
            warnings.warn(msg, RuntimeWarning)

    def enter(path, key, value, **kwargs):
        if isinstance(value, MutableMapping):
            return value, ItemsView(value)
        else:
            return default_enter(path, key, value)
    # d = remap(d, visit=visit, enter=enter)
    d = remap(d, visit=visit)
    # d['__schema'] = schema
    return d

# def prepare_from_db(schema, d):

#     def visit(path, key, value):
#         fullpath = posixpath.join(*[str(el) for el in path], str(key))
#         if fullpath in schema:
#             method = schema[fullpath].decode('utf-8')
#             if method == 'str':
#                 return key, value.decode('utf-8')
#             if 'pintq' in method:
#                 data = method.split(None, 2)
#                 klass = data[1]
#                 units = data[2]
#                 if klass == 'int':
#                     return key, Q_(int(value), units)
#                 else:
#                     return key, Q_(float(value), units)
#             elif method == 'pickle':
#                 ret = pickle.loads(value)
#                 return key, ret
#             else:
#                 raise ValueError("Invalid entry {} in schema".format(method))
#         else:
#             return key, value
#     d = remap(d, visit=visit)
#     return d
