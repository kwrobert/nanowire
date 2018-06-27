import os
import posixpath
import copy
import re
import logging
import yaml
import json
import pint
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from yaml import load as yload, dump as ydump
import tables as tb
from dicthash import generate_hash_from_dict
from collections import MutableMapping, OrderedDict
from nanowire.utils.utils import (
    remove_fields,
    numpy_arr_to_dict,
    Q_,
    cmp_dicts,
    find_keypaths,
)
from nanowire.utils.logging import add_logger


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
        if 'array_period' in k:
            print('Getting array period')
        val = self.getfromseq(parts)
        if 'array_period' in k:
            print('array period: ', val)
            print('array period type: ', type(val.magnitude))
        return val

    def __setitem__(self, k, v):
        """
        This setup allows setting a value inside a nested dictionary using
        either a slash separated string or a list/tuple with the usual []
        operator.
        """

        parts = self._get_parts(k)
        if 'array_period' in k:
            print('Setting array period')
            print('array period before: ', v)
            print('array period type before: ', type(v.magnitude))
        self.setfromseq(parts, v)
        if 'array_period' in k:
            print('array period after: ',
                  self['Simulation/array_period'].magnitude)
            print('array period type after: ',
                  type(self['Simulation/array_period'].magnitude))
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

    @classmethod
    def fromYAML(cls, *args, skip_keys=None, **kwargs):
        """
        Return an instance of a Config object given a raw YAML string or a
        file-like object containing valid YAML syntax. Handles arbitrary YAML
        and YAML dumped by another Config object
        """
        data = yload(*args, **kwargs)
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

    @classmethod
    def from_array(cls, array, skip_fields=[], **kwargs):
        if skip_fields:
            array = remove_fields(array, skip_fields)
        d = numpy_arr_to_dict(array)
        return cls(d, **kwargs)

    def write(self, f):
        """
        Dumps this config object to its YAML representation given a path to a
        file
        """
        print('array period in write: ', self['Simulation/array_period'])
        print('array period type in write: ', type(self['Simulation/array_period'].magnitude))
        if isinstance(f, str):
            f = os.path.expandvars(f)
            f = open(f, 'w')
        ydump(self, stream=f, default_flow_style=False)

    def dump(self):
        """
        Returns YAML representation of this particular config
        """
        return ydump(self, default_flow_style=False)


def represent_odict(dumper, data):
    return dumper.represent_dict(data)


def represent_pint_quantity(dumper, data):
    qty = data
    print('Input qty: {}'.format(qty))
    print(type(qty.magnitude))
    d = {'magnitude': qty.magnitude,
         'units': str(qty.units),
         'base_units': str((1.0*qty.units).to_base_units())}
    print('pint dict repr: {}'.format(d))
    return dumper.represent_mapping('!pintq', d)


def construct_pint_quantity(loader, node):
    d = loader.construct_mapping(node)
    return Q_(d['magnitude'], d['units'])


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
    The config options in the input list `confs` are not copied, and references
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
    sort_key : str, optional
        An optional key by which to sort the parameters within each group. For
        example, group by parameter A but sort each group in increasing order
        of parameter B. If a callable object is provided, that callable will be
        applied to each individual simulation in the group to generate the sort
        key. If an iterable is provided, it will be interpreted as a key in the
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

@add_logger(logging.getLogger(__name__))
def load_confs(db, base_dir='', query='', table_path='/',
               table_name='simulations'):
    """
    Load configs from disk

    Collect all the simulations contained in the HDF5 database file located
    at filesystem path `db` satisfying the query `query`.

    Parameters
    ----------
    db : str
        A path to an HDF5 file containing the database of simulation
        configs
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
    list
        A list of :py:class:`nanowire.preprocess.config.Config` objects
    """

    # Find the data files and instantiate Config objects
    if not base_dir:
        base_dir = os.path.dirname(db)
    with tb.open_file(db, 'r') as hdf:
        table = hdf.get_node(table_path, name=table_name,
                             classname='Table')
        confs = {}
        if query:
            condvars = {k.replace('/', '__'): v
                        for k, v in table.colinstances.items()}
            for row in table.read_where(query, condvars=condvars):
                short_id = row['ID'].decode()[0:10]
                conf_path = os.path.join(base_dir, short_id, 'sim_conf.yml')
                log.info('Loading config: %s', conf_path)
                conf = Config.fromYAML(row['yaml'])
                if row['ID'].decode() != conf.ID:
                    raise ValueError('ID in database and ID of loaded '
                                     'config do not match')
                confs[conf.ID] = (conf, conf_path)
        else:
            for row in table.read():
                short_id = row['ID'].decode()[0:10]
                conf_path = os.path.join(base_dir, short_id, 'sim_conf.yml')
                log.info('Loading config: %s', conf_path)
                conf = Config.fromYAML(row['yaml'])
                if row['ID'].decode() != conf.ID:
                    raise ValueError('ID in database and ID of loaded '
                                     'config do not match')
                confs[conf.ID] = (conf, conf_path)
    confs_list = [tup[0] for tup in confs.values()]
    thickness_paths = find_keypaths(confs_list[0], 'thickness')
    # We need to handle the case of thickness sweeps to take advantage of
    # a core efficiency of RCWA, which is that thickness sweeps should come
    # for free. t_sweeps is a dict whose keys and values are both
    # simulation IDs. The keys are ID's of simulations who have a different
    # thickness from the corresponding value, but otherwise have identical
    # parameters. The values in the dict will be the simulations we
    # actually run, and the keys are simulations whose solution will just
    # point to the corresponding value. The keys are all unique (by
    # definition in a dict), but there may be duplicate values
    t_sweeps = {}
    for path in thickness_paths:
        # Skip max_depth when comparing because it depends on layer
        # thicknesses
        groups = group_against(confs_list, path,
                               skip_keys=['General/max_depth'])
        for i, b in enumerate([len(group) > 1 for group in groups]):
            if b:
                log.info('Found thickness sweep over %s', path)
                keep_id = groups[i][0].ID
                for c in groups[i][1:]:
                    t_sweeps[c.ID] = keep_id
    # self.t_sweeps = t_sweeps
    # for root, dirs, files in os.walk(base_dir):
    #     if os.path.basename(root) in short_ids and 'sim_conf.yml' in files:
    #         conf_path = osp.join(root, 'sim_conf.yml')
    #         self.log.info('Loading config: %s', conf_path)
    #         conf_obj = Config.fromFile(conf_path)
    #         confs.append((conf_path, conf_obj))
    # self.sim_confs = confs
    if not confs:
        log.error('Unable to find any configs')
        raise RuntimeError('Unable to find any configs')
    return confs, t_sweeps


def dump_configs(hdf_path, table_path='/', table_name='simulations',
                 outdir=''):
    """
    Dump all the Configs in an HDF5 table to YAML files

    Parameters
    ----------

    hdf_path : str
        Path to the HDF5 file containing the database
    table_path : str, optional
        Path to the group containing the database table. Default: '/'
    table_name : str, optional
        Name of the database table. Default: 'simulations'
    outdir : str, optional
        Directory to dump the config files into. Defaults to the same directory
        as the HDF5 file. If you pass in a path that does not exist, it is
        created.

    Returns
    -------

    list
        List of absolute paths to the dumped files
    """

    if not os.path.isfile(hdf_path):
        raise ValueError('Arg {} is not a regular file'.format(hdf_path))
    outdir = outdir if outdir else os.path.dirname(hdf_path)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    hdf = tb.open_file(hdf_path, 'r')
    table = hdf.get_node(table_path, name=table_name, classname='Table')
    paths = []
    for row in table.iterrows():
        stream = row['yaml'].decode()
        ID = row['ID'].decode()
        fname = '{}.yml'.format(ID)
        outpath = os.path.join(outdir, fname)
        with open(outpath, 'w') as f:
            f.write(stream)
        paths.append(outpath)
    hdf.close()
    return paths
