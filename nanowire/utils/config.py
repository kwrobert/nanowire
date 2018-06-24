import os
import posixpath
import re
from dicthash import generate_hash_from_dict
from collections import MutableMapping, OrderedDict
import yaml
from yaml import load as yload, dump as ydump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import json
import pint
from nanowire.utils.utils import (
    remove_fields,
    numpy_arr_to_dict,
    Q_,
)


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


def find_lists(o, keypath=[], list_locs=None, lists=None):
    """
    Find all lists in a dictionary recursively
    """

    list_locs = list_locs if list_locs is not None else []
    lists = lists if lists is not None else []
    if isinstance(o, dict):
        for key in o.keys():
            loc = keypath + [key]
            val = o[key]
            if isinstance(val, list):
                list_locs.append(loc)
                lists.append(val)
            elif isinstance(val, dict):
                find_lists(val, keypath=loc, list_locs=list_locs, lists=lists)
            else:
                continue
    return list_locs, lists


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
        return self.getfromseq(parts)

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
        self._flatten(skip_branch=skip_branch, sep=sep)

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
    d = {'magnitude': qty.magnitude,
         'units': str(qty.units),
         'base_units': str((1.0*qty.units).to_base_units())}
    return dumper.represent_mapping('!pintq', d)


def construct_pint_quantity(loader, node):
    d = loader.construct_mapping(node)
    return Q_(d['magnitude'], d['units'])


yaml.add_representer(OrderedDict, represent_odict)
yaml.add_representer(Config, Config._to_yaml)
yaml.add_multi_representer(pint.quantity._Quantity, represent_pint_quantity)
yaml.add_constructor('!pintq', construct_pint_quantity)
