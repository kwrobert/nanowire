import os
import numpy
# import logging
import ruamel.yaml as yaml
import re
from collections import MutableMapping, OrderedDict
from copy import deepcopy
import pprint
from mako.template import Template
from .utils import get_combos, do_profile
from line_profiler import LineProfiler



def numpy_float64_representer(dumper, data):
    node = dumper.represent_scalar(u'tag:yaml.org,2002:float',
                                   data.__repr__())
    return node

yaml.add_representer(numpy.float64, numpy_float64_representer)


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


class Config(MutableMapping):
    """An object to represent the simulation config that behaves like a dict.
    It can be initialized with a path, or an actual python data structure as
    with the usual dict but has some extra convenience methods"""

    def __init__(self, path=None, data=None, raw_text=None):
        # self.log = logging.getLogger(name='config')
        # self.log.debug('CONFIG INIT')
        if path:
            self.data = self._parse_file(path)
        elif raw_text:
            self.data = yaml.load(raw_text, Loader=yaml.CLoader)
        else:
            self.data = {}
            self.update(dict(data))
        # self._update_params()
        # self.dep_graph = {}
        # self.resolved = False

    def _parse_file(self, path):
        """Parse the YAML file provided at the command line"""
        # self.log.info('Parsing YAML file')
        path = os.path.expandvars(path)
        with open(path, 'r') as cfile:
            text = cfile.read()
        conf = yaml.load(text, Loader=yaml.CLoader)
        return conf

    def expand_vars(self, in_table=None, old_key=None, update=True):
        """
        Recurse through a dictionary and expand environment variables found in
        any string values.

        If in_table is not provided, the self.data dict will be recursed
        through by default, updated in place and also returned.

        If in_table is provided, that dictionary will be recursed through,
        updated, and returned.

        If update=False, the relevant table will not be updated, but instead
        will be copied and a new dictionary will be returned. The original dict
        will remain unmodified
        """
        if in_table:
            t = in_table
        else:
            t = self.data
        if not update:
            t = deepcopy(t)
        for key, value in t.items():
            # If we get a dict, recurse
            if isinstance(value, dict):
                if old_key:
                    new_key = '%s.%s' % (old_key, key)
                else:
                    new_key = key
                self.expand_vars(in_table=value, old_key=new_key)
            elif isinstance(value, str):
                # If we get string, first replace environment variables
                value = re.sub('\$([A-z0-9-_]+)', get_env_variable, value)
                t[key] = value
        return t

    def _find_references(self, in_table=None, old_key=None):
        """Build out the dependency graph of references in the config. Will
        also resolve any environment variable references in any config items of
        string type. Environment variable references must begin with a dollar
        sign and can contain alphanumeric characters, dashes, or underscores.
        Everything from the dollar sign up to the first character not in the
        specified groups will be replaced with the environment variable
        matching the parsed string after the dollar sign."""
        if in_table:
            t = in_table
        else:
            t = self.data
        for key, value in t.items():
            # If we got a dict back, recurse
            if isinstance(value, dict):
                if old_key:
                    new_key = '%s.%s' % (old_key, key)
                else:
                    new_key = key
                self._find_references(in_table=value, old_key=new_key)
            elif isinstance(value, str):
                # Next check for matches to the replacement string and loop
                # through all of then
                matches = re.findall('%\(([^)]+)\)s', value)
                new_key = '%s.%s' % (old_key, key)
                for match in matches:
                    # If we've already found this reference before, increment
                    # its reference count and update the list of keys referring
                    # to it
                    if match in self.dep_graph:
                        self.dep_graph[match]['ref_count'] += 1
                        self.dep_graph[match]['ref_by'].append(new_key)
                    else:
                        self.dep_graph[match] = {
                            'ref_count': 1, 'ref_by': [new_key]}

    def build_dependencies(self):
        """
        Build out the dependency graph that determines which other config item a
        particular config item refers to, and which config items refer to it.
        We can then use this graph to determine which substitutions we need to
        perform first when calling interpolate
        """
        # First we find all the references and the exact location(s) in the config
        # that each reference ocurrs at
        self._find_references()
        # Next we determine if any of the things we refer to in the dependency
        # graph have backticks, meaning they must be evaluated before the
        # things that refer to them actually resolve their value
        for path in self.dep_graph.keys():
            key_seq = path.split('.')
            val = self.getfromseq(key_seq)
            if isinstance(val, str) and (val[0] == '`' and val[-1] == '`'):
                self.dep_graph[path].update({'evaluated': False})
            else:
                self.dep_graph[path].update({'evaluated': True})

        # Now we build out the "refers_to" entry for each reference to see if a
        # reference at one place in the table refers to some other value
        # For each reference we found
        for ref, data in self.dep_graph.items():
            # Loop through all the other references. If the above reference exists
            # in the "ref_by" table, we know the above reference refers to another
            # value and we need to resolve that value first. Note we also do this
            # for ref itself so we can catch circular references
            for other_ref, its_data in self.dep_graph.items():
                if ref in its_data['ref_by']:
                    if other_ref == ref:
                        raise ValueError('There is a circular reference in your'
                                         ' config file at %s' % ref)
                    else:
                        if 'ref_to' in data:
                            data['ref_to'].append(other_ref)
                        else:
                            data['ref_to'] = [other_ref]
            # Nothing has been resolved at this poing
            data['resolved'] = False

    def _resolve(self, ref):
        ref_data = self.dep_graph[ref]
        # Retrieve the value of this reference
        key_seq = ref.split('.')
        repl_val = self[key_seq]
        # Loop through all the locations that contain this reference
        for loc in ref_data['ref_by']:
            # Get the string we need to run the replacement on
            rep_seq = loc.split('.')
            entry_to_repl = self[rep_seq]
            # Run the actual replacement and set the value at this
            # location to the new string
            pattern = '%\({}\)s'.format(ref)
            rep_par = re.sub(pattern, str(repl_val), entry_to_repl)
            self[rep_seq] = rep_par

    def _check_resolved(self, refs):
        """Checks if a list of references have all been resolved"""
        bools = []
        for ref in refs:
            bools.append(self.dep_graph[ref]['resolved'])
        return all(bools)

    def _check_evaled(self, refs):
        """Checks if a list of references have all been evaluated"""
        bools = []
        for ref in refs:
            bools.append(self.dep_graph[ref]['evaluated'])
        return all(bools)

    def interpolate(self):
        """Scans the config for any reference strings and resolves them to
        their actual values by retrieving them from elsewhere in the config"""
        # self.log.debug('Interpolating replacement strings')
        self.build_dependencies()
        while not self.resolved:
            # self.log.debug('CONFIG NOT RESOLVED, MAKING PASS')
            # Now we can actually perform any resolutions
            for ref, ref_data in self.dep_graph.items():
                # Has this config item already been resolved?
                is_resolved = ref_data['resolved']
                if not is_resolved:
                    if 'ref_to' not in ref_data:
                        # self.log.debug('NO REFERENCES, RESOLVING')
                        # Before resolving all the places in the config
                        # that where this reference occurs, we first need
                        # to evaluate value at this path so we don't
                        # resolve references to this path with a string
                        # surrounded in backticks
                        evaled = ref_data['evaluated']
                        if not evaled:
                            key_seq = ref.split('.')
                            val = self.getfromseq(key_seq)
                            res = self.eval_expr(val)
                            self.setfromseq(key_seq, res)
                            self.dep_graph[ref]['evaluated'] = True
                        self._resolve(ref)
                        self.dep_graph[ref]['resolved'] = True
                    else:
                        # If all the locations this reference points to are
                        # resolved and evaluated, then we can go ahead and
                        # resolve this one
                        if self._check_resolved(ref_data['ref_to']) and self._check_evaled(ref_data['ref_to']):
                            evaled = ref_data['evaluated']
                            if not evaled:
                                key_seq = ref.split('.')
                                val = self.getfromseq(key_seq)
                                res = self.eval_expr(val)
                                self.setfromseq(key_seq, res)
                                self.dep_graph[ref]['evaluated'] = True
                            self._resolve(ref)
                            self.dep_graph[ref]['resolved'] = True
            self.resolved = self._check_resolved(self.dep_graph.keys())

    def eval_expr(self, expr_str):
        """
        Evaluate the provided expression string and return the result
        """
        expr = expr_str.strip('`')
        result = eval(expr)
        return result

    def evaluate(self, in_table=None, old_key=None):
        """
        Evaluates any expressions surrounded in back ticks
        """
        if in_table:
            t = in_table
        else:
            t = self.data
        for key, value in t.items():
            # If we got a table back, recurse
            if isinstance(value, dict):
                if old_key:
                    new_key = '%s.%s' % (old_key, key)
                else:
                    new_key = key
                self.evaluate(value, new_key)
            elif isinstance(value, str):
                if value[0] == '`' and value[-1] == '`':
                    result = self.eval_expr(value)
                    key_seq = old_key.split('.')
                    key_seq.append(key)
                    self.setfromseq(key_seq, result)


    def _update_params(self):
        # self.log.info('Updating params')
        self.variable = []
        self.variable_thickness = []
        self.optimized = []

    def __getitem__(self, key):
        """This setup allows us to get a value using a sequence with the usual
        [] operator"""
        # self.log.debug('Getting key: {}'.format(key))
        if isinstance(key, tuple):
            val = self.getfromseq(key)
        elif isinstance(key, list):
            val = self.getfromseq(key)
        else:
            val = self.data[key]
        # if isinstance(val, str):
        #     # If we get string, first replace environment variables
        #     val = re.sub('\$([A-z0-9-_]+)', get_env_variable, val)
        # elif isinstance(val, dict):
        #     val = self.expand_vars(in_table=val, update=False)
        return val

    def __setitem__(self, key, value):
        """This setup allows us to set a value using a sequence with the usual
        [] operator. It also updates the parameter lists to reflect any
        potential changes"""
        if isinstance(key, tuple):
            self.setfromseq(key, value)
        elif isinstance(key, list):
            self.setfromseq(key, value)
        else:
            self.data[key] = value

    def __delitem__(self, key):
        if isinstance(key, tuple):
            self.delfromseq(key)
        elif isinstance(key, list):
            self.delfromseq(key)
        else:
            del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        """We'll just borrow the string representation from dict"""
        return dict.__str__(self.data)

    def getfromseq(self, keyset):
        """
        A convenience method to get the section of the config file located
        at the end of the sequence of keys
        """

        section = self.data
        for key in keyset:
            section = section[key]
        return section

    def setfromseq(self, keyset, value):
        """A convenience method to set the a value in the config given a sequence of keys"""
        sect = self.getfromseq(keyset[:-1])
        sect[keyset[-1]] = value

    def delfromseq(self, keyset):
        """Deletes the section of the config located at the end of a sequence
        of keys"""
        del self.getfromseq(keyset[:-1])[keyset[-1]]

    def copy(self):
        """Returns a deep copy of the self.data dict"""
        return deepcopy(self.data)

    def write(self, path):
        """Dumps this config object to its YAML representation given a path to a file"""
        path = os.path.expandvars(path)
        with open(path, 'w') as out:
            out.write(yaml.dump(self.data, default_flow_style=False))
        return

    def dump(self):
        """Returns YAML representation of this particular config"""
        return yaml.dump(self.data, default_flow_style=False)


    def sorted_dict(self, adict, reverse=False):
        """Returns a sorted version of a dictionary sorted by the 'order' key.
        Used to sort layers, geometric shapes into their proper physical order.
        Can pass in a kwarg to reverse the order if desired"""
        try:
            sorted_layers = OrderedDict(sorted(adict.items(),
                                               key=lambda tup: tup[1]['order'],
                                               reverse=reverse))
        except KeyError:
            raise KeyError('The dictionary you are attempting to sort must '
                           'itself contain a dictionary that has an "order" key')
        return sorted_layers
