import os
import logging
import ruamel.yaml as yaml
import re
from collections import MutableMapping, OrderedDict
from copy import deepcopy
# from utils import configure_logger

# Configure module level logger if not running as main process
# if not __name__ == '__main__':
#     logger = configure_logger(level='DEBUG',name='config',
#                               console=True,logfile='logs/config.log')


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

    def __init__(self, path=None, data=None):
        # self.log = logging.getLogger(name='config')
        # self.log.debug('CONFIG INIT')
        if path:
            self.data = self._parse_file(path)
        else:
            self.data = dict()
            self.update(dict(data))
        self._update_params()
        self.dep_graph = {}

    def _parse_file(self, path):
        """Parse the YAML file provided at the command line"""
        # self.log.info('Parsing YAML file')
        with open(path, 'r') as cfile:
            text = cfile.read()
        conf = yaml.load(text, Loader=yaml.Loader)
        return conf

    def expand_vars(self, in_table=None, old_key=None):
        """Expand environment variables in the config and update it in place
        without changing anything else"""
        if in_table:
            t = in_table
        else:
            t = self.data
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
                # If we get string, first replace environment variables
                value = re.sub('\$([A-z0-9-_]+)', get_env_variable, value)
                t[key] = value
                # Next check for matches to the
                # replacement string and loop through all of then
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
        # First we find all the references and the exact location(s) in the config
        # that each reference ocurrs at
        self._find_references()
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
            # Make sure we store as a float if possible
            try:
                self[rep_seq] = float(rep_par)
            except:
                self[rep_seq] = rep_par

    def _check_resolved(self, refs):
        """Checks if a list of references have all been resolved"""
        bools = []
        for ref in refs:
            if 'resolved' in self.dep_graph[ref]:
                bools.append(self.dep_graph[ref]['resolved'])
            else:
                bools.append(False)
        return all(bools)

    def interpolate(self):
        """Scans the config for any reference strings and resolves them to
        their actual values by retrieving them from elsewhere in the config"""
        # self.log.debug('Interpolating replacement strings')
        self.build_dependencies()
        config_resolved = False
        while not config_resolved:
            # self.log.debug('CONFIG NOT RESOLVED, MAKING PASS')
            # Now we can actually perform any resolution
            for ref, ref_data in self.dep_graph.items():
                # If the actual location of this references doesn't itself refer to
                # something else, we can safely resolve it because we know it has a
                # value
                if 'resolved' in ref_data:
                    is_resolved = ref_data['resolved']
                else:
                    is_resolved = False
                if not is_resolved:
                    if 'ref_to' not in ref_data:
                        # self.log.debug('NO REFERENCES, RESOLVING')
                        self._resolve(ref)
                        self.dep_graph[ref]['resolved'] = True
                    else:
                        # self.log.debug('CHECKING REFERENCES')
                        # If all the locations this reference points to are resolved, then we
                        # can go ahead and resolve this one
                        if self._check_resolved(ref_data['ref_to']):
                            self._resolve(ref)
                            self.dep_graph[ref]['resolved'] = True
            config_resolved = self._check_resolved(self.dep_graph.keys())

    def evaluate(self, in_table=None, old_key=None):
        # self.log.info('Evaluating params')
        # Evaluates any expressions surrounded in back ticks `like_so+blah`
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
                    expr = value.strip('`')
                    result = eval(expr)
                    key_seq = old_key.split('.')
                    key_seq.append(key)
                    self[key_seq] = result

    def _update_params(self):
        # self.log.info('Updating params')
        self.fixed = []
        self.variable = []
        self.variable_thickness = []
        self.optimized = []
        for par, data in self.data['Simulation']['params'].items():
            if data['type'] == 'fixed':
                self.fixed.append(('Simulation', 'params', par))
            elif data['type'] == 'variable':
                if par == 'thickness':
                    self.variable_thickness.append(
                        ('Simulation', 'params', par))
                else:
                    self.variable.append(('Simulation', 'params', par))
            elif data['type'] == 'optimized':
                self.optimized.append(('Simulation', 'params', par))
            else:
                loc = '.'.join('Simulation', 'params', par)
                raise ValueError(
                    'Specified an invalid config type at {}'.format(loc))

        for layer, layer_data in self.data['Layers'].items():
            for par, data in layer_data['params'].items():
                if data['type'] == 'fixed':
                    self.fixed.append(('Layers', layer, 'params', par))
                elif data['type'] == 'variable':
                    if par == 'thickness':
                        self.variable_thickness.append(
                            ('Layers', layer, 'params', par))
                    else:
                        self.variable.append(('Layers', layer, 'params', par))
                elif data['type'] == 'optimized':
                    self.optimized.append(('Layers', layer, 'params', par))
                else:
                    loc = '.'.join('Layers', layer, 'params', par)
                    raise ValueError(
                        'Specified an invalid config type at {}'.format(loc))

    def __getitem__(self, key):
        """This setup allows us to get a value using a sequence with the usual
        [] operator"""
        # self.log.debug('Getting key: {}'.format(key))
        if isinstance(key, tuple):
            return self.getfromseq(key)
        elif isinstance(key, list):
            return self.getfromseq(key)
        else:
            return self.data[key]

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
        self._update_params()

    def __delitem__(self, key):
        if isinstance(key, tuple):
            self.delfromseq(key)
        elif isinstance(key, list):
            self.delfromseq(key)
        else:
            del self.data[key]
        self._update_params()

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        """We'll just borrow the string representation from dict"""
        return dict.__str__(self.data)

    def getfromseq(self, keyset):
        """A convenience method to get the section of the config file located at the end
        of the sequence of keys"""
        section = self.data
        for key in keyset:
            section = section[key]
        return section

    def setfromseq(self, keyset, value):
        """A convenience method to set the a value in the config given a sequence of keys"""
        sect = self.getfromseq(keyset[:-1])
        sect[keyset[-1]] = value
        self._update_params()

    def delfromseq(self, keyset):
        """Deletes the section of the config located at the end of a sequence
        of keys"""
        del self.getfromseq(keyset[:-1])[keyset[-1]]
        self._update_params()

    def copy(self):
        """Returns a copy of the current config object"""
        return deepcopy(self)

    def write(self, path):
        """Dumps this config object to its YAML representation given a path to a file"""
        with open(path, 'w') as out:
            out.write(yaml.dump(self.data, default_flow_style=False))
        return

    def dump(self):
        """Returns YAML representation of this particular config"""
        return yaml.dump(self.data, default_flow_style=False)

    def get_height(self):
        """Returns the total height of the device"""
        height = 0
        for layer, ldata in self['Layers'].items():
            layer_t = ldata['params']['thickness']['value']
            height += layer_t
        # self.log.debug('TOTAL HEIGHT = %f'%height)
        return height

    def sorted_dict(self, adict, reverse=False):
        """Returns a sorted version of a dictionary sorted by the 'order' key.
        Used to sort layers, geometric shapes into their proper physical order.
        Can pass in a kwarg to reverse the order if desired"""
        try:
            for key, data in adict.items():
                data['order']
        except KeyError:
            raise KeyError('The dictionary you are attempting to sort must '
                           'itself contain a dictionary that has an "order" key')
        sorted_layers = OrderedDict(sorted(adict.items(), key=lambda tup: tup[1]['order'],
                                           reverse=reverse))
        return sorted_layers
