import os
import logging
import ruamel.yaml as yaml
from collections import MutableMapping,OrderedDict
from copy import deepcopy 

class Config(MutableMapping):
    """An object to represent the simulation config that behaves like a dict.
    It can be initialized with a path, or an actual python data structure as
    with the usual dict but has some extra convenience methods"""

    def __init__(self,path=None,data=None):
        if path:
            self.data = self._parse_file(path)
        else:
            self.data = dict()
            self.update(dict(data))
        self._update_params()

    def _parse_file(self,path):
        """Parse the YAML file provided at the command line"""
         
        with open(path,'r') as cfile:
            text = cfile.read()
        conf = yaml.load(text,Loader=yaml.Loader)
        return conf

    def _update_params(self):
        self.fixed = []
        self.variable = []
        self.sorting = []
        self.evaluated = []
        self.optimized = []
        for par,data in self.data['Simulation']['params'].items():
            if data['type'] == 'fixed':
                self.fixed.append(('Simulation','params',par))
            elif data['type'] == 'variable':
                self.variable.append(('Simulation','params',par))
            elif data['type'] == 'sorting':
                self.sorting.append(('Simulation','params',par))
            elif data['type'] == 'evaluated':
                self.evaluated.append(('Simulation','params',par))
            elif data['type'] == 'optimized':
                self.optimized.append(('Simulation','params',par))
            else:
                loc = '.'.join('Simulation','params',par)
                raise ValueError('Specified an invalid config type at {}'.format(loc))

        for layer,layer_data in self.data['Layers'].items():
            for par,data in layer_data['params'].items(): 
                if data['type'] == 'fixed':
                    self.fixed.append(('Layers',layer,'params',par))
                elif data['type'] == 'variable':
                    self.variable.append(('Layers',layer,'params',par))
                elif data['type'] == 'sorting':
                    self.sorting.append(('Layers',layer,'params',par))
                elif data['type'] == 'evaluated':
                    self.evaluated.append(('Layers',layer,'params',par))
                elif data['type'] == 'optimized':
                    self.optimized.append(('Layers',layer,'params',par))
                else:
                    loc = '.'.join('Layers',layer,'params',par)
                    raise ValueError('Specified an invalid config type at {}'.format(loc))
        # Make sure the sorting parameters are in the proper order according to
        # the value of their keys
        getkey = lambda seq: self[seq]['key']
        self.sorting = sorted(self.sorting,key=getkey)

    def __getitem__(self, key):
        """This setup allows us to get a value using a sequence with the usual
        [] operator"""
        if isinstance(key,tuple):
            return self.getfromseq(key)
        elif isinstance(key,list):
            return self.getfromseq(key)
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        """This setup allows us to set a value using a sequence with the usual
        [] operator. It also updates the parameter lists to reflect any
        potential changes"""
        if isinstance(key,tuple):
            self.setfromseq(key,value)
        elif isinstance(key,list):
            self.setfromseq(key,value)
        else:
            self.data[key] = value
        self._update_params()

    def __delitem__(self, key):
        if isinstance(key,tuple):
            self.delfromseq(key)
        elif isinstance(key,list):
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

    def getfromseq(self,keyset):
        """A convenience method to get the section of the config file located at the end
        of the sequence of keys"""
        section = self.data
        for key in keyset:
            section = section[key]
        return section

    def setfromseq(self,keyset,value):
        """A convenience method to set the a value in the config given a sequence of keys"""
        sect = self.getfromseq(keyset[:-1])
        sect[keyset[-1]] = value
        self._update_params()

    def delfromseq(self,keyset):
        """Deletes the section of the config located at the end of a sequence
        of keys"""
        del self.getfromseq(keyset[:-1])[keyset[-1]]
        self._update_params()

    def copy(self):
        """Returns a copy of the current config object"""
        return deepcopy(self)

    def write(self,path):
        """Dumps this config object to its YAML representation given a path to a file"""
        with open(path,'w') as out:
            out.write(yaml.dump(self.data,default_flow_style=False))

    def dump(self):
        """Returns YAML representation of this particular config"""
        return yaml.dump(self.data,default_flow_style=False)

    def get_height(self):
        """Returns the total height of the device"""
        height = 0
        for layer,ldata in self['Layers'].items():
            layer_t = ldata['params']['thickness']['value']
            height += layer_t
        #self.log.debug('TOTAL HEIGHT = %f'%height)
        return height

    def sorted_dict(self,adict,reverse=False):
        """Returns a sorted version of a dictionary sorted by the 'order' key.
        Used to sort layers, geometric shapes into their proper physical order.
        Can pass in a kwarg to reverse the order if desired"""
        try: 
            for key,data in adict.items():
                data['order']
        except KeyError:
            raise KeyError('The dictionary you are attempting to sort must '
            'itself contain a dictionary that has an "order" key')
        sort_func = lambda tup: tup[1]['order']
        sorted_layers = OrderedDict(sorted(adict.items(),
                                    key=sort_func,reverse=reverse)) 
        return sorted_layers


def parse_file(path):
    """Parse the YAML file provided at the command line"""
     
    with open(path,'r') as cfile:
        text = cfile.read()
    conf = yaml.load(text)
    return conf

def configure_logger(level,logger_name,log_dir,logfile):
    # Get numeric level safely
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)
    # Set formatting
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
    # Get logger with name
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    # Set up file handler
    try:
        os.makedirs(log_dir)
    except OSError:
        # Log dir already exists
        pass
    output_file = os.path.join(log_dir,logfile)
    fhandler = logging.FileHandler(output_file)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
       
    return logger
