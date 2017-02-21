import ruamel.yaml as yaml
import hashlib

from sqlalchemy.orm import deferred
from sqlalchemy import Column, Integer, String, Binary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from collections import MutableMapping,OrderedDict


Base = declarative_base()

class SimMeta(type(MutableMapping),type(Base)):
    pass

class Simulation(MutableMapping,Base,metaclass=SimMeta):

    __metaclass__ = SimMeta
    __tablename__ = 'master'
    
    id = Column(String,primary_key = True)
    conf = Column(String)
    e_field = deferred(Column(Binary))
    state = None

    def __init__(self,path=None,data=None):
        if path:
            self.data = self._parse_file(path)
        else:
            self.data = dict()
            self.update(dict(data))
        self._update_params()
        Base.__init__(self,state='created')

    def _parse_file(self,path):
        """Parse the YAML file provided at the command line"""
         
        with open(path,'r') as cfile:
            text = cfile.read()
        self.conf = text
        return yaml.load(text,Loader=yaml.Loader)

    def _update_id(self):
        """Update the hash id to reflect any changes in the data
        dict"""
        self.id = make_hash(self.data)

    def _update_yaml(self):
        """Update the YAML representation to reflect any changes in the data
        dict"""
        self.conf = self.dump()

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
        self._update_id()
        self._update_yaml()

    def __delitem__(self, key):
        if isinstance(key,tuple):
            self.delfromseq(key)
        elif isinstance(key,list):
            self.delfromseq(key)
        else:
            del self.data[key]
        self._update_params()
        self._update_id()
        self._update_yaml()

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
        self._update_id()
        self._update_yaml()

    def delfromseq(self,keyset):
        """Deletes the section of the config located at the end of a sequence
        of keys"""
        del self.getfromseq(keyset[:-1])[keyset[-1]]
        self._update_params()
        self._update_id()
        self._update_yaml()

    def copy(self):
        """Returns a copy of the current simulation object"""
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
def make_hash(o):
    """Makes a hash from the dict representing the Simulation config. It is
    consistent across runs and handles arbitrary nesting. Right now it ignores
    any settings in the General section because those aren't really important
    when it comes to differentiating simulations"""
    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        buf = repr(o).encode('utf-8')
        return hashlib.md5(buf).hexdigest()
    new_o = OrderedDict()
    for k, v in sorted(new_o.items(),key=lambda tup: tup[0]):
        if k == 'General':
            continue
        else:
            new_o[k] = make_hash(v)
    out = repr(tuple(frozenset(sorted(new_o.items())))).encode('utf-8')
    return hashlib.md5(out).hexdigest()

