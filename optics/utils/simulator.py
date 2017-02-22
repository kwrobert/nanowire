from __future__ import print_function
import os
import numpy as np
import scipy.interpolate as spi
import scipy.constants as constants
import S4
import hashlib
import logging
import sys
import re
import ruamel.yaml as yaml
from copy import deepcopy

from collections import OrderedDict,MutableMapping
from itertools import chain

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

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
    out = open(output_file,'a')
    fhandler = logging.FileHandler(output_file)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    sl = StreamToLogger(logger,logging.INFO)
    sys.stdout = sl
    sl = StreamToLogger(logger,logging.ERROR)
    sys.stderr = sl
    # Duplicate this new log file descriptor to system stdout so we can
    # intercept output from external S4 C library
    # TODO: Make this go through the logger instead of directly to the file
    # right now entries aren't properly formatted
    os.dup2(out.fileno(),1)
    return logger

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
    for k, v in sorted(o.items(),key=lambda tup: tup[0]):
        if k == 'General':
            continue
        else:
            new_o[k] = make_hash(v)
    out = repr(tuple(frozenset(sorted(new_o.items())))).encode('utf-8')
    return hashlib.md5(out).hexdigest()

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
        self.dep_graph = {}

    def _parse_file(self,path):
        """Parse the YAML file provided at the command line"""

        with open(path,'r') as cfile:
            text = cfile.read()
        conf = yaml.load(text,Loader=yaml.Loader)
        return conf

    def _find_references(self,in_table=None,old_key=None):
        """Build out the dependency graph of references in the config"""
        if in_table:
            t = in_table
        else:
            t = self.data
        for key,value in t.items():
            # If we got a dict back, recurse
            if isinstance(value,dict):
                if old_key:
                    new_key = '%s.%s'%(old_key,key)
                else:
                    new_key = key
                self._find_references(value,new_key)
            elif isinstance(value,str):
                # If we get a string, check for matches to the
                # replacement string and loop through all of then
                matches = re.findall('%\(([^)]+)\)s',value)
                new_key = '%s.%s'%(old_key,key)
                for match in matches:
                    # If we've already found this reference before, increment its
                    # reference count and update the list of keys referring to it
                    if match in self.dep_graph:
                        self.dep_graph[match]['ref_count'] += 1
                        self.dep_graph[match]['ref_by'].append(new_key)
                    else:
                        self.dep_graph[match] = {'ref_count':1,'ref_by':[new_key]}

    def build_dependencies(self):
        # First we find all the references and the exact location(s) in the config
        # that each reference ocurrs at
        self._find_references()
        # Now we build out the "refers_to" entry for each reference to see if a
        # reference at one place in the table refers to some other value
        # For each reference we found
        for ref,data in self.dep_graph.items():
            # Loop through all the other references. If the above reference exists
            # in the "ref_by" table, we know the above reference refers to another
            # value and we need to resolve that value first. Note we also do this
            # for ref itself so we can catch circular references
            for other_ref,its_data in self.dep_graph.items():
                if ref in its_data['ref_by']:
                    if other_ref == ref:
                        raise ValueError('There is a circular reference in your'
                                         ' config file at %s'%ref)
                    else:
                        if 'ref_to' in data:
                            data['ref_to'].append(other_ref)
                        else:
                            data['ref_to'] = [other_ref]

    def _resolve(self,ref):
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
            rep_par = re.sub(pattern,str(repl_val),entry_to_repl)
            # Make sure we store as a float if possible
            try:
                self[rep_seq] = float(rep_par)
            except:
                self[rep_seq] = rep_par

    def _check_resolved(self,refs):
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
        self.build_dependencies()
        config_resolved = False
        while not config_resolved:
            #print('CONFIG NOT RESOLVED, MAKING PASS')
            # Now we can actually perform any resolution
            for ref,ref_data in self.dep_graph.items():
                # If the actual location of this references doesn't itself refer to
                # something else, we can safely resolve it because we know it has a
                # value
                if 'resolved' in ref_data:
                    is_resolved = ref_data['resolved']
                else:
                    is_resolved = False
                if not is_resolved:
                    if 'ref_to' not in ref_data:
                        #print('NO REFERENCES, RESOLVING')
                        self._resolve(ref)
                        self.dep_graph[ref]['resolved'] = True
                    else:
                        #print('CHECKING REFERENCES')
                        # If all the locations this reference points to are resolved, then we
                        # can go ahead and resolve this one
                        if self._check_resolved(ref_data['ref_to']):
                            self._resolve(ref)
                            self.dep_graph[ref]['resolved'] = True
            config_resolved = self._check_resolved(self.dep_graph.keys())

    def evaluate(self,in_table=None,old_key=None):
        # Evaluates any expressions surrounded in back ticks `like_so+blah`
        if in_table:
            t = in_table
        else:
            t = self.data
        for key, value in t.items():
            # If we got a table back, recurse
            if isinstance(value,dict):
                if old_key:
                    new_key = '%s.%s'%(old_key,key)
                else:
                    new_key = key
                self.evaluate(value,new_key)
            elif isinstance(value,str):
                if value[0] == '`' and value[-1] == '`':
                    expr = value.strip('`')
                    result = eval(expr)
                    key_seq = old_key.split('.')
                    key_seq.append(key)
                    self[key_seq] = result

    def _update_params(self):
        self.fixed = []
        self.variable = []
        self.variable_thickness = []
        self.optimized = []
        for par,data in self.data['Simulation']['params'].items():
            if data['type'] == 'fixed':
                self.fixed.append(('Simulation','params',par))
            elif data['type'] == 'variable':
                if par == 'thickness':
                    self.variable_thickness.append(('Simulation','params',par))
                else:
                    self.variable.append(('Simulation','params',par))
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
                    if par == 'thickness':
                        self.variable_thickness.append(('Layers',layer,'params',par))
                    else:
                        self.variable.append(('Layers',layer,'params',par))
                elif data['type'] == 'optimized':
                    self.optimized.append(('Layers',layer,'params',par))
                else:
                    loc = '.'.join('Layers',layer,'params',par)
                    raise ValueError('Specified an invalid config type at {}'.format(loc))

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
        sorted_layers = OrderedDict(sorted(adict.items(),key=lambda tup: tup[1]['order'],
                                           reverse=reverse))
        return sorted_layers


class Simulator():

    def __init__(self,conf,log_level='info'):
        conf.interpolate()
        conf.evaluate()
        self.conf = conf
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        period = self.conf['Simulation']['params']['array_period']['value']
        self.id = make_hash(conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'],self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir
        self.log = configure_logger(log_level,'Simulator',sim_dir,'sim.log')
        self.s4 = S4.New(Lattice=((period,0),(0,period)),NumBasis=numbasis)

    def update_id(self):
        """Update sim id. Used after changes are made to the config"""
        self.id = make_hash(self.conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'],self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir

    def configure(self):
        """Configure options for the RCWA solver"""
        self.s4.SetOptions(**self.conf['Solver'])

    def _get_epsilon(self,path):
        """Returns complex dielectric constant for a material by pulling in nk text file, interpolating,
        computing nk values at freq, and converting"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        # Get data
        freq_vec,n_vec,k_vec = np.loadtxt(path,skiprows=1,unpack=True)
        # Get n and k at specified frequency via interpolation
        f_n = spi.interp1d(freq_vec,n_vec,kind='nearest',
                           bounds_error=False,fill_value='extrapolate')
        f_k = spi.interp1d(freq_vec,k_vec,kind='nearest',
                           bounds_error=False,fill_value='extrapolate')
        n,k = f_n(freq),f_k(freq)
        # Convert to dielectric constant
        # NOTE!!: This assumes the relative magnetic permability (mew) is 1
        epsilon_real = n**2 - k**2
        epsilon_imag = 2*n*k
        epsilon = complex(epsilon_real,epsilon_imag)
        return epsilon

    def _get_incident_amplitude(self):
        """Returns the incident amplitude of a wave depending on frequency"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        polar_angle = self.conf['Simulation']['params']['polar_angle']['value']
        path = self.conf['Simulation']['input_power']
        # Get data
        freq_vec,p_vec = np.loadtxt(path,skiprows=1,unpack=True)
        # Get p at freq by interpolation
        f_p = spi.interp1d(freq_vec,p_vec,kind='nearest',
                           bounds_error=False,fill_value='extrapolate')
        # We need to reduce total incident power depending on incident polar
        # angle
        E = np.sqrt(constants.c*constants.mu_0*f_p(freq))*np.cos(polar_angle)
        return E

    def set_excitation(self):
        """Sets the exciting plane wave for the simulation"""
        f_phys = self.conf['Simulation']['params']['frequency']['value']
        self.log.info('Physical Frequency = %E'%f_phys)
        c_conv = constants.c/self.conf['Simulation']['base_unit']
        f_conv = f_phys/c_conv
        self.s4.SetFrequency(f_conv)
        E_mag = self._get_incident_amplitude()
        polar = self.conf['Simulation']['params']['polar_angle']['value']
        azimuth = self.conf['Simulation']['params']['azimuthal_angle']['value']
        # To define circularly polarized light, basically just stick a j (imaginary number) in front of
        # one of your components. The handedness is determined by the component you stick the j in front
        # of. From POV of source, looking away from source toward direction of propagation, right handed
        # circular polarization has j in front of y component. Magnitudes are the same. This means
        # E-field vector rotates clockwise when observed from POV of source. Left handed =
        # counterclockwise.
        # In S4, if indicent angles are 0, p-polarization is along x-axis. The minus sign on front of the
        # x magnitude is just to get things to look like Anna's simulations.
        polarization = self.conf['Simulation']['polarization']
        if polarization == 'rhcp':
            # Right hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(E_mag,0),
                                           pAmplitude=complex(0,E_mag))
        elif polarization == 'lhcp':
            # Left hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(0,E_mag),
                                           pAmplitude=complex(E_mag,0))
        elif polarization == 'lpx':
            # Linearly polarized along x axis (TM polarixation)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(0,0),
                                           pAmplitude=complex(E_mag,0))
        elif polarization == 'lpy':
            # Linearly polarized along y axis (TE polarization)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(E_mag,0),
                                           pAmplitude=complex(0,0))
        else:
            raise ValueError('Invalid polarization specification')

    def build_device(self):
        """Build the device geometry"""

        # First define all the materials
        for mat,mat_path in self.conf['Materials'].items():
            eps = self._get_epsilon(mat_path)
            self.s4.SetMaterial(Name=mat,Epsilon=eps)
        self.s4.SetMaterial(Name='vacuum',Epsilon=complex(1,0))
        # We need to properly sort our layers because order DOES matter. Light
        # will be incident upon the first layer specified
        for layer,ldata in sorted(self.conf['Layers'].items(),key=lambda tup: tup[1]['order']):
            self.log.info('Building layer: %s'%layer)
            self.log.info('Layer Order %i'%ldata['order'])
            base_mat = ldata['base_material']
            layer_t = ldata['params']['thickness']['value']
            self.s4.AddLayer(Name=layer,Thickness=layer_t,Material=base_mat)
            if 'geometry' in ldata:
                self.log.info('Building geometry in layer: {}'.format(layer))
                for shape,sdata in sorted(ldata['geometry'].items(),key=lambda tup: tup[1]['order']):
                    self.log.info('Building object {} of type {} at order'
                                  ' {}'.format(shape,sdata['type'],sdata['order']))
                    shape_mat = sdata['material']
                    if sdata['type'] == 'circle':
                        rad = sdata['radius']
                        cent = sdata['center']
                        coord = (cent['x'],cent['y'])
                        self.s4.SetRegionCircle(Layer=layer,Material=shape_mat,Center=coord,
                                                Radius=rad)
                    else:
                        raise NotImplementedError('Shape %s is not yet implemented'%sdata['type'])

    def get_height(self):
        """Get the total height of the device"""
        height = 0
        for layer,ldata in self.conf['Layers'].items():
            layer_t = ldata['params']['thickness']['value']
            height += layer_t
        return height

    def set_lattice(self,period):
        """Updates the S4 simulation object with a new array period"""
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        self.s4 = S4.New(Lattice=((period,0),(0,period)),NumBasis=numbasis)

    def set_basis(self,numbasis):
        """Updates the S4 simulation object with a new set of basis terms"""
        period = self.conf['Simulation']['params']['array_period']['value']
        self.s4 = S4.New(Lattice=((period,0),(0,period)),NumBasis=numbasis)

    def update_thicknesses(self):
        """Updates all layer thicknesses without rebuilding the device. This
        allows reuse of any layer eigenmodes already computed and utilizes a
        fundamental efficiency of the RCWA solver"""
        for layer,ldata in self.conf['Layers'].items():
            thickness = ldata['params']['thickness']['value']
            self.s4.SetLayerThickness(Layer=layer,Thickness=thickness)

    def get_field(self):
        """Constructs and returns a 2D numpy array of the vector electric
        field. The field components are complex numbers"""
        x_samp = self.conf['Simulation']['x_samples']
        y_samp = self.conf['Simulation']['y_samples']
        z_samp = self.conf['Simulation']['z_samples']
        height = self.get_height()
        dz = height/z_samp
        zvec = np.arange(0,height+dz,dz)
        arr = np.zeros((x_samp*y_samp*len(zvec),9))
        count = 0
        for z in zvec:
            E,H = self.s4.GetFieldsOnGrid(z=z,NumSamples=(x_samp,y_samp),Format='Array')
            xcount = 0
            for xval in E:
                ycount = 0
                for yval in xval:
                    tmp = [(c.real,c.imag) for c in yval]
                    row = [xcount,ycount,z]+list(chain(*tmp))
                    arr[count,:] = row
                count += 1
                ycount += 1
            xcount += 1
        return arr

    def save_field(self):
        # Gets the fields throughout the device
        if self.conf['General']['adaptive_convergence']:
            efield,numbasis,conv = self.adaptive_convergence()
            if conv:
                out = os.path.join(self.dir,'converged_at.txt')
            else:
                out = os.path.join(self.dir,'not_converged_at.txt')
            with open(out,'w') as outf:
                outf.write('{}\n'.format(numbasis))
        else:
            efield = self.get_field()
        out = os.path.join(self.dir,self.conf["General"]["base_name"]+'.E')
        headers = ['x','y','z','Ex','Ey','Ez']
        if self.conf['General']['save_as'] == 'text':
            np.savetxt(out,efield,header=','.join(headers))
        elif self.conf['General']['save_as'] == 'npz':
            np.savez_compressed(out,headers=headers,data=efield)
        else:
            raise ValueError('Invalid file type specified in config')

    def get_fluxes(self):
        """Get the fluxes at the top and bottom of each layer. Returns a dict
        where the keys are the layer name and the values are a length 2 tuple
        with the forward component first and the backward component second. The
        components are complex numbers"""
        self.log.info('Computing fluxes ...')
        flux_dict = {}
        for layer,ldata in self.conf['Layers'].items():
            self.log.info('Computing fluxes through layer: %s'%layer)
            # This gets flux at top of layer
            forw,back = self.s4.GetPowerFlux(Layer=layer)
            flux_dict[layer] = (forw,back)
            # This gets flux at the bottom
            offset = ldata['params']['thickness']['value']
            forw,back = self.s4.GetPowerFlux(Layer=layer,zOffset=offset)
            key = layer+'_bottom'
            flux_dict[key] = (forw,back)
        return flux_dict

    def save_fluxes(self):
        """Saves the fluxes at layer interfaces to a file"""
        path = os.path.join(self.dir,'fluxes.dat')
        fluxes = self.get_fluxes()
        with open(path,'w') as out:
            out.write('# Layer,ForwardReal,BackwardReal,ForwardImag,BackwardImag\n')
            for layer,ldata in fluxes.items():
                forw,back = ldata
                row = '%s,%s,%s,%s,%s\n'%(layer,forw.real,back.real,
                                          forw.imag,back.imag)
                out.write(row)

    def calc_diff(self,d1,d2,exclude=False):
        """Calculate the percent difference between two vector fields"""
        # This is a 2D table where each row contains the differences between
        # each electric field component at the corresponding point in space
        # squared. This is the magnitude squared of the difference vector
        # between the two datasets at each sampling point
        diffs_sq = (d1[:,3:] - d2[:,3:])**2
        # Sum along the columns to get the sum of the squares of the components
        # of the difference vector. This should be a 1D array
        mag_diffs = np.sum(diffs_sq,axis=1)
        # Now compute the norm(E)^2 of the comparison sim at each sampling
        # point
        normsq = np.sum(d1**2,axis=1)
        # We define the percent difference as the ratio of the sums of the
        # difference vector magnitudes to the comparison field magnitudes,
        # squared rooted.
        # TODO: This seems like a somewhat shitty metric that washes out any
        # large localized deviations. Should test other metrics
        diff = np.sqrt(np.sum(mag_diffs)/np.sum(normsq))
        self.log.info('Percent difference = {}'.format(diff))
        return diff

    def adaptive_convergence(self):
        """Performs adaptive convergence by checking the error between vector
        fields for simulations with two different numbers of basis terms.
        Returns the field array, last number of basis terms simulations, and a
        boolean representing whether or not the simulation is converged"""
        self.log.info('Beginning adaptive convergence procedure')
        start_basis = self.conf['Simulation']['params']['numbasis']['value']
        basis_step = self.conf['General']['basis_step']
        field1 = self.get_field()
        max_diff = self.conf['General']['max_diff']
        max_iter = self.conf['General']['max_iter']
        percent_diff = 100
        iter_count = 0
        while percent_diff > max_diff and iter_count < max_iter:
            new_basis = start_basis + basis_step
            self.log.info('Checking error between {} and {} basis'
                          ' terms'.format(start_basis,new_basis))
            self.set_basis(new_basis)
            self.build_device()
            self.set_excitation()
            field2 = self.get_field()
            percent_diff = self.calc_diff(field1,field2)
            start_basis = new_basis
            field1 = field2
            iter_count += 1

        if percent_diff > max_diff:
            self.log.warning('Exceeded maximum number of iterations')
            return field2,new_basis,False
        else:
            self.log.info('Converged at {} basis terms'.format(new_basis))
            return field2,new_basis,True
