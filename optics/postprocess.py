import numpy as np
from scipy import interpolate
import scipy.constants as c
import argparse as ap
import os
import configparser as confp 
import re
import glob
import logging
from collections import OrderedDict
import matplotlib
# Enables saving plots over ssh
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# Literally just for the initial data load
import pandas

def configure_logger(level,logger_name,log_dir,logfile):
    # Get numeric level safely
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
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

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
    # This preserves case sensitivity
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

class Processor(object):
    """Base data processor class that has some methods every other processor needs"""
    def __init__(self,global_conf,sims=None,sim_groups=None):
        self.log = logging.getLogger('postprocess')
        self.log.debug("Processor base init")
        self.gconf = global_conf
        # This allows us to pass in the list of simulation config objects if we have them, 
        # otherwise just collect them
        if not sims or not sim_groups:
            self.collect_sims()
        else:
            self.sims = sims
            self.sims_groups = sim_groups
        # Sort on the sim dir to prevent weirdness when calculating convergence. Sorts by ascending
        # param values and works for multiple variable params
        self.sort_sims()
        self.sim = None
        self.e_lookup = OrderedDict([('x',0),('y',1),('z',2),('Ex_real',3),('Ey_real',4),('Ez_real',5),
                         ('Ex_imag',6),('Ey_imag',7),('Ez_imag',8)])
        self.h_lookup = OrderedDict([('x',0),('y',1),('z',2),('Hx_real',3),('Hy_real',4),('Hz_real',5),
                         ('Hx_imag',6),('Hy_imag',7),('Hz_imag',8)])

    def collect_sims(self):
        self.sims = []
        self.sim_groups = []
        for root,dirs,files in os.walk(self.gconf.get('General','basedir')):
            if 'sim_conf.ini' in files:
                obj = parse_file(os.path.join(root,'sim_conf.ini'))
                self.sims.append(obj)
            if 'sorted_sweep_conf.ini' in files:
                if 'logs' in dirs:
                    dirs.remove('logs')
                conf_paths = [os.path.join(root,simdir,'sim_conf.ini') for simdir in dirs]
                self.log.debug('Sim group confs: %s',str(conf_paths))
                self.sim_groups.append(list(map(parse_file,conf_paths)))

    def sort_sims(self):
        """Sorts simulations by their parameters the way a human would. Called human sorting or
        natural sorting. Thanks stackoverflow"""
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(sim):
            self.log.debug(sim)
            self.log.debug(sim.sections())
            text = sim.get('General','sim_dir')
            return [ atoi(c) for c in re.split('(\d+)', text) ]

        self.sims.sort(key=natural_keys)
        for group in self.sim_groups:
            paths = [sim.get('General','sim_dir') for sim in group]
            self.log.debug('Group paths before sorting: %s',str(paths))
            group.sort(key=natural_keys)
            paths = [sim.get('General','sim_dir') for sim in group]
            self.log.debug('Group paths after sorting: %s',str(paths))

    def set_sim(self,sim):
        """Wrapper method to set the current simulation and retrieve and set the data attributes"""
        self.log.info('Setting sim to %s',sim.get('General','sim_dir'))
        self.sim = sim
        self.e_data,self.e_lookup,self.h_data,self.h_lookup,self.pos_inds = self.get_data(sim)

    def get_data(self,sim):
        """Returns the E and H data for this particular sim"""
        self.log.info('Collecting data for sim %s',sim.get('General','sim_dir'))
        sim_path = sim.get('General','sim_dir')
        base_name = self.gconf.get('General','base_name')
        ftype = self.gconf.get('General','save_as')
        # If data was saved into text files
        if ftype == 'text':
            e_path = os.path.join(sim_path,base_name+'.E.crnch')
            h_path = os.path.join(sim_path,base_name+'.H.crnch')
            # Load E field data
            try: 
                e_data = np.loadtxt(e_path)
                with open(e_path,'r') as efile:
                    hline = efile.readlines()[0]
                    if hline[0] == '#':
                        # We have headers in the file
                        e_headers = hline.strip('#\n').split(',')
                        e_lookup = {e_headers[ind]:ind for ind in range(len(e_headers))}
                        self.log.debug('Here is the E field header lookup: %s',str(e_lookup))
                    else:
                        self.log.debug('File is missing headers')
                
            except FileNotFoundError:
                self.log.error('Following file missing: %s',e_path)
            # Load the H field data
            try:
                h_data = np.loadtxt(h_path)
                with open(h_path,'r') as hfile:
                    hline = hfile.readlines()[0]
                    if hline[0] == '#':
                        # We have headers in the file
                        h_headers = hline.strip('#\n').split(',')
                        h_lookup = {h_headers[ind]:ind for ind in range(len(h_headers))}
                        self.log.debug('Here is the H field header lookup: %s',str(h_lookup))
                    else:
                        self.log.debug('File is missing headers')
            except FileNotFoundError:
                self.log.error('Following file missing: %s',h_path)
            pos_inds = np.zeros((e_data.shape[0],3))
            pos_inds[:,:] = e_data[:,0:3] 
        # If data was saved in in npz format
        elif ftype == 'npz':
            # Get the paths
            e_path = os.path.join(sim_path,base_name+'.E.npz')
            h_path = os.path.join(sim_path,base_name+'.H.npz')
            try: 
                # Get the headers and the data
                with np.load(e_path) as loaded:
                    e_data = loaded['data']
                    e_lookup = loaded['headers'][0]
                    self.log.debug(str(type(e_lookup)))
                    self.log.debug(str(e_lookup))
                    self.log.debug('Here is the E field header lookup: %s',str(e_lookup))
            except IOError:
                self.log.error('Following file missing or unloadable: %s',e_path)
            try:
                with np.load(h_path) as loaded:
                    h_data = loaded['data']            
                    h_lookup = loaded['headers'][0]
                    self.log.debug('Here is the H field header lookup: %s',str(h_lookup))
            except IOError:
                self.log.error('Following file missing or unloadable: %s',h_path)
            pos_inds = np.zeros((e_data.shape[0],3))
            pos_inds[:,:] = e_data[:,0:3]
        else:
            raise ValueError('Incorrect file type specified in [General] section of config file')
        self.log.info('Collection complete!')
        return e_data,e_lookup,h_data,h_lookup,pos_inds

    def get_scalar_quantity(self,quantity):
        self.log.debug('Retrieving scalar quantity %s',str(quantity))
        self.log.debug(type(self.e_lookup))
        self.log.debug('E Header: %s',str(self.e_lookup))
        self.log.debug('H Header: %s',str(self.h_lookup))
        try:
            col = self.e_lookup[quantity]
            self.log.debug(col)
            self.log.debug('Column of E field quantity %s is %s',str(quantity),str(col))
            return self.e_data[:,col]
        except KeyError:
            col = self.h_lookup[quantity]
            self.log.debug(col)
            self.log.debug('Column of H field quantitty %s is %s',str(quantity),str(col))
            return self.h_data[:,col]
        except KeyError:
            self.log.error('You attempted to retrieve a quantity that does not exist in the e and h \
                    matrices')
            raise

class Cruncher(Processor):
    """Crunches all the raw data. Calculates quantities specified in the global config file and
    either appends them to the existing data files or creates new ones as needed"""

    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug("This is THE CRUNCHER!!!!!")
    
    def get_data(self,sim):
        """We need to override the base Processor get_data function because the Cruncher object is
        the only object that touches raw data files spit out by the simulations. Every other child 
        of Processor consumes files parsed and tidied up by Cruncher"""
        self.log.debug('CRUNCHER GET_DATA')
        self.log.info('Collecting raw data for sim %s',sim.get('General','sim_dir'))
        sim_path = sim.get('General','sim_dir')
        base_name = self.gconf.get('General','base_name')
        ftype = self.gconf.get('General','save_as')
        e_path = os.path.join(sim_path,base_name+'.E')
        h_path = os.path.join(sim_path,base_name+'.H')
        # Load E field data
        try: 
            #e_data = np.loadtxt(e_path)
            # pandas read_csv is WAY faster than loadtxt
            d = pandas.read_csv(e_path,delim_whitespace=True)
            e_data = d.as_matrix()
        except FileNotFoundError:
            self.log.error('Following file missing: %s',e_path)
            raise
        # Load the H field data
        try:
            #h_data = np.loadtxt(h_path)
            d = pandas.read_csv(h_path,delim_whitespace=True)
            h_data = d.as_matrix()
        except FileNotFoundError:
            self.log.error('Following file missing: %s',h_path)
            raise
        pos_inds = np.zeros((e_data.shape[0],3))
        pos_inds[:,:] = e_data[:,0:3] 
        self.log.debug('E shape after getting: %s',str(e_data.shape))        
        self.log.info('Collection complete!')
        return e_data,self.e_lookup,h_data,self.h_lookup,pos_inds

    def crunch(self):
        self.log.info('Beginning data crunch ...')
        for sim in self.sims:
            # Set it as the current sim and grab its data
            self.set_sim(sim)
            self.log.info('Crunching data for %s',
                          os.path.basename(self.sim.get('General','sim_dir')))
            # For each quantity 
            for quant,args in self.gconf.items('Cruncher'):
                self.log.info('Computing %s with args %s',str(quant),str(args))
                for argset in args.split(';'):
                    if argset:
                        self.calculate(quant,argset.split(','))
                    else:
                        self.calculate(quant,[])
                self.log.debug('E lookup: %s',str(self.e_lookup))
                self.log.debug('H lookup: %s',str(self.h_lookup))
            self.write_data()
    
    def calculate(self,quantity,args):
        try:
            getattr(self,quantity)(*args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity,exc_info=True,stack_info=True)
            raise

    def write_data(self):
        """Writes the crunched data"""
        # Get the current path
        base = self.sim.get('General','sim_dir')
        fname = self.sim.get('General','base_name')
        epath = os.path.join(base,fname+'.E')
        hpath = os.path.join(base,fname+'.H')
        # Save matrices in specified file tipe 
        self.log.debug('Here are the E matrix headers: %s',str(self.e_lookup))
        #self.log.debug('Here is the E matrix: \n %s',str(self.e_data))
        self.log.debug('Here are the H matrix headers: %s',str(self.h_lookup))
        #self.log.debug('Here is the H matrix: \n %s',str(self.h_data))
        ftype = self.gconf.get('General','save_as') 
        if ftype == 'text':
            epath = epath+'.crnch'
            hpath= hpath+'.crnch'
            np.savetxt(epath,self.e_data,header=','.join(self.e_lookup.keys()))
            np.savetxt(hpath,self.h_data,header=','.join(self.h_lookup.keys()))
        elif ftype == 'npz':
            # Save the headers and the data
            np.savez(epath,headers = np.array([self.e_lookup]), data = self.e_data)
            np.savez(hpath,headers = np.array([self.h_lookup]), data = self.h_data)
        else:
            raise ValueError('Specified saving in an unsupported file format')

    def normE(self):
        """Calculate and returns the norm of E"""
        
        if not hasattr(self,'e_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of E and add it to our data
        E_mag = np.zeros(self.e_data.shape[0])
        for i in range(3,9):
            E_mag += self.e_data[:,i]*self.e_data[:,i]
        E_mag = np.sqrt(E_mag)
        
        # This approach is 4 times faster than np.column_stack()
        self.log.debug('E mat shape: %s',str(self.e_data.shape))
        dat = np.zeros((self.e_data.shape[0],self.e_data.shape[1]+1))
        self.log.debug('dat mat shape: %s',str(dat.shape))
        dat[:,:-1] = self.e_data
        dat[:,-1] = E_mag
        self.e_data = dat 
        # Now append this quantity and its column the the header dict
        self.e_lookup['normE'] = dat.shape[1]-1 
        return E_mag

    def normEsquared(self):
        """Calculates and returns normE squared"""
        if not hasattr(self,'e_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of E and add it to our data
        E_magsq = np.zeros(self.e_data.shape[0])
        for i in range(3,9):
            E_magsq += self.e_data[:,i]*self.e_data[:,i]
        
        # This approach is 4 times faster than np.column_stack()
        dat = np.zeros((self.e_data.shape[0],self.e_data.shape[1]+1))
        dat[:,:-1] = self.e_data
        dat[:,-1] = E_magsq
        self.e_data = dat 
        # Now append this quantity and its column the the header dict
        self.e_lookup['normEsquared'] = dat.shape[1]-1 
        return E_magsq

    def normH(self):
        """Calculate and returns the norm of H"""
        
        if not hasattr(self,'h_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of H and add it to our data. This loops through each components real and
        # imaginary parts and squared it (which is what would happen if you took the complex number
        # for each component and multiplied it by its conjugate). 
        H_mag = np.zeros(self.h_data.shape[0])
        for i in range(3,9):
            H_mag += self.h_data[:,i]*self.h_data[:,i]
        H_mag = np.sqrt(H_mag)
        
        # This approach is 4 times faster than np.column_stack()
        dat = np.zeros((self.h_data.shape[0],self.h_data.shape[1]+1))
        dat[:,:-1] = self.h_data
        dat[:,-1] = H_mag
        self.h_data = dat 
        # Now append this quantity and its column the the header dict
        self.h_lookup['normH'] = dat.shape[1]-1 
        return H_mag
    
    def normHsquared(self):
        """Calculates and returns the norm of H squared"""
        
        if not hasattr(self,'h_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of H and add it to our data
        H_magsq = np.zeros(self.h_data.shape[0])
        for i in range(3,9):
            H_magsq += self.h_data[:,i]*self.h_data[:,i]
        # This approach is 4 times faster than np.column_stack()
        dat = np.zeros((self.h_data.shape[0],self.h_data.shape[1]+1))
        dat[:,:-1] = self.h_data
        dat[:,-1] = H_magsq
        self.h_data = dat 
        # Now append this quantity and its column the the header dict
        self.h_lookup['normHsquared'] = dat.shape[1]-1 
        return H_magsq

    def get_nk(self,path,freq):
        """Returns functions to compute index of refraction components n and k at a given
        frequency"""
        # Get data
        freq_vec,n_vec,k_vec = np.loadtxt(path,skiprows=1,unpack=True)
        # Get n and k at specified frequency via interpolation 
        f_n = interpolate.interp1d(freq_vec,n_vec,kind='linear',
                                   bounds_error=False,fill_value='extrapolate')
        f_k = interpolate.interp1d(freq_vec,k_vec,kind='linear',
                                   bounds_error=False,fill_value='extrapolate')
        return f_n(freq), f_k(freq)

    def genRate(self):
        """Computes and return the generation rate at each point in space"""

        # We need to compute normEsquared before we can compute the generation rate
        normEsq = self.get_scalar_quantity('normEsquared') 
        gvec = np.zeros_like(normEsq)
        # Convenient lambda function to actual compute G
        fact = c.epsilon_0/c.hbar
        # Get the indices of refraction at this frequency
        freq = self.sim.get('Parameters','frequency')
        nk = {mat[0]:(self.get_nk(mat[1],freq)) for mat in self.gconf.items('Materials')}
        self.log.debug(nk) 
        height = self.sim.getfloat('Parameters','total_height')
        # Get spatial discretization
        z_samples = self.gconf.getint('General','z_samples')
        x_samples = self.gconf.getint('General','x_samples')
        y_samples = self.gconf.getint('General','y_samples')
        dz = height/z_samples
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/x_samples
        dy = period/y_samples
        # Get boundaries between layers
        air_ito = self.sim.getfloat('Parameters','air_t')
        ito_nw = self.sim.getfloat('Parameters','ito_t')+air_ito
        nw_sio2 = self.sim.getfloat('Parameters','alinp_height')+ito_nw
        sio2_sub = self.sim.getfloat('Parameters','sio2_height')+nw_sio2
        air_line = sub_line+self.sim.getfloat('Parameters','substrate_t')
        # Compute ITO generation (note air generation is already set to zero)
        start = int(air_ito/dz)*x_samples*y_samples 
        end = int(ito_nw/dz)*x_samples*y_samples 
        gvec[start:end] = fact*nk['ITO'][0]*nk['ITO'][1]*normEsq[start:end]
        # Compute nw generation
        start = end
        end = start + int(self.sim.getfloat('Parameters','alinp_height')/dz) 
        xvec = np.linspace(0,period,x_samples)
        yvec = np.linspace(0,period,y_samples)
        center = period/2
        nw_radius = self.sim.getfloat('Parameters','nw_radius') 
        # Loop through each z layer in nw with AlInP shell
        counter = start
        for layer in range(start,end):
            for x in xvec:
                for y in yvec:
                    if (x-center)**2 + (y-center)**2 <= nw_radius:
                        gvec[counter] = fact*nk['GaAs'][0]*nk['GaAs'][1]*normEsq[counter] 
                    elif nw_radius < (x-center)**2 + (y-center)**2 <= core_rad:
                        gvec[counter] = fact*nk['AlInP'][0]*nk['AlInP'][1]*normEsq[counter]
                    else:
                        gvec[counter] = fact*nk['Cyclotene'][0]*nk['Cyclotene'][1]*normEsq[counter]
                    counter += 1
        # So same for SiO2 shell
        start = counter
        end = start + self.sim.float('Parameters','sio2_height')
        for layer in range(start,end):
            for x in xvec:
                for y in yvec:
                    if (x-center)**2 + (y-center)**2 <= nw_radius:
                        gvec[counter] = fact*nk['GaAs'][0]*nk['GaAs'][1]*normEsq[counter] 
                    elif nw_radius < (x-center)**2 + (y-center)**2 <= core_rad:
                        gvec[counter] = fact*nk['SiO2'][0]*nk['SiO2'][1]*normEsq[counter]
                    else:
                        gvec[counter] = fact*nk['Cyclotene'][0]*nk['Cyclotene'][1]*normEsq[counter]
                    counter += 1
        # The rest is just the substrate
        gvec[counter:] = fact*nk['GaAs'][0]*nk['GaAs'][1]*normEsq[counter:]
        # This approach is 4 times faster than np.column_stack()
        dat = np.zeros((self.e_data.shape[0],self.e_data.shape[1]+1))
        dat[:,:-1] = self.e_data
        dat[:,-1] = gvec
        self.e_data = dat 
        # Now append this quantity and its column the the header dict
        self.e_lookup['generation_rate'] = dat.shape[1]-1 
        return gvec

class Global_Cruncher(Processor):
    """Computes global quantities for an entire run, instead of local quantities for an individual
    simulation"""
    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug('This is the global cruncher') 

    def calculate(self,quantity,args):
        try:
            getattr(self,quantity)(*args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity,exc_info=True,stack_info=True)
            raise

    def crunch(self):
        self.log.info('Beginning global data crunch ...')
        # For each quantity 
        for quant,args in self.gconf.items('Global_Cruncher'):
            self.log.info('Computing %s with args %s',str(quant),str(args))
            for argset in args.split(';'): 
                if argset:
                    self.calculate(quant,argset.split(','))
                else:
                    self.calculate(quant,[])

    def diff_sq(self,x,y):
        """Returns the magnitude of the difference vector squared between two vector fields at each
        point in space"""
        if x.size != y.size:
            self.log.error("You have attempted to compare datasets with an unequal number of points!!!!")
            quit()
        # Calculate the magnitude of the difference vector SQUARED at each point in space
        # This is mag(vec(x) - vec(y))^2 at each point in space. This should be a 1D array
        # with # of elements = # sampling points
        mag_diff_vec = np.sum((x-y)**2,axis=1)
        return mag_diff_vec

    def get_slice(self):
        """Returns indices for data that strip out air and substrate regions"""
        x_samples = self.gconf.getint('General','x_samples')
        y_samples = self.gconf.getint('General','y_samples')
        z_samples = self.gconf.getint('General','z_samples')
        h = sum((self.gconf.getfloat('Fixed Parameters','nw_height'),self.gconf.getfloat('Fixed Parameters','substrate_t'),
                self.gconf.getfloat('Fixed Parameters','air_t'),self.gconf.getfloat('Fixed Parameters','ito_t')))
        arr = np.linspace(0,h,z_samples)
        dz = arr[1] - arr[0]
        start_plane = int(round(self.gconf.getfloat('Fixed Parameters','air_t')/dz))
        start = start_plane*(x_samples*y_samples)
        end_plane = int(round(sum((self.gconf.getfloat('Fixed Parameters','nw_height'),self.gconf.getfloat('Fixed Parameters','air_t'),
                self.gconf.getfloat('Fixed Parameters','ito_t')))/dz))
        end = end_plane*(x_samples*y_samples)
        return start,end

    def get_comp_vec(self,field,start,end):
        """Returns the comparison vector"""
        # Compare all other sims to our best estimate, which is sim with highest number of
        # basis terms (last in list cuz sorting)

        # Get the proper file extension depending on the field.
        if field == 'E':
            ext = '.E'
            # Get the comparison vector
            vec1 = self.e_data[start:end,3:9]
            normvec = self.get_scalar_quantity('normE')
            normvec = normvec[start:end]**2
        elif field == 'H':
            ext = '.H'
            vec1 = self.h_data[start:end,3:9]
            normvec = self.get_scalar_quantity('normH')
            normvec = normvec[start:end]**2
        else:
            self.log.error('The quantity for which you want to compute the error has not yet been calculated')
            quit()
        return vec1, normvec, ext

    def local_error(self,field,exclude=False):
        """Computes the average of the local error between the vector fields of two simulations at
        each point in space"""
        self.log.info('Running the local error computation for quantity %s',field) 
        # If we need to exclude calculate the indices
        if exclude:
            start,end = self.get_slice()    
            excluded = '_excluded'
        else:
            start = 0
            end = None
            excluded = ''
        for group in self.sim_groups:
            base = group[0].get('General','basedir')
            errpath = os.path.join(base,'localerror_%s%s.dat'%(field,excluded))
            with open(errpath,'w') as errfile:
                self.log.info('Computing local error for sweep %s',base)
                # Set comparison sim to current sim
                self.set_sim(group[-1])
                # Get the comparison vector
                vec1,normvec,ext = self.get_comp_vec(field,start,end)    
                # For all other sims in the groups, compare to best estimate and write to error file 
                for i in range(0,len(group)-1):
                    sim2 = group[i]
                    e_data,elook,h_data,hlook,inds = self.get_data(sim2)
                    if field == 'E': 
                        vec2 = e_data[start:end,3:9]
                    elif field == 'H':
                        vec2 = h_data[start:end,3:9]
                    self.log.info("Computing local error between numbasis %i and numbasis %i",
                                  self.sim.getint('Parameters','numbasis'),
                                  sim2.getint('Parameters','numbasis'))
                    # Get the array containing the magnitude of the difference vector at each point
                    # in space
                    self.log.debug('vec1 shape: %s',str(vec1.shape))
                    self.log.debug('vec2 shape: %s',str(vec2.shape))
                    mag_diff_vec = self.diff_sq(vec1,vec2)
                    # Normalize the magnitude squared of the difference vector by the magnitude squared of
                    # the local electric field of the comparison simulation at each point in space
                    if len(mag_diff_vec) != len(normvec):
                        self.log.error("The normalization vector has an incorrect number of elements!!!")
                        quit()
                    norm_mag_diff = mag_diff_vec / normvec
                    # Compute the average of the normalized magnitude of all the difference vectors 
                    avg_diffvec_mag = np.sum(norm_mag_diff)/norm_mag_diff.size
                    self.log.info(str(avg_diffvec_mag))
                    errfile.write('%i,%f\n'%(sim2.getint('Parameters','numbasis'),avg_diffvec_mag))

    def global_error(self,field,exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell"""

        self.log.info('Running the global error computation for quantity %s',field) 
        # If we need to exclude calculate the indices
        if exclude:
            start,end = self.get_slice()    
            excluded = '_excluded'
        else:
            start = 0
            end = None
            excluded = ''
        for group in self.sim_groups:
            base = group[0].get('General','basedir')
            errpath = os.path.join(base,'globalerror_%s%s.dat'%(field,excluded))
            with open(errpath,'w') as errfile:
                self.log.info('Computing global error for sweep %s',base)
                # Set comparison sim to current sim
                self.set_sim(group[-1])
                # Get the comparison vector
                vec1,normvec,ext = self.get_comp_vec(field,start,end)    
                # For all other sims in the groups, compare to best estimate and write to error file 
                for i in range(0,len(group)-1):
                    sim2 = group[i]
                    e_data,elook,h_data,hlook,inds = self.get_data(sim2)
                    if field == 'E': 
                        vec2 = e_data[start:end,3:9]
                    elif field == 'H':
                        vec2 = h_data[start:end,3:9]
                    self.log.info("Computing global error between numbasis %i and numbasis %i",
                                  self.sim.getint('Parameters','numbasis'),
                                  sim2.getint('Parameters','numbasis'))
                    # Get the array containing the magnitude of the difference vector at each point
                    # in space 
                    mag_diff_vec = self.diff_sq(vec1,vec2)
                    # Check for equal lengths between norm array and diff mag array 
                    if len(mag_diff_vec) != len(normvec):
                        self.log.error("The normalization vector has an incorrect number of elements!!!")
                        quit()
                    # Error as a percentage should be the square root of the ratio of sum of mag diff vec 
                    # squared to mag efield squared
                    error = np.sqrt(np.sum(mag_diff_vec)/np.sum(normvec))
                    self.log.info(str(error))
                    errfile.write('%i,%f\n'%(sim2.getint('Parameters','numbasis'),error))

class Plotter(Processor):
    """Plots all the things listed in the config file"""
    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug("This is the plotter")
    
    def plot(self):
        self.log.info("Beginning local plotter method ...")
        for sim in self.sims:
            # Set it as the current sim and grab its data
            self.set_sim(sim)
            # For each plot 
            self.log.info('Plotting data for sim %s',
                          str(os.path.basename(self.sim.get('General','sim_dir'))))
            for plot,args in self.gconf.items('Plotter'):
                self.log.info('Plotting %s with args %s',str(plot),str(args))
                for argset in args.split(';'):
                    self.log.info('Passing following arg set to function %s: %s',str(plot),str(argset))
                    if argset:
                        self.gen_plot(plot,argset.split(','))
                    else:
                        self.gen_plot(plot,[])

    def gen_plot(self,plot,args):
        try:
            getattr(self,plot)(*args)
        except KeyError:
            self.log.error("Unable to plot the following quantity: %s",
                           quantity,exc_info=True,stack_info=True)
            raise

    def heatmap2d(self,x,y,cs,labels,ptype,draw=False,fixed=None,colorsMap='jet'):
        """A general utility method for plotting a 2D heat map"""
        cm = plt.get_cmap(colorsMap)
        if fixed:
            cNorm = matplotlib.colors.Normalize(vmin=np.amin(5.0), vmax=np.amax(100.0))
        else:
            cNorm = matplotlib.colors.Normalize(vmin=np.amin(cs), vmax=np.amax(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111)
        ax.pcolormesh(x, y, cs,cmap=cm,norm=cNorm,alpha=.5)
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[2])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start,end,0.1))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start,end,0.1))
        fig.suptitle(labels[3])
        # Draw geometric indicators and labels
        if draw:
            if ptype[-1] == 'z':
                self.log.info('draw nanowire circle')
                cent = self.sim.getfloat('Parameters','array_period')/2.0
                rad = self.sim.getfloat('Parameters','nw_radius')
                circ = mpatches.Circle((cent,cent),radius=rad,fill=False)
                ax.add_artist(circ)
            elif ptype[-1] == 'y' or ptype[-1] == 'x':
                self.log.info('draw layers')
                # Draw a line at the interface between each layer
                ito_line = self.sim.getfloat('Parameters','air_t')
                nw_line = self.sim.getfloat('Parameters','ito_t')+ito_line
                sub_line = self.sim.getfloat('Parameters','nw_height')+nw_line
                air_line = sub_line+self.sim.getfloat('Parameters','substrate_t')
                for line_h in [(ito_line,'ITO'),(nw_line,'NW'),(sub_line,'Substrate'),(air_line,'Air')]:
                    x = [0,self.sim.getfloat('Parameters','array_period')]
                    y = [line_h[0],line_h[0]]
                    label_y = line_h[0] + 0.01
                    label_x = x[-1]
                    plt.text(label_x,label_y,line_h[-1],ha='right',family='sans-serif',size=12)
                    line = mlines.Line2D(x,y,linestyle='solid',linewidth=2.0,color='black')
                    ax.add_line(line)
                # Draw two vertical lines to show the edges of the nanowire
                cent = self.sim.getfloat('Parameters','array_period')/2.0
                rad = self.sim.getfloat('Parameters','nw_radius')
                shell = self.sim.getfloat('Parameters','shell_t')
                bottom = self.sim.getfloat('Parameters','ito_t')+ito_line
                top = self.sim.getfloat('Parameters','nw_height')+nw_line
                for x in (cent-rad,cent+rad,cent-rad-shell,cent+rad+shell):
                    xv = [x,x]
                    yv = [bottom,top]
                    line = mlines.Line2D(xv,yv,linestyle='solid',linewidth=2.0,color='black')
                    ax.add_line(line)
        if self.gconf.getboolean('General','save_plots'):
            name = labels[2]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)

    def plane_2d(self,quantity,plane,pval,draw=False,fixed=None):
        """Plots a heatmap of a fixed 2D plane"""
        if plane == 'x' or plane == 'y':
            pval = int(pval)
        else:
            # Find the nearest zval to the one we want. This is necessary because comparing floats
            # rarely works
            height = self.sim.getfloat('Parameters','total_height')
            z_samples = self.sim.getint('General','z_samples')
            desired_val = (height/z_samples)*int(pval)
            ind = np.abs(self.pos_inds[:,2]-desired_val).argmin()
            pval = self.pos_inds[ind,2]
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # Maps planes to an integer for extracting data
        plane_table = {'x': 0,'y': 1,'z':2}
        # Get the scalar values
        scalar = self.get_scalar_quantity(quantity)
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((self.pos_inds[:,0],self.pos_inds[:,1],self.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[plane_table[plane]] == pval])
        # Get all unique values for x,y,z and convert them to actual values not indices
        x,y,z = np.unique(planes[:,0])*dx,np.unique(planes[:,1])*dy,np.unique(planes[:,2])
        # Super hacky and terrible way to fix the minimum and maximum values of the color bar
        # for a plot across all sims
        freq = self.sim.getfloat('Parameters','frequency')
        wvlgth = (c.c/freq)*1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(freq,wvlgth)
        if fixed:
            fixed = tuple(fixed.split(':'))
        if plane == 'x':
            cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
            labels = ('y [um]','z [um]', quantity,title)
            self.heatmap2d(y,z,cs,labels,'plane_2d_x',draw,fixed)
        elif plane == 'y':
            cs = planes[:,-1].reshape(z.shape[0],x.shape[0])
            labels = ('x [um]','z [um]', quantity,title)
            self.heatmap2d(x,z,cs,labels,'plane_2d_y',draw,fixed)
        elif plane == 'z':
            cs = planes[:,-1].reshape(y.shape[0],x.shape[0])
            labels = ('y [um]','x [um]', quantity,title)
            self.heatmap2d(x,y,cs,labels,'plane_2d_z',draw,fixed)
    
    def scatter3d(self,x,y,z,cs,labels,ptype,colorsMap='jet'):
        """A general utility method for scatter plots in 3D"""
        #fig = plt.figure(figsize=(8,6)) 
        #ax = fig.add_subplot(111,projection='3d')
        #colors = cm.hsv(E_mag/max(E_mag))
        #colmap = cm.ScalarMappable(cmap=cm.hsv)
        #colmap.set_array(E_mag)
        #yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
        #cb = fig.colorbar(colmap)
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9,7))
              
        #ax = Axes3D(fig)
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs),edgecolor='none')
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[3])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        fig.suptitle(os.path.basename(self.sim.get('General','sim_dir')))
        if self.gconf.getboolean('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)
    
    def full_3d(self,quantity):
        """Generates a full 3D plot of a specified scalar quantity"""
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # The data just tells you what integer grid point you are on. Not what actual x,y coordinate you
        # are at
        xpos = self.pos_inds[:,0]*dx
        ypos = self.pos_inds[:,1]*dy
        zpos = self.pos_inds[:,2] 
        # Get the scalar
        scalar = self.get_scalar_quantity(quantity)
        labels = ('X [um]','Y [um]','Z [um]',quantity)
        # Now plot! 
        self.scatter3d(xpos,ypos,zpos,scalar,labels,'full_3d')
        
    def planes_3d(self,quantity,xplane,yplane):
        """Plots some scalar quantity in 3D but only along specified x-z and y-z planes"""
        xplane = int(xplane)
        yplane = int(yplane)
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # Get the scalar values
        scalar = self.get_scalar_quantity(quantity) 
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((self.pos_inds[:,0],self.pos_inds[:,1],self.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[0] == xplane or row[1] == yplane])
        planes[:,0] = planes[:,0]*dx
        planes[:,1] = planes[:,1]*dy
        labels = ('X [um]','Y [um]','Z [um]',quantity)
        # Now plot!
        self.scatter3d(planes[:,0],planes[:,1],planes[:,2],planes[:,3],labels,'planes_3d')

    def line_plot(self,x,y,ptype,labels):
        """Make a simple line plot"""
        fig = plt.figure()
        plt.plot(x,y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(labels[2])
        if self.gconf.getboolean('General','save_plots'):
            name = labels[1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)

    def fixed_line(self,quantity,direction,coord1,coord2):
        """Plot a scalar quantity on a line along a the z direction at some pair of
        coordinates in the plane perpendicular to that direction"""
        coord1 = int(coord1)
        coord2 = int(coord2)
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # Get the scalar values
        scalar = self.get_scalar_quantity(quantity) 
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((self.pos_inds[:,0],self.pos_inds[:,1],self.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[0] == coord1 and row[1] == coord2])
        planes[:,0] = planes[:,0]*dx
        planes[:,1] = planes[:,1]*dy
        freq = self.sim.getfloat('Parameters','frequency')
        wvlgth = (c.c/freq)*1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(freq,wvlgth)
        labels = ('Z [um]',quantity,title) 
        ptype = "%s_line_plot_%i_%i"%(direction,coord1,coord2)
        self.line_plot(planes[:,2],planes[:,3],ptype,labels)

class Global_Plotter(Plotter):
    """Plots global quantities for an entire run that are not specific to a single simulation"""
    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug("Global plotter init")

    def plot(self):
        self.log.info('Beginning global plotter method ...')
        for plot,args in self.gconf.items('Global_Plotter'):
            self.log.info('Plotting %s with args %s',str(plot),str(args))
            for argset in args.split(';'):
                self.log.info('Passing following arg set to function %s: %s',str(plot),str(argset))
                if argset:
                    self.gen_plot(plot,argset.split(','))
                else:
                    self.gen_plot(plot,[])

    def convergence(self,quantity,err_type='global',scale='linear'):
        """Plots the convergence of a field across all available simulations"""
        self.log.info('Plotting convergence')
        for group in self.sim_groups:
            base = group[0].get('General','basedir')
            if err_type == 'local':
                fglob = os.path.join(base,'localerror_%s*.dat'%quantity) 
            elif err_type == 'global':
                fglob = os.path.join(base,'globalerror_%s*.dat'%quantity) 
            else:
                log.error('Attempting to plot an unsupported error type')
                quit()
            paths = glob.glob(fglob)
            for path in paths:
                labels = []
                errors = []
                with open(path,'r') as datf:
                    for line in datf.readlines():
                        lab, err = line.split(',')
                        labels.append(lab)
                        errors.append(err)
                x = range(len(errors))
                fig = plt.figure(figsize=(9,7))
                plt.ylabel('M.S.E of %s'%quantity)
                plt.xlabel('Number of Fourier Terms')
                plt.plot(labels,errors,linestyle='-',marker='o',color='b')
                plt.yscale(scale)
                #plt.xticks(x,labels,rotation='vertical')
                plt.tight_layout()
                plt.title(os.path.basename(base))
                if self.gconf.getboolean('General','save_plots'):
                    if '_excluded' in path:
                        excluded='_excluded'
                    else:
                        excluded=''
                    name = '%s_%sconvergence_%s%s.pdf'%(os.path.basename(base),err_type,quantity,excluded)
                    path = os.path.join(base,name)
                    fig.savefig(path)
                if self.gconf.getboolean('General','show_plots'):
                    plt.show() 
                plt.close(fig)

def main():
    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('-nc','--no_crunch',action="store_true",default=False,help="""Do not perform crunching
            operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-ngc','--no_gcrunch',action="store_true",default=False,help="""Do not
            perform global crunching operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-np','--no_plot',action="store_true",default=False,help="""Do not perform plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('-ngp','--no_gplot',action="store_true",default=False,help="""Do not perform global plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('--log_level',type=str,default='info',choices=['debug','info','warning','error','critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()
    
    # Configure logger
    logger = configure_logger(args.log_level,'postprocess',
                              os.path.join(conf.get('General','basedir'),'logs'),
                              'postprocess.log')

    # Now do all the work
    if not args.no_crunch:
        crunchr = Cruncher(conf)
        crunchr.crunch()
    if not args.no_gcrunch:
        gcrunchr = Global_Cruncher(conf)
        gcrunchr.crunch()
    if not args.no_plot:
        pltr = Plotter(conf) 
        pltr.plot()
    if not args.no_gplot:
        gpltr = Global_Plotter(conf)
        gpltr.plot()

if __name__ == '__main__':
    main()
