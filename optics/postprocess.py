import numpy as np
from scipy import interpolate
import scipy.constants as c
import scipy.integrate as intg
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
import multiprocessing as mp
import multiprocessing.dummy as mpd
#import pickle

def counted(fn):
    def wrapper(self):
        wrapper.called+= 1
        return fn(self)
    wrapper.called= 0
    wrapper.__name__= fn.__name__
    return wrapper

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


class Simulation(object):
    """An object that represents a simulation. It contains the data for the sim, the data file
    headers, and its configuration object as attributes. This object contains all the information
    about a simulation, but DOES NOT perform any actions on that data. That job is relegated to the
    various processor objects"""

    def __init__(self,conf):
        self.conf = conf
        self.log = logging.getLogger('postprocess')
        self.e_data = None
        self.h_data = None
        self.pos_inds = None
        self.failed = False
        self.e_lookup = OrderedDict([('x',0),('y',1),('z',2),('Ex_real',3),('Ey_real',4),('Ez_real',5),
                         ('Ex_imag',6),('Ey_imag',7),('Ez_imag',8)])
        self.h_lookup = OrderedDict([('x',0),('y',1),('z',2),('Hx_real',3),('Hy_real',4),('Hz_real',5),
                         ('Hx_imag',6),('Hy_imag',7),('Hz_imag',8)])
        self.avgs = {}

    def load_txt(self,path):
        try: 
            with open(path,'r') as dfile:
                hline = dfile.readline()
                if hline[0] == '#':
                    # We have headers in the file
                    # pandas read_csv is WAY faster than loadtxt. We've already read the header line
                    # so we pass header=None
                    d = pandas.read_csv(dfile,delim_whitespace=True,header=None,skip_blank_lines=True)
                    data = d.as_matrix()
                    headers = hline.strip('#\n').split(',')
                    lookup = {headers[ind]:ind for ind in range(len(headers))}
                    self.log.debug('Here is the E field header lookup: %s',str(lookup))
                else:
                    # Make sure we seek to beginning of file so we don't lose the first row
                    dfile.seek(0)
                    d = pandas.read_csv(dfile,delim_whitespace=True,header=None,skip_blank_lines=True)
                    data = d.as_matrix()
                    self.log.debug('File is missing headers')
                    lookup = None
        except FileNotFoundError:
            self.log.error('Following file missing: %s',path)
            self.failed = True
            data, lookup = None, None
        return data, lookup

    def load_npz(self,path):
        try: 
            # Get the headers and the data
            with np.load(path) as loaded:
                data = loaded['data']
                lookup = loaded['headers'][0]
                self.log.debug(str(type(lookup)))
                self.log.debug(str(lookup))
                self.log.debug('Here is the E field header lookup: %s',str(lookup))
        except IOError:
            self.log.error('Following file missing or unloadable: %s',path)
            self.failed = True
            data, lookup = None, None
        return data,lookup

    def get_raw_data(self):
        """Collect raw, unprocessed data spit out by S4 simulations"""
        self.log.info('Collecting raw data for sim %s',self.conf.get('General','sim_dir'))
        sim_path = self.conf.get('General','sim_dir')
        base_name = self.conf.get('General','base_name')
        ignore = self.conf.getboolean('General','ignore_h')
        ftype = self.conf.get('General','save_as')
        if ftype == 'text':
            # Load E field data
            e_path = os.path.join(sim_path,base_name+'.E')
            e_data, e_lookup = self.load_txt(e_path)
            self.log.debug('E shape after getting: %s',str(e_data.shape))        
            pos_inds = np.zeros((e_data.shape[0],3))
            pos_inds[:,:] = e_data[:,0:3] 
            # Load H field data
            if not ignore: 
                h_path = os.path.join(sim_path,base_name+'.H')
                h_data, h_lookup = self.load_txt(h_path)
            else:
                h_data = None
            self.e_data,self.h_data,self.pos_inds = e_data,h_data,pos_inds
            self.log.info('Collection complete!')
        elif ftype == 'npz':
            # Load E field data
            e_path = os.path.join(sim_path,base_name+'.E.raw.npz')
            e_data, e_lookup = self.load_npz(e_path)
            self.log.debug('E shape after getting: %s',str(e_data.shape))        
            pos_inds = np.zeros((e_data.shape[0],3))
            pos_inds[:,:] = e_data[:,0:3] 
            # Load H field data
            if not ignore: 
                h_path = os.path.join(sim_path,base_name+'.H.raw.npz')
                h_data, h_lookup = self.load_npz(h_path)
            else:
                h_data = None
            self.e_data,self.h_data,self.pos_inds = e_data,h_data,pos_inds
            self.log.info('Collection complete!')
        else:
            raise ValueError('Incorrect file type specified in [General] section of config file')

        return e_data,self.e_lookup,h_data,self.h_lookup,pos_inds

    def get_data(self):
        """Returns the already crunched E and H data for this particular sim"""
        self.log.info('Collecting data for sim %s',self.conf.get('General','sim_dir'))
        sim_path = self.conf.get('General','sim_dir')
        base_name = self.conf.get('General','base_name')
        ftype = self.conf.get('General','save_as')
        ignore = self.conf.getboolean('General','ignore_h')
        # If data was saved into text files
        if ftype == 'text':
            # Load E field data
            e_path = os.path.join(sim_path,base_name+'.E.crnch')
            e_data, e_lookup = self.load_txt(e_path)
            # Load H field data
            if not ignore:
                h_path = os.path.join(sim_path,base_name+'.H.crnch')
                h_data, h_lookup = self.load_txt(h_path)
            else:
                h_data,h_lookup = None,None
        # If data was saved in in npz format
        elif ftype == 'npz':
            # Get the paths
            e_path = os.path.join(sim_path,base_name+'.E.npz')
            e_data, e_lookup = self.load_npz(e_path)
            if not ignore:
                h_path = os.path.join(sim_path,base_name+'.H.npz')
                h_data, h_lookup = self.load_npz(h_path)
            else:
                h_data,h_lookup = None,None
        else:
            raise ValueError('Incorrect file type specified in [General] section of config file')
        pos_inds = np.zeros((e_data.shape[0],3))
        pos_inds[:,:] = e_data[:,0:3] 
        self.e_data,self.e_lookup,self.h_data,self.h_lookup,self.pos_inds = e_data,e_lookup,h_data,h_lookup,pos_inds
        self.log.info('Collection complete!')
        return e_data,e_lookup,h_data,h_lookup,pos_inds

    def get_avgs(self):
        """Load all averages"""
        # Get the current path
        base = self.conf.get('General','sim_dir')
        ftype = self.conf.get('General','save_as') 
        if ftype == 'text':
            globstr = os.path.join(base,'*avg.crnch')
            for f in glob.glob(globstr):
                fname = os.path.basename(f)
                key = fname[0:fname.index('.')]
                self.avgs[key] = np.loadtxt(f)
            self.log.info('Available averages: %s'%str(self.avgs.keys()))
        elif ftype == 'npz':
            path = os.path.join(base,'all.avg.npz')
            with np.load(path) as avgs_file:
                for arr in avgs_file.files:
                    self.avgs[arr] = avgs_file[arr]
            self.log.info('Available averages: %s'%str(self.avgs.keys()))
        
    def write_data(self):
        """Writes the data"""
        # Get the current path
        base = self.conf.get('General','sim_dir')
        ignore = self.conf.getboolean('General','ignore_h')
        self.log.info('Writing data for %s'%base)
        fname = self.conf.get('General','base_name')
        epath = os.path.join(base,fname+'.E')
        hpath = os.path.join(base,fname+'.H')
        # Save matrices in specified file tipe 
        self.log.debug('Here are the E matrix headers: %s',str(self.e_lookup))
        #self.log.debug('Here is the E matrix: \n %s',str(self.e_data))
        self.log.debug('Here are the H matrix headers: %s',str(self.h_lookup))
        #self.log.debug('Here is the H matrix: \n %s',str(self.h_data))
        ftype = self.conf.get('General','save_as') 
        if ftype == 'text':
            epath = epath+'.crnch'
            np.savetxt(epath,self.e_data,header=','.join(self.e_lookup.keys()))
            if not ignore:
                hpath= hpath+'.crnch'
                np.savetxt(hpath,self.h_data,header=','.join(self.h_lookup.keys()))
            # Save any local averages we have computed
            for avg, mat in self.avgs.items():
                dpath = os.path.join(base,avg+'.avg.crnch')
                np.savetxt(dpath,mat)
        elif ftype == 'npz':
            # Save the headers and the data
            np.savez(epath,headers = np.array([self.e_lookup]), data = self.e_data)
            if not ignore:
                np.savez(hpath,headers = np.array([self.h_lookup]), data = self.h_data)
            # Save any local averages we have computed
            dpath = os.path.join(base,'all.avg')
            np.savez(dpath,**self.avgs)
        else:
            raise ValueError('Specified saving in an unsupported file format')

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
    
    def clear_data(self):
        """Clears all the data attributes to free up memory"""
        self.e_data = None
        self.h_data = None
        self.pos_inds = None
        self.avgs = {}

class Processor(object):
    """Base data processor class that has some methods every other processor needs"""
    def __init__(self,global_conf,sims=[],sim_groups=[],failed_sims=[]):
        self.log = logging.getLogger('postprocess')
        self.log.debug("Processor base init")
        self.gconf = global_conf
        self.sims = sims
        self.sim_groups = sim_groups
        # A place to store any failed sims (i.e sims that are missing their data file)
        self.failed_sims = failed_sims
    
    def dir_keys(self,path):
        """A function to take a path, and return a list of all the numbers in the path. This is
        mainly used for sorting by the parameters they contain"""

        regex = '[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?' # matching any floating point
        m = re.findall(regex, path)
        if m: 
            val = m
            val = list(map(float,val))
        else: 
            self.log.info('Your path does not contain any numbers')
            val = os.path.basename(path)
        return val
    
    def sim_key(self,sim):
        """A wrapper function around dir_keys that takes a sim object as an arg, extracts it path,
        then passes that to dir_keys"""
        path = sim.conf.get('General','sim_dir')
        key = self.dir_keys(path)
        return key

    def collect_sims(self):
        """Collect all the simulations beneath the base of the directory tree"""
        # Clear out the lists
        self.sims = []
        self.sim_groups = []
        self.failed_sims = []
        ftype = self.gconf.get('General','save_as')
        if ftype == 'text': 
            datfile = self.gconf.get('General','base_name')+'.E'
        else:
            datfile = self.gconf.get('General','base_name')+'.E.raw.npz' 

        for root,dirs,files in os.walk(self.gconf.get('General','basedir')):
            if 'sim_conf.ini' in files and datfile in files:
                self.log.info('Gather sim at %s',root)
                sim_obj = Simulation(parse_file(os.path.join(root,'sim_conf.ini')))
                self.sims.append(sim_obj)
            elif 'sim_conf.ini' in files:
                sim_obj = Simulation(parse_file(os.path.join(root,'sim_conf.ini')))
                self.log.error('The following sim is missing its data file: %s',
                                sim_obj.conf.get('General','sim_dir'))
                self.failed_sims.append(sim_obj)
            if 'sorted_sweep_conf.ini' in files:
                self.log.info('Gather sim group at %s',root)
                if 'logs' in dirs:
                    dirs.remove('logs')
                conf_paths = [os.path.join(root,simdir,'sim_conf.ini') for simdir in dirs]
                self.log.debug('Sim group confs: %s',str(conf_paths))
                confs = list(map(parse_file,conf_paths))
                group = [Simulation(conf) for conf in confs]
                self.sim_groups.append(group)
        # This makes it so we can compute convergence if we have every other param fixed but we
        # swept through # of basis terms
        if not self.sim_groups:
            self.sim_groups = [self.sims]
        self.sort_sims()
        self.log.debug('Sims: %s'%str(self.sims))
        return self.sims,self.sim_groups,self.failed_sims

    def sort_sims(self):
        """Sorts simulations by their parameters the way a human would. Called human sorting or
        natural sorting. Thanks stackoverflow"""
        
        self.sims.sort(key=self.sim_key)
        for group in self.sim_groups:
            paths = [sim.conf.get('General','sim_dir') for sim in group]
            self.log.debug('Group paths before sorting: %s',str(paths))
            group.sort(key=self.sim_key)
            paths = [sim.conf.get('General','sim_dir') for sim in group]
            self.log.debug('Group paths after sorting: %s',str(paths))

    def get_param_vals(self,par):
        """Return all possible values of the provided parameter for this sweep"""
        vals = []
        for sim in self.sims:
            val = sim.conf.get('Parameters',par)
            if val not in vals:
                vals.append(val)
        return vals

    def filter_by_param(self,pars):
        """Accepts a dict where the keys are parameter names and the values are a list of possible 
        values for that parameter. Any simulation whose parameter does not match any of the provided 
        values is removed from the sims list attribute"""
        
        assert(type(pars) == dict)
        filt = lambda sim,par,vals: sim.conf.get('Parameters',par) in vals
        for par,vals in pars.items():
            self.sims = [sim for sim in self.sims if filt(sim,par,vals)]
            groups = []
            for group in self.sim_groups:
                filt_group = [sim for sim in group if filt(sim,par,vals)]
                groups.append(filt_group)
            self.sim_groups = groups
        assert(len(self.sims) >= 1)
        return self.sims,self.sim_groups

    def get_plane(self,arr,xsamp,ysamp,zsamp,plane,pval):
        """Given a 1D array containing values for a 3D scalar field, reshapes the array into 3D and
        returns a 2D array containing the data on a given plane, for a specified index value (pval) of that
        plane. So, specifying plane=x and pval=30 would return data on the 30th y,z plane (a plane
        at the given x index). The number of samples (i.e data points) in each coordinate direction
        need not be equal"""
        scalar = arr.reshape(zsamp+1,xsamp,ysamp)
        if plane == 'x':
            # z along rows, y along columns
            return scalar[:,pval,:]
        elif plane == 'y':
            # x along columns, z along rows
            return scalar[:,:,pval]
        elif plane == 'z':
            # x along rows, y along columns
            return scalar[pval,:,:]

    def process(self,sim):
        """Retrieves data for a particular simulation, then processes that data"""
        raise NotImplementedError 
    
    def process_all(self):
        """Processes all the sims collected and stored in self.sims and self.sim_groups"""
        raise NotImplementedError 
        
class Cruncher(Processor):
    """Crunches all the raw data. Calculates quantities specified in the global config file and
    either appends them to the existing data files or creates new ones as needed"""

    def __init__(self,global_conf,sims=[],sim_groups=[],failed_sims=[]):
        super().__init__(global_conf,sims,sim_groups,failed_sims)
        self.log.debug("This is THE CRUNCHER!!!!!")
    
    def calculate(self,quantity,sim,args):
        try:
            getattr(self,quantity)(sim,*args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity,exc_info=True,stack_info=True)
            raise

    def process(self,sim):
        sim_path = os.path.basename(sim.conf.get('General','sim_dir'))
        self.log.info('Crunching data for sim %s',sim_path)
        sim.get_raw_data()
        self.log.debug('SHAPE BEFORE CALCULATING: %s'%str(sim.e_data.shape))
        if sim.failed:
            self.log.error('Following simulation missing data: %s',sim_path)
            self.failed_sims.append(sim)
        else:
            # For each quantity 
            for quant,args in self.gconf.items('Cruncher'):
                self.log.info('Computing %s with args %s',str(quant),str(args))
                for argset in args.split(';'):
                    if argset:
                        self.calculate(quant,sim,argset.split(','))
                    else:
                        self.calculate(quant,sim,[])
                    self.log.debug('SHAPE AFTER CALCULATING: %s'%str(sim.e_data.shape))
                self.log.debug('E lookup: %s',str(sim.e_lookup))
                self.log.debug('H lookup: %s',str(sim.h_lookup))
                
            sim.write_data()
            sim.clear_data()
    
    def process_all(self):
        self.log.info('Beginning data crunch ...')
        if not self.gconf.getboolean('General','post_parallel'):
            for sim in self.sims:
                self.process(sim)
        else:
            num_procs = mp.cpu_count() - self.gconf.getint('General','reserved_cores')
            self.log.info('Crunching sims in parallel using %s cores ...',str(num_procs))
            pool = mpd.Pool(processes=num_procs)
            pool.map(self.process,self.sims)

    def normE(self,sim):
        """Calculate and returns the norm of E"""
        
        # Get the magnitude of E and add it to our data
        E_mag = np.zeros(sim.e_data.shape[0])
        for i in range(3,9):
            E_mag += sim.e_data[:,i]*sim.e_data[:,i]
        E_mag = np.sqrt(E_mag)
        
        # This approach is 4 times faster than np.column_stack()
        self.log.debug('E mat shape: %s',str(sim.e_data.shape))
        dat = np.zeros((sim.e_data.shape[0],sim.e_data.shape[1]+1))
        self.log.debug('dat mat shape: %s',str(dat.shape))
        dat[:,:-1] = sim.e_data
        dat[:,-1] = E_mag
        sim.e_data = dat 
        # Now append this quantity and its column the the header dict
        sim.e_lookup['normE'] = dat.shape[1]-1 
        return E_mag

    def normEsquared(self,sim):
        """Calculates and returns normE squared"""
        
        # Get the magnitude of E and add it to our data
        E_magsq = np.zeros(sim.e_data.shape[0])
        for i in range(3,9):
            E_magsq += sim.e_data[:,i]*sim.e_data[:,i]
        
        # This approach is 4 times faster than np.column_stack()
        dat = np.zeros((sim.e_data.shape[0],sim.e_data.shape[1]+1))
        dat[:,:-1] = sim.e_data
        dat[:,-1] = E_magsq
        sim.e_data = dat 
        # Now append this quantity and its column the the header dict
        sim.e_lookup['normEsquared'] = dat.shape[1]-1 
        return E_magsq

    def normH(self,sim):
        """Calculate and returns the norm of H"""
        
        # Get the magnitude of H and add it to our data. This loops through each components real and
        # imaginary parts and squared it (which is what would happen if you took the complex number
        # for each component and multiplied it by its conjugate). 
        H_mag = np.zeros(sim.h_data.shape[0])
        for i in range(3,9):
            H_mag += sim.h_data[:,i]*sim.h_data[:,i]
        H_mag = np.sqrt(H_mag)
        
        # This approach is 4 times faster than np.column_stack()
        dat = np.zeros((sim.h_data.shape[0],sim.h_data.shape[1]+1))
        dat[:,:-1] = sim.h_data
        dat[:,-1] = H_mag
        sim.h_data = dat 
        # Now append this quantity and its column the the header dict
        sim.h_lookup['normH'] = dat.shape[1]-1 
        return H_mag
    
    def normHsquared(self,sim):
        """Calculates and returns the norm of H squared"""
        
        # Get the magnitude of H and add it to our data
        H_magsq = np.zeros(sim.h_data.shape[0])
        for i in range(3,9):
            H_magsq += sim.h_data[:,i]*sim.h_data[:,i]
        # This approach is 4 times faster than np.column_stack()
        dat = np.zeros((sim.h_data.shape[0],sim.h_data.shape[1]+1))
        dat[:,:-1] = sim.h_data
        dat[:,-1] = H_magsq
        sim.h_data = dat 
        # Now append this quantity and its column the the header dict
        sim.h_lookup['normHsquared'] = dat.shape[1]-1 
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

    def genRate(self,sim):
        # We need to compute normEsquared before we can compute the generation rate
        try: 
            normEsq = sim.get_scalar_quantity('normEsquared') 
        except KeyError:
            normEsq = self.normEsquared(sim)
        # Prefactor for generation rate. Not we gotta convert from m^3 to cm^3, hence 1e6 factor
        fact = c.epsilon_0/(c.hbar*1e6) 
        # Get the indices of refraction at this frequency
        freq = sim.conf.get('Parameters','frequency')
        nk = {mat[0]:(self.get_nk(mat[1],freq)) for mat in self.gconf.items('Materials')}
        self.log.debug(nk) 
        # Get spatial discretization
        z_samples = sim.conf.getint('General','z_samples')
        x_samples = sim.conf.getint('General','x_samples')
        y_samples = sim.conf.getint('General','y_samples')
        # Reshape into an actual 3D matrix. Rows correspond to different y fixed x, columns to fixed
        # y variable x, and each layer in depth is a new z value
        normEsq = np.reshape(normEsq,(z_samples+1,x_samples,y_samples))
        gvec = np.zeros_like(normEsq)
        height = sim.conf.getfloat('Parameters','total_height')
        dz = height/z_samples
        period = sim.conf.getfloat('Parameters','array_period')
        dx = period/x_samples
        dy = period/y_samples
        # Get boundaries between layers
        air_ito = sim.conf.getfloat('Parameters','air_t')
        ito_nw = sim.conf.getfloat('Parameters','ito_t')+air_ito
        nw_sio2 = sim.conf.getfloat('Parameters','alinp_height')+ito_nw
        sio2_sub = sim.conf.getfloat('Parameters','sio2_height')+nw_sio2
        air_line = sim.conf.getfloat('Parameters','substrate_t')+sio2_sub
        # ITO Generation
        start = int(air_ito/dz)+1
        end = int(ito_nw/dz)+1
        self.log.debug('ITO START = %i'%start)
        self.log.debug('ITO LAYER SHAPE = %s'%str(normEsq[start:end,:,:].shape))
        gvec[start:end,:,:] = fact*nk['ITO'][0]*nk['ITO'][1]*normEsq[start:end,:,:]
        # NW/Shell/Cyclotene Generation
        center = period/2.0
        core_radius = sim.conf.getfloat('Parameters','nw_radius') 
        core_radius_sq = core_radius*core_radius
        total_radius = core_radius + sim.conf.getfloat('Parameters','shell_t')
        total_radius_sq = total_radius*total_radius
        # Build the matrices containing the NK profile in the x-y plane
        al_nk_mat = np.zeros_like(gvec[0,:,:])
        si_nk_mat = np.zeros_like(gvec[0,:,:])
        for xi in range(x_samples):
            for yi in range(y_samples):
                dist = ((xi*dx)-center)**2 + ((yi*dy)-center)**2 
                if  dist <= core_radius_sq:
                    al_nk_mat[yi,xi] = fact*nk['GaAs'][0]*nk['GaAs'][1]
                    si_nk_mat[yi,xi] = fact*nk['GaAs'][0]*nk['GaAs'][1]
                elif core_radius_sq <= dist <= total_radius_sq:
                    al_nk_mat[yi,xi] = fact*nk['AlInP'][0]*nk['AlInP'][1]
                    si_nk_mat[yi,xi] = fact*nk['SiO2'][0]*nk['SiO2'][1]
                else:
                    al_nk_mat[yi,xi] = fact*nk['Cyclotene'][0]*nk['Cyclotene'][1]
                    si_nk_mat[yi,xi] = fact*nk['Cyclotene'][0]*nk['Cyclotene'][1]
        # Generation for AlInP shell region
        start = end
        end = int(nw_sio2/dz)+1
        gvec[start:end+1,:,:] = (al_nk_mat*normEsq[start:end+1,:,:])
        # Generation for SiO2 shell region
        start = end 
        end = int(sio2_sub/dz)+1
        gvec[start:end+1,:,:] = (si_nk_mat*normEsq[start:end+1,:,:])
        # The rest is just the substrate
        start = end
        gvec[end:,:,:] = fact*nk['GaAs'][0]*nk['GaAs'][1]*normEsq[end:,:,:]
        # Reshape back to 1D array
        gvec = gvec.reshape((x_samples*y_samples*(z_samples+1)))
        # This approach is 4 times faster than np.column_stack()
        assert(sim.e_data.shape[0] == len(gvec))
        dat = np.zeros((sim.e_data.shape[0],sim.e_data.shape[1]+1))
        dat[:,:-1] = sim.e_data
        dat[:,-1] = gvec
        sim.e_data = dat 
        # Now append this quantity and its column the the header dict
        sim.e_lookup['genRate'] = dat.shape[1]-1 
        return gvec
    
    def angularAvg(self,sim,quantity):
        """Perform an angular average of some quantity for either the E or H field"""
        quant = sim.get_scalar_quantity(quantity)
        # Get spatial discretization
        z_samples = sim.conf.getint('General','z_samples')
        x_samples = sim.conf.getint('General','x_samples')
        y_samples = sim.conf.getint('General','y_samples')
        rsamp = sim.conf.getint('General','r_samples') 
        thsamp = sim.conf.getint('General','theta_samples')
        # Reshape into an actual 3D matrix. Rows correspond to different y fixed x, columns to fixed
        # y variable x, and each layer in depth is a new z value
        values = np.reshape(quant,(z_samples+1,x_samples,y_samples))
        height = sim.conf.getfloat('Parameters','total_height')
        dz = height/z_samples
        period = sim.conf.getfloat('Parameters','array_period')
        dx = period/x_samples
        dy = period/y_samples
        x = np.linspace(0,period,x_samples)
        y = np.linspace(0,period,y_samples)
        # Maximum r value such that circle and square unit cell have equal area
        rmax = period/np.sqrt(np.pi)
        # Diff between rmax and unit cell boundary at point of maximum difference
        delta = rmax - period/2.0 
        # Extra indices we need to expand layers by
        x_inds = int(np.ceil(delta/dx))
        y_inds = int(np.ceil(delta/dy))
        # Use periodic BCs to extend the data in the x-y plane
        ext_vals = np.zeros((values.shape[0],values.shape[1]+2*x_inds,values.shape[2]+2*y_inds))
        # Left-Right extensions. This indexing madness extracts the slice we want, flips it along the correct dimension
        # then sticks in the correct spot in the extended array
        ext_vals[:,x_inds:-x_inds,0:y_inds] = values[:,:,0:y_inds][:,:,::-1]
        ext_vals[:,x_inds:-x_inds,-y_inds:] = values[:,:,-y_inds:][:,:,::-1]
        # Top-Bottom extensions
        ext_vals[:,0:x_inds,y_inds:-y_inds] = values[:,0:x_inds,:][:,::-1,:]
        ext_vals[:,-x_inds:,y_inds:-y_inds] = values[:,-x_inds:,:][:,::-1,:]
        # Corners, slightly trickier
        # Top left
        ext_vals[:,0:x_inds,0:y_inds] = ext_vals[:,x_inds:2*x_inds,0:y_inds][:,::-1,:]
        # Bottom left
        ext_vals[:,-x_inds:,0:y_inds] = ext_vals[:,-2*x_inds:-x_inds,0:y_inds][:,::-1,:]
        # Top right
        ext_vals[:,0:x_inds,-y_inds:] = ext_vals[:,0:x_inds,-2*y_inds:-y_inds][:,:,::-1]
        # Bottom right
        ext_vals[:,-x_inds:,-y_inds:] = ext_vals[:,-x_inds:,-2*y_inds:-y_inds][:,:,::-1]
        # Now the center
        ext_vals[:,x_inds:-x_inds,y_inds:-y_inds] = values[:,:,:]
        # Extend the points arrays to include these new regions
        x = np.concatenate((np.array([dx*i for i in range(-x_inds,0)]),x,np.array([x[-1]+dx*i for i in range(1,x_inds+1)])))
        y = np.concatenate((np.array([dy*i for i in range(-y_inds,0)]),y,np.array([y[-1]+dy*i for i in range(1,y_inds+1)])))
        # The points on which we have data
        points = (x,y)
        # The points corresponding to "rings" in cylindrical coordinates. Note we construct these
        # rings around the origin so we have to shift them to actually correspond to the center of
        # the nanowire
        rvec = np.linspace(0,rmax,rsamp)
        thvec = np.linspace(0,2*np.pi,thsamp)
        cyl_coords = np.zeros((len(rvec)*len(thvec),2))
        start = 0
        for r in rvec:
            xring = r*np.cos(thvec)
            yring = r*np.sin(thvec)
            cyl_coords[start:start+len(thvec),0] = xring
            cyl_coords[start:start+len(thvec),1] = yring
            start += len(thvec)
        cyl_coords += period/2.0        
        # For every z layer in the 3D matrix of our quantity
        avgs = np.zeros((ext_vals.shape[0],len(rvec)))
        i = 0
        for layer in ext_vals:
            interp_vals = interpolate.interpn(points,layer,cyl_coords,method='linear')
            rings = interp_vals.reshape((len(rvec),len(thvec)))
            avg = np.average(rings,axis=1)
            avgs[i,:] = avg
            i += 1
        avgs = avgs[:,::-1]
        # Save to avgs dict for this sim
        key = quantity+'_angularAvg'
        sim.avgs[key] = avgs
        return avgs

    def transmissionData(self,sim):
        """Computes reflection, transmission, and absorbance"""
        base = sim.conf.get('General','sim_dir')
        path = os.path.join(base,'fluxes.dat')
        data = {}
        with open(path,'r') as f:
            d = f.readlines()
            headers = d.pop(0)
            for line in d:
                els = line.split(',')
                key = els.pop(0)
                data[key] = list(map(float,els))
        # NOTE: Take only the real part of the power as per https://en.wikipedia.org/wiki/Poynting_vector#Time-averaged_Poynting_vector
        p_inc = data['air'][0]
        p_ref = np.abs(data['air'][1]) 
        p_trans = data['substrate'][0] 
        #p_inc = np.sqrt(data['air'][0]**2+data['air'][2]**2)
        #p_ref = np.sqrt(data['air'][1]**2+data['air'][3]**2) 
        #p_trans = np.sqrt(data['substrate_bottom'][0]**2+data['substrate_bottom'][2]**2)
        reflectance = p_ref / p_inc
        transmission = p_trans / p_inc
        absorbance = 1 - reflectance - transmission
        #absorbance = 1 - reflectance
        tot = reflectance+transmission+absorbance
        delta = np.abs(tot-1)
        #self.log.info('Total = %f'%tot)
        assert(delta < .0001)
        self.log.debug('Reflectance %f'%reflectance)       
        self.log.debug('Transmission %f'%transmission)       
        self.log.debug('Absorbance %f'%absorbance)       
        #assert(reflectance >= 0 and transmission >= 0 and absorbance >= 0)
        outpath = os.path.join(base,'ref_trans_abs.dat')
        with open(outpath,'w') as out:
            out.write('# Reflectance,Transmission,Absorbance\n')
            out.write('%f,%f,%f'%(reflectance,transmission,absorbance))
        return reflectance,transmission,absorbance

    def integrated_absorbtion(self,sim):
        """Computes the absorption of a layer by using the volume integral of the product of the
        imaginary part of the relative permittivity and the norm squared of the E field""" 
        base = sim.conf.get('General','sim_dir')
        path = os.path.join(base,'integrated_absorption.dat')
        inpath = os.path.join(base,'energy_densities.dat')
        freq = sim.conf.getfloat('Parameters','frequency')
        # TODO: Assuming incident amplitude and therefore incident power is just 1 for now
        fact = -.5*freq*c.epsilon_0
        with open(inpath,'r') as inf:
            lines = inf.readlines()
            # Remove header line
            lines.pop(0)
            # Dict where key is layer name and value is list of length 2 containing real and
            # imaginary parts of energy density integral
            data = {line.strip().split(',')[0]:line.strip().split(',')[1:] for line in lines}
        self.log.info('Energy densities: %s'%str(data))
        with open(path,'w') as outf:
            outf.write('# Layer, Absorption\n')
            for layer,vals in data.items():
                absorb = fact*float(vals[1])
                outf.write('%s,%s\n'%(layer,absorb))
            
class Global_Cruncher(Cruncher):
    """Computes global quantities for an entire run, instead of local quantities for an individual
    simulation"""
    def __init__(self,global_conf,sims=[],sim_groups=[],failed_sims=[]):
        super().__init__(global_conf,sims,sim_groups,failed_sims)
        self.log.debug('This is the global cruncher') 

    def calculate(self,quantity,args):
        try:
            getattr(self,quantity)(*args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity,exc_info=True,stack_info=True)
            raise

    def process_all(self):
        # For each quantity 
        self.log.info('Beginning global cruncher processing ...')
        for quant,args in self.gconf.items('Global_Cruncher'):
            self.log.info('Computing %s with args %s',str(quant),str(args))
            for argset in args.split(';'):
                self.log.info('Passing following arg set to function %s: %s',str(quant),str(argset))
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

    def get_slice(self,sim):
        """Returns indices for data that strip out air and substrate regions"""
        x_samples = sim.conf.getint('General','x_samples')
        y_samples = sim.conf.getint('General','y_samples')
        z_samples = sim.conf.getint('General','z_samples')
        h = sum((sim.conf.getfloat('Parameters','nw_height'),sim.conf.getfloat('Parameters','substrate_t'),
                sim.conf.getfloat('Parameters','air_t'),sim.conf.getfloat('Parameters','ito_t')))
        arr = np.linspace(0,h,z_samples)
        dz = arr[1] - arr[0]
        start_plane = int(round(sim.conf.getfloat('Parameters','air_t')/dz))
        start = start_plane*(x_samples*y_samples)
        end_plane = int(round(sum((sim.conf.getfloat('Parameters','nw_height'),sim.conf.getfloat('Parameters','air_t'),
                sim.conf.getfloat('Parameters','ito_t')))/dz))
        end = end_plane*(x_samples*y_samples)
        return start,end

    def get_comp_vec(self,sim,field,start,end):
        """Returns the comparison vector"""
        # Compare all other sims to our best estimate, which is sim with highest number of
        # basis terms (last in list cuz sorting)

        # Get the proper file extension depending on the field.
        if field == 'E':
            ext = '.E'
            # Get the comparison vector
            vec1 = sim.e_data[start:end,3:9]
            normvec = sim.get_scalar_quantity('normE')
            normvec = normvec[start:end]**2
        elif field == 'H':
            ext = '.H'
            vec1 = sim.h_data[start:end,3:9]
            normvec = sim.get_scalar_quantity('normH')
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
        for group in self.sim_groups:
            if exclude:
                start,end = self.get_slice(group[0])    
                excluded = '_excluded'
            else:
                start = 0
                end = None
                excluded = ''
            base = group[0].conf.get('General','basedir')
            errpath = os.path.join(base,'localerror_%s%s.dat'%(field,excluded))
            with open(errpath,'w') as errfile:
                self.log.info('Computing local error for sweep %s',base)
                # Set the reference sim 
                ref_sim = group[-1]
                ref_sim.get_data()
                # Get the comparison vector
                vec1,normvec,ext = self.get_comp_vec(ref_sim,field,start,end)    
                # For all other sims in the groups, compare to best estimate and write to error file 
                for i in range(0,len(group)-1):
                    sim2 = group[i]
                    sim2.get_data()
                    if field == 'E': 
                        vec2 = sim2.e_data[start:end,3:9]
                    elif field == 'H':
                        vec2 = sim2.h_data[start:end,3:9]
                    self.log.info("Computing local error between numbasis %i and numbasis %i",
                                  ref_sim.conf.getint('Parameters','numbasis'),
                                  sim2.conf.getint('Parameters','numbasis'))
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
                    errfile.write('%i,%f\n'%(sim2.conf.getint('Parameters','numbasis'),avg_diffvec_mag))
                    sim2.clear_data()
                ref_sim.clear_data()

    def global_error(self,field,exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell"""

        self.log.info('Running the global error computation for quantity %s',field) 
        # If we need to exclude calculate the indices
        for group in self.sim_groups:
            if exclude:
                start,end = self.get_slice(group[0])    
                excluded = '_excluded'
            else:
                start = 0
                end = None
                excluded = ''
            base = group[0].conf.get('General','basedir')
            errpath = os.path.join(base,'globalerror_%s%s.dat'%(field,excluded))
            with open(errpath,'w') as errfile:
                self.log.info('Computing global error for sweep %s',base)
                # Set reference sim
                ref_sim = group[-1]
                ref_sim.get_data()
                # Get the comparison vector
                vec1,normvec,ext = self.get_comp_vec(ref_sim,field,start,end)    
                # For all other sims in the groups, compare to best estimate and write to error file 
                for i in range(0,len(group)-1):
                    sim2 = group[i]
                    sim2.get_data()
                    if field == 'E': 
                        vec2 = sim2.e_data[start:end,3:9]
                    elif field == 'H':
                        vec2 = sim2.h_data[start:end,3:9]
                    self.log.info("Computing global error between numbasis %i and numbasis %i",
                                  ref_sim.conf.getint('Parameters','numbasis'),
                                  sim2.conf.getint('Parameters','numbasis'))
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
                    errfile.write('%i,%f\n'%(sim2.conf.getint('Parameters','numbasis'),error))
                    sim2.clear_data()
                ref_sim.clear_data()

    def adjacent_error(self,field,exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell.
        This computes error between adjacent sims in a sweep through basis terms."""

        self.log.info('Running the global error computation for quantity %s',field) 
        # If we need to exclude calculate the indices
        for group in self.sim_groups:
            if exclude:
                start,end = self.get_slice(group[0])    
                excluded = '_excluded'
            else:
                start = 0
                end = None
                excluded = ''
            base = group[0].conf.get('General','basedir')
            errpath = os.path.join(base,'adjacenterror_%s%s.dat'%(field,excluded))
            with open(errpath,'w') as errfile:
                self.log.info('Computing adjacent error for sweep %s',base)
                # For all other sims in the groups, compare to best estimate and write to error file 
                for i in range(1,len(group)):
                    # Set reference sim
                    ref_sim = group[i]
                    ref_sim.get_data()
                    # Get the comparison vector
                    vec1,normvec,ext = self.get_comp_vec(ref_sim,field,start,end)    
                    sim2 = group[i-1]
                    sim2.get_data()
                    if field == 'E': 
                        vec2 = sim2.e_data[start:end,3:9]
                    elif field == 'H':
                        vec2 = sim2.h_data[start:end,3:9]
                    self.log.info("Computing adjacent error between numbasis %i and numbasis %i",
                                  ref_sim.conf.getint('Parameters','numbasis'),
                                  sim2.conf.getint('Parameters','numbasis'))
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
                    errfile.write('%i,%f\n'%(sim2.conf.getint('Parameters','numbasis'),error))
                    sim2.clear_data()
                    ref_sim.clear_data()

    def global_avg(self,quantity,avg_type):
        """Combine local average of a specific quantity for all leaves in each group"""
        for group in self.sim_groups:
            base = group[0].conf.get('General','basedir')
            self.log.info('Computing global averages for group at %s'%base)
            key = '%s_%s'%(quantity,avg_type)
            group[0].get_avgs()
            first = group[0].avgs[key]
            group[0].clear_data()
            group_avg = np.zeros((len(group),first.shape[0],first.shape[1]))
            group_avg[0,:,:] = first
            for i in range(1,len(group)):
                group[i].get_avgs()
                group_avg[i,:,:] = group[i].avgs[key]
            group_avg = np.average(group_avg,axis=0)
            path = os.path.join(base,'%s_%s_global.avg.crnch'%(quantity,avg_type))
            np.savetxt(path,group_avg)

    def Jsc(self):
        """Computes photocurrent density"""
        Jsc_list = []
        for group in self.sim_groups:
            base = group[0].conf.get('General','basedir')
            self.log.info('Computing photocurrent density for group at %s'%base)
            Jsc_vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Assuming the leaves contain frequency values, sum over all of them
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf.get('General','sim_dir'),'ref_trans_abs.dat')
                with open(dpath,'r') as f:
                    ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
                freq = sim.conf.getfloat('Parameters','frequency')
                wvlgth = c.c/freq
                wvlgth_nm = wvlgth*1e9
                freqs[i] = freq
                wvlgths[i] = wvlgth
                # Get solar power from chosen spectrum
                path = self.gconf.get('General','input_power_wv')
                wv_vec,p_vec = np.loadtxt(path,skiprows=2,usecols=(0,2),unpack=True,delimiter=',')
                # Get p at wvlength by interpolation 
                p_wv = interpolate.interp1d(wv_vec,p_vec,kind='linear',
                                           bounds_error=False,fill_value='extrapolate')
                sun_pow = p_wv(wvlgth_nm) 
                spectra[i] = sun_pow*wvlgth_nm
                Jsc_vals[i] = absorb*sun_pow*wvlgth_nm
            # Use Simpsons rule to perform the integration
            wvlgths = wvlgths[::-1]
            Jsc_vals = Jsc_vals[::-1]
            spectra = spectra[::-1]
            #plt.figure()
            #plt.plot(wvlgths,Jsc_vals)
            #plt.show()
            #Jsc = intg.simps(Jsc_vals,x=wvlgths,even='avg')
            Jsc = intg.trapz(Jsc_vals,x=wvlgths*1e9)
            power = intg.trapz(spectra,x=wvlgths*1e9)
            # factor of 1/10 to convert A*m^-2 to mA*cm^-2
            #wv_fact = c.e/(c.c*c.h*10)
            #wv_fact = .1
            #Jsc = (Jsc*wv_fact)/power
            Jsc = Jsc/power
            outf = os.path.join(base,'jsc.dat')
            with open(outf,'w') as out:
                out.write('%f\n'%Jsc)
            print('Jsc = %f'%Jsc)
            Jsc_list.append(Jsc)
        return Jsc_list

    def weighted_transmissionData(self):
        """Computes spectrally weighted absorption,transmission, and reflection""" 
        for group in self.sim_groups:
            base = group[0].conf.get('General','basedir')
            self.log.info('Computing spectrally weighted transmission data for group at %s'%base)
            abs_vals = np.zeros(len(group))
            ref_vals = np.zeros(len(group))
            trans_vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Get solar power from chosen spectrum
            path = self.gconf.get('General','input_power_wv')
            wv_vec,p_vec = np.loadtxt(path,skiprows=2,usecols=(0,2),unpack=True,delimiter=',')
            # Get interpolating function for power
            p_wv = interpolate.interp1d(wv_vec,p_vec,kind='linear',
                                       bounds_error=False,fill_value='extrapolate')
            # Assuming the leaves contain frequency values, sum over all of them
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf.get('General','sim_dir'),'ref_trans_abs.dat')
                with open(dpath,'r') as f:
                    ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
                freq = sim.conf.getfloat('Parameters','frequency')
                wvlgth = c.c/freq
                wvlgth_nm = wvlgth*1e9
                freqs[i] = freq
                wvlgths[i] = wvlgth_nm
                sun_pow = p_wv(wvlgth_nm) 
                spectra[i] = sun_pow
                abs_vals[i] = sun_pow*absorb
                ref_vals[i] = sun_pow*ref
                trans_vals[i] = sun_pow*trans
            # Now integrate all the weighted spectra and divide by the power of the spectra
            wvlgths = wvlgths[::-1]
            abs_vals = abs_vals[::-1]
            ref_vals = ref_vals[::-1]
            trans_vals = trans_vals[::-1]
            spectra = spectra[::-1]
            power = intg.trapz(spectra,x=wvlgths)
            wght_ref = intg.trapz(ref_vals,x=wvlgths)/power 
            wght_abs = intg.trapz(abs_vals,x=wvlgths)/power
            wght_trans = intg.trapz(trans_vals,x=wvlgths)/power
            out = os.path.join(base,'weighted_transmission_data.dat')
            with open(out,'w') as outf:
                outf.write('# Reflection, Transmission, Absorbtion\n')
                outf.write('%f,%f,%f'%(wght_ref,wght_trans,wght_abs))
        return wght_ref,wght_trans,wght_abs 

class Plotter(Processor):
    """Plots all the things listed in the config file"""
    def __init__(self,global_conf,sims=[],sim_groups=[],failed_sims=[]):
        super().__init__(global_conf,sims,sim_groups,failed_sims)
        self.log.debug("This is the plotter")
   
    def process(self,sim):
        sim.get_data()
        sim_path = os.path.basename(sim.conf.get('General','sim_dir'))
        self.log.info('Plotting data for sim %s',sim_path)
        # For each plot 
        for plot,args in self.gconf.items('Plotter'):
            self.log.info('Plotting %s with args %s',str(plot),str(args))
            for argset in args.split(';'):
                self.log.info('Passing following arg set to function %s: %s',str(plot),str(argset))
                if argset:
                    self.gen_plot(plot,sim,argset.split(','))
                else:
                    self.gen_plot(plot,sim,[])
        sim.clear_data()

    def process_all(self):
        self.log.info("Beginning local plotter method ...")
        for sim in self.sims:
            self.process(sim)

    def gen_plot(self,plot,sim,args):
        try:
            getattr(self,plot)(sim,*args)
        except KeyError:
            self.log.error("Unable to plot the following quantity: %s",
                           quantity,exc_info=True,stack_info=True)
            raise

    def heatmap2d(self,sim,x,y,cs,labels,ptype,draw=False,fixed=None,colorsMap='jet'):
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
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_ylim((np.amin(y),np.amax(y)))
        fig.suptitle(labels[3])
        # Draw geometric indicators and labels
        if draw:
            if ptype[-1] == 'z':
                self.log.info('draw nanowire circle')
                cent = sim.conf.getfloat('Parameters','array_period')/2.0
                rad = sim.conf.getfloat('Parameters','nw_radius')
                circ = mpatches.Circle((cent,cent),radius=rad,fill=False)
                ax.add_artist(circ)
            elif ptype[-1] == 'y' or ptype[-1] == 'x':
                self.log.info('draw layers')
                # Draw a line at the interface between each layer
                ito_line = sim.conf.getfloat('Parameters','air_t')
                nw_line = sim.conf.getfloat('Parameters','ito_t')+ito_line
                sub_line = sim.conf.getfloat('Parameters','nw_height')+nw_line
                air_line = sim.conf.getfloat('Parameters','substrate_t')+sub_line
                for line_h in [(ito_line,'ITO'),(nw_line,'NW'),(sub_line,'Substrate'),(air_line,'Air')]:
                    x = [0,sim.conf.getfloat('Parameters','array_period')]
                    y = [line_h[0],line_h[0]]
                    label_y = line_h[0] + 0.01
                    label_x = x[-1] - .01
                    plt.text(label_x,label_y,line_h[-1],ha='right',family='sans-serif',size=12)
                    line = mlines.Line2D(x,y,linestyle='solid',linewidth=2.0,color='black')
                    ax.add_line(line)
                # Draw two vertical lines to show the edges of the nanowire
                cent = sim.conf.getfloat('Parameters','array_period')/2.0
                rad = sim.conf.getfloat('Parameters','nw_radius')
                shell = sim.conf.getfloat('Parameters','shell_t')
                bottom = sim.conf.getfloat('Parameters','ito_t')+ito_line
                top = sim.conf.getfloat('Parameters','nw_height')+nw_line
                for x in (cent-rad,cent+rad,cent-rad-shell,cent+rad+shell):
                    xv = [x,x]
                    yv = [bottom,top]
                    line = mlines.Line2D(xv,yv,linestyle='solid',linewidth=2.0,color='black')
                    ax.add_line(line)
        if self.gconf.getboolean('General','save_plots'):
            name = labels[2]+'_'+ptype+'.pdf'
            path = os.path.join(sim.conf.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()             
        plt.close(fig)

    def plane_2d(self,sim,quantity,plane,pval,draw=False,fixed=None):
        """Plots a heatmap of a fixed 2D plane"""
        zs = sim.conf.getint('General','z_samples')
        xs = sim.conf.getint('General','x_samples')
        ys = sim.conf.getint('General','y_samples')
        height = sim.conf.getfloat('Parameters','total_height')
        #if plane == 'x' or plane == 'y':
        #    pval = int(pval)
        #else:
        #    # Find the nearest zval to the one we want. This is necessary because comparing floats
        #    # rarely works
        #    desired_val = (height/zs)*int(pval)
        #    pval = np.abs(sim.pos_inds[:,2]-desired_val).argmin()
        #    #pval = int(sim.pos_inds[ind,2])
        pval = int(pval)
        period = sim.conf.getfloat('Parameters','array_period')
        dx = period/xs
        dy = period/ys
        dz = height/zs
        x = np.arange(0,period,dx)
        y = np.arange(0,period,dy)
        z = np.arange(0,height+dz,dz)
        # Maps planes to an integer for extracting data
        plane_table = {'x': 0,'y': 1,'z':2}
        # Get the scalar values
        self.log.info('Retrieving scalar %s'%quantity)
        scalar = sim.get_scalar_quantity(quantity)
        ## Filter out any undesired data that isn't on the planes
        #mat = np.column_stack((sim.pos_inds[:,0],sim.pos_inds[:,1],sim.pos_inds[:,2],scalar))
        #planes = np.array([row for row in mat if row[plane_table[plane]] == pval])
        #self.log.debug("Planes shape: %s"%str(planes.shape))
        ## Get all unique values for x,y,z and convert them to actual values not indices
        #x,y,z = np.unique(planes[:,0])*dx,np.unique(planes[:,1])*dy,np.unique(planes[:,2])
        freq = sim.conf.getfloat('Parameters','frequency')
        wvlgth = (c.c/freq)*1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(freq,wvlgth)
        if fixed:
            # Super hacky and terrible way to fix the minimum and maximum values of the color bar
            # for a plot across all sims
            fixed = tuple(fixed.split(':'))
        # Get the plane we wish to plot
        self.log.info('Retrieving plane ...')
        cs = self.get_plane(scalar,xs,ys,zs,plane,pval)
        self.log.info('Plotting plane')
        if plane == 'x':
            #cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
            labels = ('y [um]','z [um]', quantity,title)
            self.heatmap2d(sim,y,z,cs,labels,'plane_2d_x',draw,fixed)
        elif plane == 'y':
            #cs = planes[:,-1].reshape(z.shape[0],x.shape[0])
            labels = ('x [um]','z [um]', quantity,title)
            self.heatmap2d(sim,x,z,cs,labels,'plane_2d_y',draw,fixed)
        elif plane == 'z':
            #cs = planes[:,-1].reshape(y.shape[0],x.shape[0])
            labels = ('y [um]','x [um]', quantity,title)
            self.heatmap2d(sim,x,y,cs,labels,'plane_2d_z',draw,fixed)
    
    def scatter3d(self,sim,x,y,z,cs,labels,ptype,colorsMap='jet'):
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
        fig.suptitle(os.path.basename(sim.conf.get('General','sim_dir')))
        if self.gconf.getboolean('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(sim.conf.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)
    
    def full_3d(self,sim,quantity):
        """Generates a full 3D plot of a specified scalar quantity"""
        period = sim.conf.getfloat('Parameters','array_period')
        dx = period/sim.conf.getfloat('General','x_samples')
        dy = period/sim.conf.getfloat('General','y_samples')
        # The data just tells you what integer grid point you are on. Not what actual x,y coordinate you
        # are at
        xpos = sim.pos_inds[:,0]*dx
        ypos = sim.pos_inds[:,1]*dy
        zpos = sim.pos_inds[:,2] 
        # Get the scalar
        scalar = sim.get_scalar_quantity(quantity)
        labels = ('X [um]','Y [um]','Z [um]',quantity)
        # Now plot! 
        self.scatter3d(sim,xpos,ypos,zpos,scalar,labels,'full_3d')
        
    def planes_3d(self,sim,quantity,xplane,yplane):
        """Plots some scalar quantity in 3D but only along specified x-z and y-z planes"""
        xplane = int(xplane)
        yplane = int(yplane)
        period = sim.conf.getfloat('Parameters','array_period')
        dx = period/sim.conf.getfloat('General','x_samples')
        dy = period/sim.conf.getfloat('General','y_samples')
        # Get the scalar values
        scalar = sim.get_scalar_quantity(quantity) 
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((sim.pos_inds[:,0],sim.pos_inds[:,1],sim.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[0] == xplane or row[1] == yplane])
        planes[:,0] = planes[:,0]*dx
        planes[:,1] = planes[:,1]*dy
        labels = ('X [um]','Y [um]','Z [um]',quantity)
        # Now plot!
        self.scatter3d(sim,planes[:,0],planes[:,1],planes[:,2],planes[:,3],labels,'planes_3d')

    def line_plot(self,sim,x,y,ptype,labels):
        """Make a simple line plot"""
        fig = plt.figure()
        plt.plot(x,y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(labels[2])
        if self.gconf.getboolean('General','save_plots'):
            name = labels[1]+'_'+ptype+'.pdf'
            path = os.path.join(sim.conf.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)

    def fixed_line(self,sim,quantity,direction,coord1,coord2):
        """Plot a scalar quantity on a line along a the z direction at some pair of
        coordinates in the plane perpendicular to that direction"""
        coord1 = int(coord1)
        coord2 = int(coord2)
        period = sim.conf.getfloat('Parameters','array_period')
        dx = period/sim.conf.getfloat('General','x_samples')
        dy = period/sim.conf.getfloat('General','y_samples')
        # Get the scalar values
        scalar = sim.get_scalar_quantity(quantity) 
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((sim.pos_inds[:,0],sim.pos_inds[:,1],sim.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[0] == coord1 and row[1] == coord2])
        planes[:,0] = planes[:,0]*dx
        planes[:,1] = planes[:,1]*dy
        freq = sim.conf.getfloat('Parameters','frequency')
        wvlgth = (c.c/freq)*1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(freq,wvlgth)
        labels = ('Z [um]',quantity,title) 
        ptype = "%s_line_plot_%i_%i"%(direction,coord1,coord2)
        self.line_plot(sim,planes[:,2],planes[:,3],ptype,labels)

class Global_Plotter(Plotter):
    """Plots global quantities for an entire run that are not specific to a single simulation"""
    def __init__(self,global_conf,sims=[],sim_groups=[],failed_sims=[]):
        super().__init__(global_conf,sims,sim_groups,failed_sims)
        self.log.debug("Global plotter init")

    def process_all(self):
        self.log.info('Beginning global plotter method ...')
        for plot,args in self.gconf.items('Global_Plotter'):
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

    def convergence(self,quantity,err_type='global',scale='linear'):
        """Plots the convergence of a field across all available simulations"""
        self.log.info('Plotting convergence')
        for group in self.sim_groups:
            base = group[0].conf.get('General','basedir')
            if err_type == 'local':
                fglob = os.path.join(base,'localerror_%s*.dat'%quantity) 
            elif err_type == 'global':
                fglob = os.path.join(base,'globalerror_%s*.dat'%quantity) 
            elif err_type == 'adjacent':
                fglob = os.path.join(base,'adjacenterror_%s*.dat'%quantity) 
            else:
                self.log.error('Attempting to plot an unsupported error type')
                raise ValueError
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

    
    def global_avg(self,quantity,avg_type):
        """Combine local average of a specific quantity for all leaves in each group"""
        for group in self.sim_groups:
            base = group[0].conf.get('General','basedir')
            self.log.info('Computing global averages for group at %s'%base)
            key = '%s_%s'%(quantity,avg_type)
            path = os.path.join(base,'%s_%s_global.avg.crnch'%(quantity,avg_type))
            cs = np.loadtxt(path)
            cm = plt.get_cmap('jet')
            cNorm = matplotlib.colors.Normalize(vmin=np.amin(cs), vmax=np.amax(cs))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            fig = plt.figure(figsize=(9,7))
            ax = fig.add_subplot(111)
            labels = ('Radial Distance','Depth',quantity,avg_type)
            ax.pcolormesh(cs,cmap=cm,norm=cNorm,alpha=.5)
            scalarMap.set_array(cs)
            cb = fig.colorbar(scalarMap)
            cb.set_label(labels[2])
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            rmax = group[0].conf.getfloat('Parameters','array_period')/np.sqrt(np.pi)
            dr = rmax/group[0].conf.getint('General','r_samples')
            dz = group[0].conf.getfloat('Parameters','total_height')/group[0].conf.getfloat('General','z_samples')
            #fig.canvas.draw()
            #xlabels = ['%.2f'%(int(item.get_text())*dr) for item in ax.get_xticklabels()]
            #ylabels = ['%.2f'%(int(item.get_text())*dz) for item in ax.get_yticklabels()]
            #ax.set_xticklabels(xlabels)
            #ax.set_yticklabels(ylabels)
            fig.suptitle(labels[3])
            fig.savefig(os.path.join(base,'%s_%s_global.avg.pdf'%(quantity,avg_type)))

    
    def transmission_data(self,absorbance,reflectance,transmission):
        """Plot transmissions, absorption, and reflectance assuming leaves are frequency"""
        truthy = ['True','true','t','yes']
        for group in self.sim_groups:
            base = group[0].conf.get('General','basedir')
            self.log.info('Plotting transmission data for group at %s'%base)
            # Assuming the leaves contain frequency values, sum over all of them
            freqs = np.zeros(len(group))
            refl_l = np.zeros(len(group))
            trans_l = np.zeros(len(group))
            absorb_l = np.zeros(len(group))
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf.get('General','sim_dir'),'ref_trans_abs.dat')
                with open(dpath,'r') as f:
                    ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
                freq = sim.conf.getfloat('Parameters','frequency')
                freqs[i] = freq
                trans_l[i] = trans
                refl_l[i] = ref
                absorb_l[i] = absorb
            freqs = (c.c/freqs[::-1])*1e9
            refl_l = refl_l[::-1]
            absorb_l = absorb_l[::-1]
            trans_l = trans_l[::-1]
            plt.figure()
            if absorbance in truthy:
                self.log.info('Plotting absorbance')
                plt.plot(freqs,absorb_l,label='Absorption')
            if reflectance in truthy:
                plt.plot(freqs,refl_l,label='Reflectance')
            if transmission in truthy:
                plt.plot(freqs,trans_l,label='Transmission')
            plt.legend(loc='best')
            figp = os.path.join(base,'transmission_plots.pdf')
            plt.xlabel('Wavelength (nm)') 
            #plt.ylim((0,.5))
            plt.savefig(figp)
            plt.close()

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
    parser.add_argument('--filter_by',nargs='*',help="""List of parameters you wish to filter by,
            specified like: p1:v1,v2,v3 p2:v1,v2,v3""")
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
  

    # Collect the sims once up here and reuse them later
    proc = Processor(conf)
    sims, sim_groups, failed_sims = proc.collect_sims()
    # Filter if specified
    if args.filter_by:
        filt_dict = {}
        for item in args.filter_by:
            par,vals = item.split(':')
            vals = vals.split(',')
            filt_dict[par] = vals
        logger.info('Here is the filter dictionary: %s'%filt_dict)
        sims, sim_groups = proc.filter_by_param(filt_dict)
    # Now do all the work
    if not args.no_crunch:
        crunchr = Cruncher(conf,sims,sim_groups,failed_sims)
        #crunchr.process_all()
        for sim in crunchr.sims:
            crunchr.transmissionData(sim)
    if not args.no_gcrunch:
        gcrunchr = Global_Cruncher(conf,sims,sim_groups,failed_sims)
        gcrunchr.process_all()
    if not args.no_plot:
        pltr = Plotter(conf,sims,sim_groups,failed_sims) 
        pltr.process_all()
    if not args.no_gplot:
        gpltr = Global_Plotter(conf,sims,sim_groups,failed_sims)
        #gpltr.collect_sims()
        gpltr.process_all()

if __name__ == '__main__':
    main()
