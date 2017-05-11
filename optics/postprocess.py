import numpy as np
from scipy import interpolate
import scipy.constants as c
import scipy.integrate as intg
import argparse as ap
import os
import time
import copy
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
plt.style.use(['ggplot', 'paper'])
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# Literally just for the initial data load
import pandas
import multiprocessing as mp
import multiprocessing.dummy as mpd

from utils.config import Config
from utils.utils import configure_logger,cmp_dicts, open_atomic

# Configure module level logger if not running as main process
if not __name__ == '__main__':
    logger = configure_logger(level='INFO',name='postprocess',
                              console=True,logfile='logs/postprocess.log')

def counted(fn):
    def wrapper(self):
        wrapper.called += 1
        return fn(self)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper

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
        self.log.info('Collecting raw data for sim %s',self.conf['General']['sim_dir'])
        sim_path = self.conf['General']['sim_dir']
        base_name = self.conf['General']['base_name']
        ignore = self.conf['General']['ignore_h']
        ftype = self.conf['General']['save_as']
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
            e_path = os.path.join(sim_path,base_name+'.E.npz')
            e_data, e_lookup = self.load_npz(e_path)
            self.log.debug('E shape after getting: %s',str(e_data.shape))
            pos_inds = np.zeros((e_data.shape[0],3))
            pos_inds[:,:] = e_data[:,0:3]
            # Load H field data
            if not ignore:
                h_path = os.path.join(sim_path,base_name+'.H.npz')
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
        self.log.info('Collecting data for sim %s',self.conf['General']['sim_dir'])
        sim_path = self.conf['General']['sim_dir']
        base_name = self.conf['General']['base_name']
        ftype = self.conf['General']['save_as']
        ignore = self.conf['General']['ignore_h']
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
        base = self.conf['General']['sim_dir']
        ftype = self.conf['General']['save_as']
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
        """Writes the data. All writes have been wrapped with an atomic context
        manager than ensures all writes are atomic and do not corrupt data
        files if they are interrupted for any reason"""
        start = time.time()
        # Get the current path
        base = self.conf['General']['sim_dir']
        ignore = self.conf['General']['ignore_h']
        self.log.info('Writing data for %s'%base)
        fname = self.conf['General']['base_name']
        epath = os.path.join(base,fname+'.E')
        hpath = os.path.join(base,fname+'.H')
        # Save matrices in specified file tipe
        self.log.debug('Here are the E matrix headers: %s',str(self.e_lookup))
        self.log.debug('Here are the H matrix headers: %s',str(self.h_lookup))
        ftype = self.conf['General']['save_as']
        if ftype == 'text':
            epath = epath+'.crnch'
            # These make sure all writes are atomic and thus we won't get any
            # partially written files if processing is interrupted for any
            # reason (like a keyboard interrupt)
            with open_atomic(epath,'w',npz=False) as out:
                np.savetxt(out,self.e_data,header=','.join(self.e_lookup.keys()))
            if not ignore:
                hpath = hpath+'.crnch'
                with open_atomic(hpath,'w',npz=False) as out:
                    np.savetxt(out,self.h_data,header=','.join(self.h_lookup.keys()))
            # Save any local averages we have computed
            for avg, mat in self.avgs.items():
                dpath = os.path.join(base,avg+'.avg.crnch')
                with open_atomic(dpath,'w',npz=False) as out:
                    np.savetxt(out,mat)
        elif ftype == 'npz':
            # Save the headers and the data
            with open_atomic(epath,'w') as out:
                np.savez_compressed(out,headers=np.array([self.e_lookup]), data=self.e_data)
            if not ignore:
                with open_atomic(hpath,'w') as out:
                    np.savez_compressed(out,headers=np.array([self.h_lookup]), data=self.h_data)
            # Save any local averages we have computed
            dpath = os.path.join(base,'all.avg')
            with open_atomic(dpath,'w') as out:
                np.savez_compressed(out,**self.avgs)
        else:
            raise ValueError('Specified saving in an unsupported file format')
        end = time.time()
        self.log.info('Write time: {:.2} seconds'.format(end-start))

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

    def collect_sims(self):
        """Collect all the simulations beneath the base of the directory tree"""
        # Clear out the lists
        self.sims = []
        self.failed_sims = []
        ftype = self.gconf['General']['save_as']
        if ftype == 'text':
            datfile = self.gconf['General']['base_name']+'.E'
        else:
            datfile = self.gconf['General']['base_name']+'.E.npz'

        group_dict = {}
        for root,dirs,files in os.walk(self.gconf['General']['base_dir']):
            if 'sim_conf.yml' in files and datfile in files:
                self.log.info('Gather sim at %s',root)
                sim_obj = Simulation(Config(os.path.join(root,'sim_conf.yml')))
                self.sims.append(sim_obj)
                # This retrieves the lowest node in the tree, stores that node
                # as the key and the list of sims beneath that node as values
                parent_dir = os.path.dirname(root)
                if parent_dir not in group_dict:
                    group_dict[parent_dir] = [sim_obj]
                else:
                    group_dict[parent_dir].append(sim_obj)
            elif 'sim_conf.yml' in files:
                sim_obj = Simulation(Config(os.path.join(root,'sim_conf.yml')))
                self.log.error('The following sim is missing its data file: %s',
                               sim_obj.conf['General']['sim_dir'])
                self.failed_sims.append(sim_obj)
        return self.sims,self.failed_sims

    def sort_sims(self):
        """Sorts simulations by their parameters the way a human would. Called human sorting or
        natural sorting. Thanks stackoverflow"""

        self.sims.sort(key=self.sim_key)
        for group in self.sim_groups:
            paths = [sim.conf['General']['sim_dir'] for sim in group]
            self.log.debug('Group paths before sorting: %s',str(paths))
            group.sort(key=self.sim_key)
            paths = [sim.conf['General']['sim_dir'] for sim in group]
            self.log.debug('Group paths after sorting: %s',str(paths))

    def get_param_vals(self,parseq):
        """Return all possible values of the provided parameter for this sweep"""
        vals = []
        for sim in self.sims:
            val = sim.conf[parseq]
            if val not in vals:
                vals.append(val)
        return vals

    def filter_by_param(self,pars):
        """Accepts a dict where the keys are parameter names and the values are a list of possible
        values for that parameter. Any simulation whose parameter does not match any of the provided
        values is removed from the sims and sim_groups attribute"""

        assert(type(pars) == dict)
        for par,vals in pars.items():
            self.sims = [sim for sim in self.sims if sim.conf[par] in vals]
            groups = []
            for group in self.sim_groups:
                filt_group = [sim for sim in group if sim.conf[par] in vals]
                groups.append(filt_group)
            self.sim_groups = groups
        assert(len(self.sims) >= 1)
        return self.sims,self.sim_groups

    def group_against(self, key, variable_params, sort_key=None):
        """Groups simulations by against particular parameter. Within each
        group, the parameter specified will vary, and all other
        parameters will remain fixed. Populates the sim_groups attribute and
        also returns a list of lists. The simulations with each group will be
        sorted in increasing order of the specified parameter. An optional key
        may be passed in, the groups will be sorted in increasing order of the
        specified key"""

        self.log.info('Grouping sims against: %s'%str(key))
        # We need only need a shallow copy of the list containing all the sim objects
        # We don't want to modify the orig list but we wish to share the sim
        # objects the two lists contain
        sims = copy.copy(self.sims)
        sim_groups = [[sims.pop()]]
        # While there are still sims that havent been assigned to a group
        while sims:
            # Get the comparison dict for this sim
            sim = sims.pop()
            val1 = sim.conf[key]
            # We want the specified key to vary, so we remove it from the
            # comparison dict
            del sim.conf[key]
            cmp1 = {'Simulation':sim.conf['Simulation'],'Layers':sim.conf['Layers']}
            match = False
            # Loop through each group, checking if this sim belongs in the
            # group
            for group in sim_groups:
                sim2 = group[0]
                val2 = sim2.conf[key]
                del sim2.conf[key]
                cmp2 = {'Simulation':group[0].conf['Simulation'],'Layers':group[0].conf['Layers']}
                params_same = cmp_dicts(cmp1,cmp2)
                if params_same:
                    match = True
                    # We need to restore the param we removed from the
                    # configuration earlier
                    sim.conf[key] = val1
                    group.append(sim)
                group[0].conf[key] = val2
            # If we didnt find a matching group, we need to create a new group
            # for this simulation
            if not match:
                sim.conf[key] = val1
                sim_groups.append([sim])
        # Get the params that will define the path in the results dir for each
        # group that will be stored
        ag_key = tuple(key[0:-1])
        result_pars = [var for var in variable_params if var != ag_key]
        for group in sim_groups:
            # Sort the individual sims within a group in increasing order of
            # the parameter we are grouping against a
            group.sort(key=lambda sim: sim.conf[key])
            path = '{}/grouped_against_{}'.format(group[0].conf['General']['treebase'],
                                                  ag_key[-1])
            # If the only variable param is the one we grouped against, make
            # the top dir
            if not result_pars:
                try:
                    os.makedirs(path)
                except OSError:
                    pass
            # Otherwise make the top dir and all the subdirs
            else:
                for par in result_pars:
                    full_key = par+('value',)
                    # All sims in the group will have the same values for
                    # result_pars so we can just use the first sim in the group
                    path = os.path.join(path, '{}_{:.4E}/'.format(par[-1],
                                        group[0].conf[full_key]))
                    self.log.info('RESULTS DIR: {}'.format(path))
                    try:
                        os.makedirs(path)
                    except OSError:
                        pass
            for sim in group:
                sim.conf['General']['results_dir'] = path
        # Sort the groups in increasing order of the provided sort key
        if sort_key:
            sim_groups.sort(key=lambda group: group[0].conf[key])
        self.sim_groups = sim_groups
        return sim_groups

    def group_by(self,key,sort_key=None):
        """Groups simulations by a particular parameter. Within each group, the
        parameter specified will remain fixed, and all other parameters will
        vary. Populates the sim_groups attribute and also returns a list of
        lists. The groups will be sorted in increasing order of the specified
        parameter. An optional key may be passed in, the individual sims within
        each group will be sorted in increasing order of the specified key"""

        self.log.info('Grouping sims by: %s'%str(key))
        # This works by storing the different values of the specifed parameter
        # as keys, and a list of sims whose value matches the key as the value
        pdict = {}
        for sim in self.sims:
            if sim.conf[key] in pdict:
                pdict[sim.conf[key]].append(sim)
            else:
                pdict[sim.conf[key]] = [sim]
        # Now all the sims with matching values for the provided key are just
        # the lists located at each key. We sort the groups in increasing order
        # of the provided key
        groups = sorted(pdict.values(),key=lambda group: group[0].conf[key])
        # If specified, sort the sims within each group in increasing order of
        # the provided sorting key
        if sort_key:
            for group in groups:
                group.sort(key=lambda sim: sim.conf[sort_key])
        self.sim_groups = groups
        return groups

    def get_plane(self,arr,xsamp,ysamp,zsamp,plane,pval):
        """Given a 1D array containing values for a 3D scalar field, reshapes
        the array into 3D and returns a 2D array containing the data on a given
        plane, for a specified index value (pval) of that plane. So, specifying
        plane=x and pval=30 would return data on the 30th y,z plane (a plane at
        the given x index). The number of samples (i.e data points) in each
        coordinate direction need not be equal"""
	
        zsamp = int(zsamp)
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

    def get_line(self,arr,xsamp,ysamp,zsamp,line_dir,c1,c2):
        """Given a 1D array containing values for a 3D scalar field, reshapes
        the array into 3D and returns a new 1D array containing the data on a
        line in a given direction, for a specified index value for the other
        two spatial coordinates. So, specifying line_dir=z and c1=5,c2=5 would
        return all the data along the z-direction at the 5th x,y index. Note
        coordinates c1,c2 must always be specified in (x,y,z) order"""

        scalar = arr.reshape(zsamp+1,xsamp,ysamp)
        if line_dir == 'x':
            # z along rows, y along columns
            return scalar[c2,:,c1]
        elif line_dir == 'y':
            # x along columns, z along rows
            return scalar[c2,c1,:]
        elif line_dir == 'z':
            # x along rows, y along columns
            return scalar[:,c1,c2]

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
        super(Cruncher,self).__init__(global_conf,sims,sim_groups,failed_sims)
        self.log.debug("This is THE CRUNCHER!!!!!")

    def calculate(self,quantity,sim,args):
        try:
            getattr(self,quantity)(sim,*args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity,exc_info=True,stack_info=True)
            raise

    def process(self,sim):
        sim_path = os.path.basename(sim.conf['General']['sim_dir'])
        self.log.info('Crunching data for sim %s',sim_path)
        sim.get_raw_data()
        self.log.debug('SHAPE BEFORE CALCULATING: %s'%str(sim.e_data.shape))
        if sim.failed:
            self.log.error('Following simulation missing data: %s',sim_path)
            self.failed_sims.append(sim)
        else:
            # For each quantity
            for quant,data in self.gconf['Postprocessing']['Cruncher'].items():
                if data['compute']:
                    argsets = data['args']
                    self.log.info('Computing %s with args %s',
                                  str(quant),str(argsets))
                    if argsets and type(argsets[0]) == list:
                        for argset in argsets:
                            self.log.info('Computing individual argset'
                                          ' %s',str(argset))
                            if argset:
                                self.calculate(quant,sim,argset)
                            else:
                                self.calculate(quant,sim,[])
                            self.log.debug('SHAPE AFTER CALCULATING: %s'%str(sim.e_data.shape))
                    else:
                        if argsets:
                            self.calculate(quant,sim,argsets)
                        else:
                            self.calculate(quant,sim,[])
                        self.log.debug('SHAPE AFTER CALCULATING: %s'%str(sim.e_data.shape))
                    self.log.debug('E lookup: %s',str(sim.e_lookup))
                    self.log.debug('H lookup: %s',str(sim.h_lookup))

            sim.write_data()
            sim.clear_data()

    def process_all(self):
        self.log.info('Beginning data crunch ...')
        if not self.gconf['General']['post_parallel']:
            for sim in self.sims:
                self.process(sim)
        else:
            num_procs = mp.cpu_count() - self.gconf['General']['reserved_cores']
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

    def _get_circle_nk(self,shape,sdata,nk,samps,steps):
        """Returns a 2D matrix containing the N,K values at each point in space
        for a circular in-plane geometry"""
        # Set up the parameters
        cx,cy = sdata['center']['x'],sdata['center']['y']
        rad = sdata['radius']
        rad_sq = rad*rad
        mat = sdata['material']
        dx = steps[0]
        dy = steps[1]
        # Build the matrix
        nk_mat = np.zeros((samps[1],samps[0]))
        for xi in range(samps[0]):
            for yi in range(samps[1]):
                dist = ((xi*dx)-cx)**2 + ((yi*dy)-cy)**2
                if dist <= rad_sq:
                    nk_mat[yi,xi] = nk[mat][0]*nk[mat][1]
        return nk_mat

    def _genrate_nk_geometry(self,sim,lname,nk,samps,steps):
        """Computes the nk profile in a layer with a nontrivial internal
        geometry. Returns a 2D matrix containing the product of n and k at each
        point
        lname: Name of the layer as a string
        ldata: This dict containing all the data for this layer
        nk: A dictionary with the material name as the key an a tuple
        containing (n,k) as the value
        samps: A tuple/list containing the number of sampling points in each
        spatial direction in (x,y,z) order
        steps: Same as samps but instead contains the step sizes in each
        direction"""
        # Initialize the matrix with values for the base material
        base_mat = sim.conf['Layers'][lname]['base_material']
        nk_mat = nk[base_mat][0]*nk[base_mat][1]*np.ones((samps[1],samps[0]))
        # Get the shapes sorted in reverse order
        shapes = sim.conf.sorted_dict(sim.conf['Layers'][lname]['geometry'])
        # Loop through the layers. We want them in reverse order so the
        # smallest shape, which is contained within all the other shapes and
        # should override their nk values, goes last
        for shape,sdata in shapes.items():
            if sdata['type'] == 'circle':
                update = self._get_circle_nk(shape,sdata,nk,samps,steps)
            else:
                raise NotImplementedError('Computing generation rate for layers'
                                          ' with %s shapes is not currently supported'%sdata['type'])
            # Update the matrix with the values from this new shape. The update
            # array will contain nonzero values within the shape, and zero
            # everywhere else. This line updates the nk_mat with only the
            # nonzero from the update matrix, and leaves all other elements
            # untouched
            nk_mat = np.where(update != 0,update,nk_mat)
        return nk_mat

    def genRate(self,sim):
        # We need to compute normEsquared before we can compute the generation rate
        try:
            normEsq = sim.get_scalar_quantity('normEsquared')
        except KeyError:
            normEsq = self.normEsquared(sim)
            # Make sure we don't compute it twice
            try:
                sim.conf['Postprocessing']['Cruncher']['normEsquared']['compute'] = False
            except KeyError:
                pass
        # Prefactor for generation rate. Not we gotta convert from m^3 to cm^3, hence 1e6 factor
        fact = c.epsilon_0/(c.hbar*1e6)
        # Get the indices of refraction at this frequency
        freq = sim.conf['Simulation']['params']['frequency']['value']
        nk = {mat:(self.get_nk(matpath,freq)) for mat,matpath in
              sim.conf['Materials'].items()}
        nk['vacuum'] = (1,0)
        self.log.debug(nk)
        # Get spatial discretization
        z_samples = sim.conf['Simulation']['z_samples']
        x_samples = sim.conf['Simulation']['x_samples']
        y_samples = sim.conf['Simulation']['y_samples']
        z_samples = int(z_samples)
        samps = (x_samples,y_samples,z_samples)
        # Reshape into an actual 3D matrix. Rows correspond to different y fixed x, columns to fixed
        # y variable x, and each layer in depth is a new z value
        normEsq = np.reshape(normEsq,(z_samples+1,x_samples,y_samples))
        gvec = np.zeros_like(normEsq)
        max_depth = sim.conf['Simulation']['max_depth']
        dz = max_depth/z_samples
        period = sim.conf['Simulation']['params']['array_period']['value']
        dx = period/x_samples
        dy = period/y_samples
        steps = (dx,dy,dz)
        # Main loop to compute generation in each layer
        boundaries = []
        count = 0
        ordered_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        for layer,ldata in ordered_layers.items():
            # Get boundaries between layers and their starting and ending indices
            layer_t = ldata['params']['thickness']['value']
            self.log.debug('LAYER: %s'%layer)
            self.log.debug('LAYER T: %f'%layer_t)
            if count == 0:
                start = 0
                end = int(layer_t/dz)+1
                boundaries.append((layer_t,start,end))
            else:
                prev_tup = boundaries[count-1]
                dist = prev_tup[0]+layer_t
                start = prev_tup[2]
                end = int(dist/dz) + 1
                boundaries.append((dist,start,end))
            self.log.debug('START: %i'%start)
            self.log.debug('END: %i'%end)
            if 'geometry' in ldata:
                self.log.debug('HAS GEOMETRY')
                # This function returns the N,K profile in that layer as a 2D
                # matrix. Each element contains the product of n and k at that
                # point, using the NK values for the appropriate material
                nk_mat = self._genrate_nk_geometry(sim,layer,nk,samps,steps)
                gvec[start:end,:,:] = fact*nk_mat*normEsq[start:end,:,:]
            else:
                # Its just a simple slab
                self.log.debug('NO GEOMETRY')
                lmat = ldata['base_material']
                self.log.debug('LAYER MATERIAL: %s'%lmat)
                self.log.debug('MATERIAL n: %s'%str(nk[lmat][0]))
                self.log.debug('MATERIAL k: %s'%str(nk[lmat][1]))
                region = fact*nk[lmat][0]*nk[lmat][1]*normEsq[start:end,:,:]
                self.log.debug('REGION SHAPE: %s'%str(region.shape))
                self.log.debug('REGION: ')
                self.log.debug(str(region))
                gvec[start:end,:,:] = region 
            self.log.debug('GEN RATE MATRIX: ')
            self.log.debug(str(gvec))
            count += 1
        # Reshape back to 1D array
        gvec = gvec.reshape((x_samples*y_samples*(z_samples+1)))
        self.log.debug('GVEC AFTER FLATTENING: ')
        self.log.debug(str(gvec))
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
        try:
            quant = sim.get_scalar_quantity(quantity)
        except KeyError:
            self.calculate(quantity,sim,[])
            quant = sim.get_scalar_quantity(quantity)
            # Make sure we don't compute it twice
            try:
                sim.conf['Postprocessing']['Cruncher'][quantity]['compute'] = False
            except KeyError:
                pass
        quant = sim.get_scalar_quantity(quantity)
        # Get spatial discretization
        z_samples = sim.conf['Simulation']['z_samples']
        x_samples = sim.conf['Simulation']['x_samples']
        y_samples = sim.conf['Simulation']['y_samples']
        z_samples = int(z_samples)
        rsamp = sim.conf['Simulation']['r_samples']
        thsamp = sim.conf['Simulation']['theta_samples']
        # Reshape into an actual 3D matrix. Rows correspond to different y fixed x, columns to fixed
        # y variable x, and each layer in depth is a new z value
        values = np.reshape(quant,(z_samples+1,x_samples,y_samples))
        period = sim.conf['Simulation']['params']['array_period']['value']
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
        self.log.info('Computing transmission data ...')
        base = sim.conf['General']['sim_dir']
        path = os.path.join(base,'fluxes.dat')
        data = {}
        with open(path,'r') as f:
            d = f.readlines()
            for line in d[1:]:
                els = line.split(',')
                key = els.pop(0)
                data[key] = list(map(float,els))
        # sorted_layers is an OrderedDict, and thus has the popitem method
        sorted_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        self.log.info('SORTED LAYERS: %s'%str(sorted_layers))
        first_layer = sorted_layers.popitem(last=False)
        # self.log.info('FIRST LAYER: %s'%str(first_layer))
        # An ordered dict is actually just a list of tuples so we can access
        # the key directly like so
        first_name = first_layer[0]
        last_layer = sorted_layers.popitem()
        # Port at top of substrate
        last_name = last_layer[0]
        # Port at bottom of substrate
        # last_name = last_layer[0]+'_bottom'
        # self.log.info('LAST LAYER: %s'%str(last_layer))
        # p_inc = data[first_name][0]
        # p_ref = np.abs(data[first_name][1])
        # p_trans = data[last_name][0]
        p_inc = np.sqrt(data[first_name][0]**2+data[first_name][2]**2)
        p_ref = np.sqrt(data[first_name][1]**2+data[first_name][3]**2)
        p_trans = np.sqrt(data[last_name][0]**2+data[last_name][2]**2)
        reflectance = p_ref / p_inc
        transmission = p_trans / p_inc
        absorbance = 1 - reflectance - transmission
        #absorbance = 1 - reflectance
        tot = reflectance+transmission+absorbance
        delta = np.abs(tot-1)
        #self.log.info('Total = %f'%tot)
        assert(reflectance > 0)
        assert(transmission > 0)
        assert(delta < .0001)
        self.log.debug('Reflectance %f'%reflectance)
        self.log.debug('Transmission %f'%transmission)
        self.log.debug('Absorbance %f'%absorbance)
        #assert(reflectance >= 0 and transmission >= 0 and absorbance >= 0)
        outpath = os.path.join(base,'ref_trans_abs.dat')
        self.log.info('Writing transmission file')
        with open(outpath,'w') as out:
            out.write('# Reflectance,Transmission,Absorbance\n')
            out.write('%f,%f,%f'%(reflectance,transmission,absorbance))
        return reflectance,transmission,absorbance

    def integrated_absorbtion(self,sim):
        """Computes the absorption of a layer by using the volume integral of the product of the
        imaginary part of the relative permittivity and the norm squared of the E field"""
        raise NotImplementedError('There are some bugs in S4 and other reasons'
                                  ' that this function doesnt work yet')
        base = sim.conf['General']['sim_dir']
        path = os.path.join(base,'integrated_absorption.dat')
        inpath = os.path.join(base,'energy_densities.dat')
        freq = sim.conf['Simulation']['params']['frequency']['value']
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
        super(Global_Cruncher,self).__init__(global_conf,sims,sim_groups,failed_sims)
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
        for quant,data in self.gconf['Postprocessing']['Global_Cruncher'].items():
            if data['compute']:
                argsets = data['args']
                self.log.info('Computing %s with args %s',
                              str(quant),str(argsets))
                if argsets and type(argsets[0]) == list:
                    for argset in argsets:
                        self.log.info('Computing individual argset'
                                      ' %s',str(argset))
                        if argset:
                            self.calculate(quant,argset)
                        else:
                            self.calculate(quant,[])
                else:
                    if argsets:
                        self.calculate(quant,argsets)
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
        #TODO: This function is definitely not general. We need to get a list
        # of layers to exclude from the user. For now, just assume we want to
        # exclude the top and bottom regions
        x_samples = sim.conf['Simulation']['x_samples']
        y_samples = sim.conf['Simulation']['y_samples']
        z_samples = sim.conf['Simulation']['z_samples']
        total_h = sim.conf.get_height()
        dz = total_h/z_samples
        # sorted_layers is an OrderedDict, and thus has the popitem method
        sorted_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        first_layer = sorted_layers.popitem(last=False)
        last_layer = sorted_layers.popitem()
        # We can get the starting and ending planes from their heights
        start_plane = int(round(first_layer['params']['thickness']['value']/dz))
        start = start_plane*(x_samples*y_samples)
        end_plane = int(round(last_layer['params']['thickness']['value']/dz))
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
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
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
                                  ref_sim.conf['Simulation']['params']['numbasis']['value'],
                                  sim2.conf['Simulation']['params']['numbasis']['value'])
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
                    errfile.write('%i,%f\n'%(sim2.conf['Simulation']['params']['numbasis']['value'],avg_diffvec_mag))
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
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
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
                                  ref_sim.conf['Simulation']['params']['numbasis']['value'],
                                  sim2.conf['Simulation']['params']['numbasis']['value'])
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
                    errfile.write('%i,%f\n'%(sim2.conf['Simulation']['params']['numbasis']['value'],error))
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
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dirs']
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
                                  ref_sim.conf['Simulation']['params']['numbasis']['value'],
                                  sim2.conf['Simulation']['params']['numbasis']['value'])
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
                    errfile.write('%i,%f\n'%(sim2.conf['Simulation']['params']['numbasis']['value'],error))
                    sim2.clear_data()
                    ref_sim.clear_data()

    def scalar_reduce(self,quantity,avg=False):
        """Combine a scalar quantity across all simulations in each group. If
        avg=False then a direct sum is computed, otherwise an average is
        computed"""
        for group in self.sim_groups:
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info('Performing scalar reduction for group at %s'%base)
            group[0].get_data()
            group_comb = group[0].get_scalar_quantity(quantity)
            group[0].clear_data()
            # This approach is more memory efficient then building a 2D array
            # of all the data from each group and summing along an axis
            for i in range(1,len(group)):
                group[i].get_data()
                group_comb += group[i].get_scalar_quantity(quantity)
                group[i].clear_data()
            if avg:
                group_comb = group_comb/len(group)
                fname = 'scalar_reduce_avg_%s'%quantity
            else:
                fname = 'scalar_reduce_%s'%quantity

            path = os.path.join(base,fname)
            if group[0].conf['General']['save_as'] == 'npz':
                np.save(path,group_comb)
            elif group[0].conf['General']['save_as'] == 'text':
                path += '.crnch'
                np.savetxt(path,group_comb)
            else:
                raise ValueError('Invalid file type in config')

    def fractional_absorbtion(self):
        """Computes the fraction of the incident spectrum that is absorbed by
        the device. This is a unitless number, and its interpretation somewhat
        depends on the units you express the incident power in. If you
        expressed your incident spectrum in photon number, this can be
        interpreted as the fraction of incident photons that were absorbed. If
        you expressed your incident spectrum in terms of power per unit area,
        then this can be interpreted as the fraction of incident power per unit
        area that gets absorbed. In summary, its the amount of _whatever you
        put in_ that is being absorbed by the device."""
        valuelist = []
        for group in self.sim_groups:
            base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info('Computing fractional absorbtion for group at %s'%base)
            vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Assuming the sims have been grouped by frequency, sum over all of them
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf['General']['sim_dir'],'ref_trans_abs.dat')
                with open(dpath,'r') as f:
                    ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
                freq = sim.conf['Simulation']['params']['frequency']['value']
                wvlgth = c.c/freq
                wvlgth_nm = wvlgth*1e9
                freqs[i] = freq
                wvlgths[i] = wvlgth
                # Get solar power from chosen spectrum
                path = sim.conf['Simulation']['input_power_wv']
                wv_vec,p_vec = np.loadtxt(path,skiprows=2,usecols=(0,2),unpack=True,delimiter=',')
                # Get p at wvlength by interpolation
                p_wv = interpolate.interp1d(wv_vec,p_vec,kind='linear',
                                            bounds_error=False,fill_value='extrapolate')
                sun_pow = p_wv(wvlgth_nm)
                spectra[i] = sun_pow*wvlgth_nm
                vals[i] = absorb*sun_pow*wvlgth_nm
            # Use Trapezoid rule to perform the integration. Note all the
            # necessary factors of the wavelength have already been included
            # above
            wvlgths = wvlgths[::-1]
            vals = vals[::-1]
            spectra = spectra[::-1]
            #Jsc = intg.simps(Jsc_vals,x=wvlgths,even='avg')
            integrated_absorbtion = intg.trapz(vals,x=wvlgths*1e9)
            power = intg.trapz(spectra,x=wvlgths*1e9)
            # factor of 1/10 to convert A*m^-2 to mA*cm^-2
            #wv_fact = c.e/(c.c*c.h*10)
            #wv_fact = .1
            #Jsc = (Jsc*wv_fact)/power
            frac_absorb = integrated_absorbtion/power
            outf = os.path.join(base,'fractional_absorbtion.dat')
            with open(outf,'w') as out:
                out.write('%f\n'%frac_absorb)
            self.log.info('Fractional Absorbtion = %f'%frac_absorb)
            valuelist.append(frac_absorb)
        return valuelist

    def Jsc(self):
        """Computes photocurrent density. This is just the integrated
        absorbtion scaled by a unitful factor. Assuming perfect carrier
        collection, meaning every incident photon gets converted to 1 collected
        electron, this factor is q/(hbar*c) which converts to a current per
        unit area"""
        valuelist = []
        for group in self.sim_groups:
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info('Computing fractional absorbtion for group at %s'%base)
            vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Assuming the sims have been grouped by frequency, sum over all of them
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf['General']['sim_dir'],'ref_trans_abs.dat')
                with open(dpath,'r') as f:
                    ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
                freq = sim.conf['Simulation']['params']['frequency']['value']
                wvlgth = c.c/freq
                wvlgth_nm = wvlgth*1e9
                freqs[i] = freq
                wvlgths[i] = wvlgth
                # Get solar power from chosen spectrum
                path = sim.conf['Simulation']['input_power_wv']
                wv_vec,p_vec = np.loadtxt(path,skiprows=2,usecols=(0,2),unpack=True,delimiter=',')
                # Get p at wvlength by interpolation
                p_wv = interpolate.interp1d(wv_vec,p_vec,kind='linear',
                                            bounds_error=False,fill_value='extrapolate')
                sun_pow = p_wv(wvlgth_nm)
                spectra[i] = sun_pow*wvlgth_nm
                vals[i] = absorb*sun_pow*wvlgth_nm
            # Use Trapezoid rule to perform the integration. Note all the
            # necessary factors of the wavelength have already been included
            # above
            wvlgths = wvlgths[::-1]
            vals = vals[::-1]
            spectra = spectra[::-1]
            #Jsc = intg.simps(Jsc_vals,x=wvlgths,even='avg')
            integrated_absorbtion = intg.trapz(vals,x=wvlgths)
            # integrated_absorbtion = intg.trapz(vals,x=wvlgths*1e9)
            # factor of 1/10 to convert A*m^-2 to mA*cm^-2
            wv_fact = c.e/(c.c*c.h*10)
            Jsc = wv_fact*integrated_absorbtion
            outf = os.path.join(base,'jsc.dat')
            with open(outf,'w') as out:
                out.write('%f\n'%Jsc)
            self.log.info('Jsc = %f'%Jsc)
            valuelist.append(Jsc)
        return valuelist    
    
    def weighted_transmissionData(self):
        """Computes spectrally weighted absorption,transmission, and reflection"""
        for group in self.sim_groups:
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info('Computing spectrally weighted transmission data for group at %s'%base)
            abs_vals = np.zeros(len(group))
            ref_vals = np.zeros(len(group))
            trans_vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Get solar power from chosen spectrum
            path = group[0].conf['Simulation']['input_power_wv']
            wv_vec,p_vec = np.loadtxt(path,skiprows=2,usecols=(0,2),unpack=True,delimiter=',')
            # Get interpolating function for power
            p_wv = interpolate.interp1d(wv_vec,p_vec,kind='linear',
                                        bounds_error=False,fill_value='extrapolate')
            # Assuming the leaves contain frequency values, sum over all of them
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf['General']['sim_dir'],'ref_trans_abs.dat')
                with open(dpath,'r') as f:
                    ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
                freq = sim.conf['Simulation']['params']['frequency']['value']
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
        super(Plotter,self).__init__(global_conf,sims,sim_groups,failed_sims)
        self.log.debug("This is the plotter")

    def process(self,sim):
        sim.get_data()
        sim_path = os.path.basename(sim.conf['General']['sim_dir'])
        self.log.info('Plotting data for sim %s',sim_path)
        # For each plot
        for plot,data in self.gconf['Postprocessing']['Plotter'].items():
            if data['compute']:
                argsets = data['args']
                self.log.info('Plotting %s with args %s',str(plot),str(argsets))
                if argsets and type(argsets[0]) == list:
                    for argset in argsets:
                        self.log.info('Plotting individual argset'
                                      ' %s',str(argset))
                        if argset:
                            self.gen_plot(plot,sim,argset)
                        else:
                            self.gen_plot(plot,sim,[])
                else:
                    if argsets:
                        self.gen_plot(plot,sim,argsets)
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
            #  self.log.error("Unable to plot the following quantity: %s",
            #                 plot,exc_info=True,stack_info=True)
            self.log.error("Unable to plot the following quantity: %s",
                           plot,exc_info=True)
            raise

    def draw_geometry_2d(self,sim,plane,ax_hand):
        """This function draws the layer boundaries and in-plane geometry on 2D
        heatmaps"""
        # TODO: If we want to make things general, this function could actually
        # get really complicated for several reasons.
        # 1. Each layer can have a unique in plane geometry. If we are plotting
        # an xy plane (plane = z, poor notation which should be fixed), we need
        # to identify which layer we are in so we can extract the geometry
        # 2. We need to be able to handle any in-plane geometry thrown at us
        # 3. If plotting an xz, yz plane (plane = x, plane = y), we need to
        # know which plane we are at so we can scale the cross-sectional width
        # of any geometric features appropriately. This will definitely
        # involved some seriously nontrivial geometry depending on what shape
        # we are dealing with, how its rotated, etc.
        self.log.warning('Drawing capabilities are severely limited until I'
                         ' figure out how to handle a general geometry')
        period = sim.conf['Simulation']['params']['array_period']['value']
        cent = sim.conf['Layers']['NW_AlShell']['geometry']['core']['center']
        core_rad = sim.conf['Layers']['NW_AlShell']['geometry']['core']['radius']
        try:
            shell_rad = sim.conf['Layers']['NW_AlShell']['geometry']['shell']['radius']
        except KeyError:
            shell_rad = False
        dx = period/sim.conf['Simulation']['x_samples']
        dy = period/sim.conf['Simulation']['y_samples']
        max_depth = sim.conf['Simulation']['max_depth']
        if max_depth:
            dz = max_depth/sim.conf['Simulation']['z_samples']
            height = max_depth
        else:
            height = sim.conf.get_height()
            dz = height/sim.conf['Simulation']['z_samples']
        if plane[-1] == 'z':
            self.log.info('draw nanowire circle')
            core = mpatches.Circle((cent['x'],cent['y']),radius=core_rad,fill=False)
            ax_hand.add_artist(core)
            if shell_rad:
                shell = mpatches.Circle((cent['x'],cent['y']),radius=shell_rad,fill=False)
                ax_hand.add_artist(shell)
        elif plane[-1] == 'y' or plane[-1] == 'x':
            boundaries = []
            count = 0
            ordered_layers = sim.conf.sorted_dict(sim.conf['Layers'])
            for layer,ldata in ordered_layers.items():
                # Get boundaries between layers and their starting and ending indices
                layer_t = ldata['params']['thickness']['value']
                if count == 0:
                    start = 0
                    end = int(layer_t/dz)+1
                    boundaries.append((layer_t,start,end,layer))
                else:
                    prev_tup = boundaries[count-1]
                    dist = prev_tup[0]+layer_t
                    start = prev_tup[2]
                    end = int(dist/dz) + 1
                    boundaries.append((dist,start,end))
                if layer_t > 0:
                    x = [0,period]
                    y = [height-start*dz,height-start*dz]
                    label_y = y[0] - 0.15
                    label_x = x[-1] - .01
                    plt.text(label_x,label_y,layer,ha='right',family='sans-serif',size=12)
                    line = mlines.Line2D(x,y,linestyle='solid',linewidth=2.0,color='black')
                    ax_hand.add_line(line)
                    count += 1
                if layer == 'NW_AlShell':
                    if shell_rad:
                        rads = (core_rad, shell_rad)
                    else:
                        rads = (core_rad,)
                    for rad in rads:
                        for x in (cent['x']-rad,cent['x']+rad):
                            # Need two locations w/ same x values
                            xv = [x,x]
                            yv = [height-start*dz,height-end*dz]
                            line = mlines.Line2D(xv,yv,linestyle='solid',linewidth=2.0,color='black')
                            ax_hand.add_line(line)
            return ax_hand

    def heatmap2d(self,sim,x,y,cs,labels,ptype,save_path=None,show=False,draw=False,fixed=None,colorsMap='jet'):
        """A general utility method for plotting a 2D heat map"""
        cm = plt.get_cmap(colorsMap)
        if fixed:
            cNorm = matplotlib.colors.Normalize(vmin=np.amin(5.0), vmax=np.amax(100.0))
        else:
            cNorm = matplotlib.colors.Normalize(vmin=np.amin(cs), vmax=np.amax(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111)
        #  ax.pcolormesh(x, y, cs,cmap=cm,norm=cNorm,alpha=.5,linewidth=0)
        #  ax.pcolor(x, y,
        #          cs,cmap=cm,norm=cNorm,alpha=.5,linewidth=0,edgecolors='none')
        ax.imshow(cs,cmap=cm,norm=cNorm,extent=[x.min(),x.max(),y.min(),y.max()],aspect='auto')
        #  ax.matshow(cs,cmap=cm,norm=cNorm)
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[2])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start,end,0.1))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start,end,0.2))
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_ylim((np.amin(y),np.amax(y)))
        fig.suptitle(labels[3])
        if draw:
            ax = self.draw_geometry_2d(sim,ptype,ax)
        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def plane_2d(self,sim,quantity,plane,pval,draw=False,fixed=None):
        """Plots a heatmap of a fixed 2D plane"""
        zs = sim.conf['Simulation']['z_samples']
        xs = sim.conf['Simulation']['x_samples']
        xs = int(float(xs))
        ys = sim.conf['Simulation']['y_samples']
        ys = int(float(ys))
        max_depth = sim.conf['Simulation']['max_depth']
        if max_depth:
            self.log.info('Plotting to max depth of {}'.format(max_depth))
            height = max_depth
        else:
            height = sim.conf.get_height()
        pval = int(pval)
        period = sim.conf['Simulation']['params']['array_period']['value']
        dx = period/xs
        dy = period/ys
        dz = height/zs
        x = np.arange(0,period,dx)
        y = np.arange(0,period,dy)
        z = np.arange(0,height+dz,dz)
        # Maps planes to an integer for extracting data
        # plane_table = {'x': 0,'y': 1,'z':2}
        # Get the scalar values
        self.log.info('Retrieving scalar %s'%quantity)
        scalar = sim.get_scalar_quantity(quantity)
        self.log.info('DATA SHAPE: %s'%str(scalar.shape))
        ## Filter out any undesired data that isn't on the planes
        #mat = np.column_stack((sim.pos_inds[:,0],sim.pos_inds[:,1],sim.pos_inds[:,2],scalar))
        #planes = np.array([row for row in mat if row[plane_table[plane]] == pval])
        #self.log.debug("Planes shape: %s"%str(planes.shape))
        ## Get all unique values for x,y,z and convert them to actual values not indices
        #x,y,z = np.unique(planes[:,0])*dx,np.unique(planes[:,1])*dy,np.unique(planes[:,2])
        freq = sim.conf['Simulation']['params']['frequency']['value']
        wvlgth = (c.c/freq)*1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(freq,wvlgth)
        # Get the plane we wish to plot
        self.log.info('Retrieving plane ...')
        cs = self.get_plane(scalar,xs,ys,zs,plane,pval)
        self.log.info('Plotting plane')
        show = sim.conf['General']['show_plots']
        if plane == 'x':
            labels = ('y [um]','z [um]', quantity,title)
            if sim.conf['General']['save_plots']:
                p = os.path.join(sim.conf['General']['sim_dir'],
                                 '%s_plane_2d_x.pdf'%quantity)
            else:
                p = False
            self.heatmap2d(sim,y,z,cs,labels,plane,save_path=p,show=show,draw=draw,fixed=fixed)
        elif plane == 'y':
            labels = ('x [um]','z [um]', quantity,title)
            if sim.conf['General']['save_plots']:
                p = os.path.join(sim.conf['General']['sim_dir'],
                                 '%s_plane_2d_y.pdf'%quantity)
            else:
                p = False
            self.heatmap2d(sim,x,z,cs,labels,plane,save_path=p,show=show,draw=draw,fixed=fixed)
        elif plane == 'z':
            labels = ('y [um]','x [um]', quantity,title)
            if sim.conf['General']['save_plots']:
                p = os.path.join(sim.conf['General']['sim_dir'],
                                 '%s_plane_2d_z.pdf'%quantity)
            else:
                p = False
            self.heatmap2d(sim,x,y,cs,labels,plane,save_path=p,show=show,draw=draw,fixed=fixed)

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
        fig.suptitle(os.path.basename(sim.conf['General']['sim_dir']))
        if sim.conf['General']['save_plots']:
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(sim.conf['General']['sim_dir'],name)
            fig.savefig(path)
        if sim.conf['General']['show_plots']:
            plt.show()
        plt.close(fig)

    def full_3d(self,sim,quantity):
        """Generates a full 3D plot of a specified scalar quantity"""
        period = sim.conf['Simulation']['params']['array_period']['value']
        dx = period/sim.conf['Simulation']['x_samples']
        dy = period/sim.conf['Simulation']['y_samples']
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
        period = sim.conf['Simulation']['params']['array_period']['value']
        dx = period/sim.conf['Simulation']['x_samples']
        dy = period/sim.conf['Simulation']['y_samples']
        # Get the scalar values
        scalar = sim.get_scalar_quantity(quantity)
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((sim.pos_inds[:,0],sim.pos_inds[:,1],sim.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if round(row[0]) == xplane or
                          round(row[1]) == yplane])
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
        if sim.conf['General']['save_plots']:
            name = labels[1]+'_'+ptype+'.pdf'
            path = os.path.join(sim.conf['General']['sim_dir'],name)
            fig.savefig(path)
        if sim.conf['General']['show_plots']:
            plt.show()
        plt.close(fig)

    def fixed_line(self,sim,quantity,direction,coord1,coord2):
        """Plot a scalar quantity on a line along a the z direction at some pair of
        coordinates in the plane perpendicular to that direction"""
        coord1 = int(coord1)
        coord2 = int(coord2)
        period = sim.conf['Simulation']['params']['array_period']['value']
        xsamp = sim.conf['Simulation']['x_samples']
        ysamp = sim.conf['Simulation']['y_samples']
        zsamp = sim.conf['Simulation']['z_samples']
        dx = period/xsamp
        dy = period/ysamp
        dz = period/zsamp
        # Get the scalar values
        scalar = sim.get_scalar_quantity(quantity)
        # Filter out any undesired data that isn't on the planes
        data = self.get_line(scalar,xsamp,ysamp,zsamp,direction,coord1,coord2)
        if direction == 'x':
            # z along rows, y along columns
            pos_data = np.unique(sim.pos_inds[:,0]) * dx
        elif direction == 'y':
            # x along columns, z along rows
            pos_data = np.unique(sim.pos_inds[:,1]) * dy
        elif direction == 'z':
            # x along rows, y along columns
            pos_data = np.unique(sim.pos_inds[:,2]) * dz
        #mat = np.column_stack((sim.pos_inds[:,0],sim.pos_inds[:,1],sim.pos_inds[:,2],scalar))
        #planes = np.array([row for row in mat if row[0] == coord1 and row[1] == coord2])
        #planes[:,0] = planes[:,0]*dx
        #planes[:,1] = planes[:,1]*dy
        freq = sim.conf['Simulation']['params']['frequency']['value']
        wvlgth = (c.c/freq)*1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(freq,wvlgth)
        labels = ('Z [um]',quantity,title)
        ptype = "%s_line_plot_%i_%i"%(direction,coord1,coord2)
        self.line_plot(sim,pos_data,data,ptype,labels)

class Global_Plotter(Plotter):
    """Plots global quantities for an entire run that are not specific to a single simulation"""
    def __init__(self,global_conf,sims=[],sim_groups=[],failed_sims=[]):
        super(Global_Plotter,self).__init__(global_conf,sims,sim_groups,failed_sims)
        self.log.debug("Global plotter init")

    def process_all(self):
        self.log.info('Beginning global plotter method ...')
        for plot,data in self.gconf['Postprocessing']['Global_Plotter'].items():
            if data['compute']:
                argsets = data['args']
                self.log.info('Plotting %s with args %s',str(plot),str(argsets))
                if argsets and type(argsets[0]) == list:
                    for argset in argsets:
                        self.log.info('Plotting individual argset'
                                      ' %s',str(argset))
                        if argset:
                            self.gen_plot(plot,argset)
                        else:
                            self.gen_plot(plot,[])
                else:
                    if argsets:
                        self.gen_plot(plot,argsets)
                    else:
                        self.gen_plot(plot,[])

    def gen_plot(self,plot,args):
        try:
            getattr(self,plot)(*args)
        except KeyError:
            self.log.error("Unable to plot the following quantity: %s",
                           plot,exc_info=True,stack_info=True)
            raise

    def convergence(self,quantity,err_type='global',scale='linear'):
        """Plots the convergence of a field across all available simulations"""
        self.log.info('Plotting convergence')
        for group in self.sim_groups:
            base = group[0].conf['General']['base_dir']
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
                fig = plt.figure(figsize=(9,7))
                plt.ylabel('M.S.E of %s'%quantity)
                plt.xlabel('Number of Fourier Terms')
                plt.plot(labels,errors,linestyle='-',marker='o',color='b')
                plt.yscale(scale)
                #plt.xticks(x,labels,rotation='vertical')
                plt.tight_layout()
                plt.title(os.path.basename(base))
                if self.gconf['General']['save_plots']:
                    if '_excluded' in path:
                        excluded = '_excluded'
                    else:
                        excluded = ''
                    name = '%s_%sconvergence_%s%s.pdf'%(os.path.basename(base),err_type,quantity,excluded)
                    path = os.path.join(base,name)
                    fig.savefig(path)
                if self.gconf['General']['show_plots']:
                    plt.show()
                plt.close(fig)

    def scalar_reduce(self,quantity,plane,pval,draw=False,fixed=None):
        """Plot the result of a particular scalar reduction for each group"""
        for group in self.sim_groups:
            sim = group[0]
            base = sim.conf['General']['base_dir']
            self.log.info('Plotting scalar reduction of %s for quantity'
                          ' %s'%(base,quantity))
            cm = plt.get_cmap('jet')
            zs = sim.conf['Simulation']['z_samples']
            xs = sim.conf['Simulation']['x_samples']
            xs = int(float(xs))
            ys = sim.conf['Simulation']['y_samples']
            ys = int(float(ys))
            height = sim.conf.get_height()
            period = sim.conf['Simulation']['params']['array_period']['value']
            dx = period/xs
            dy = period/ys
            dz = height/zs
            x = np.arange(0,period,dx)
            y = np.arange(0,period,dy)
            z = np.arange(0,height+dz,dz)
            if sim.conf['General']['save_as'] == 'npz':
                globstr = os.path.join(base,'scalar_reduce*_%s.npy'%quantity)
                files = glob.glob(globstr)
            elif sim.conf['General']['save_as'] == 'text':
                globstr = os.path.join(base,'scalar_reduce*_%s.crnch'%quantity)
                files = glob.glob(globstr)
            else:
                raise ValueError('Incorrect file type in config')
            title = 'Reduction of %s'%quantity
            for datfile in files:
                if sim.conf['General']['save_as'] == 'npz':
                    scalar = np.load(datfile)
                elif sim.conf['General']['save_as'] == 'text':
                    scalar = np.loadtxt(datfile,group_comb)
                else:
                    raise ValueError('Incorrect file type in config')
                cs = self.get_plane(scalar,xs,ys,zs,plane,pval)
                if plane == 'x':
                    labels = ('y [um]','z [um]', quantity,title)
                    if sim.conf['General']['save_plots']:
                        fname = 'scalar_reduce_%s_plane_2d_x.pdf'%quantity
                        p = os.path.join(sim.conf['General']['base_dir'],fname)
                    else:
                        p = False
                    show = sim.conf['General']['show_plots']
                    self.heatmap2d(sim,y,z,cs,labels,plane,save_path=p,show=show,draw=draw,fixed=fixed)
                elif plane == 'y':
                    labels = ('x [um]','z [um]', quantity,title)
                    if sim.conf['General']['save_plots']:
                        fname = 'scalar_reduce_%s_plane_2d_y.pdf'%quantity
                        p = os.path.join(sim.conf['General']['base_dir'],fname)
                    else:
                        p = False
                    show = sim.conf['General']['show_plots']
                    self.heatmap2d(sim,x,z,cs,labels,plane,save_path=p,show=show,draw=draw,fixed=fixed)
                elif plane == 'z':
                    labels = ('y [um]','x [um]', quantity,title)
                    if sim.conf['General']['save_plots']:
                        fname = 'scalar_reduce_%s_plane_2d_z.pdf'%quantity
                        p = os.path.join(sim.conf['General']['base_dir'],fname)
                    else:
                        p = False
                    self.heatmap2d(sim,x,y,cs,labels,plane,save_path=p,show=show,draw=draw,fixed=fixed)

    def transmission_data(self,absorbance,reflectance,transmission):
        """Plot transmissions, absorption, and reflectance assuming leaves are frequency"""
        for group in self.sim_groups:
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info('Plotting transmission data for group at %s'%base)
            # Assuming the leaves contain frequency values, sum over all of them
            freqs = np.zeros(len(group))
            refl_l = np.zeros(len(group))
            trans_l = np.zeros(len(group))
            absorb_l = np.zeros(len(group))
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf['General']['sim_dir'],'ref_trans_abs.dat')
                with open(dpath,'r') as f:
                    ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
                freq = sim.conf['Simulation']['params']['frequency']['value']
                freqs[i] = freq
                trans_l[i] = trans
                refl_l[i] = ref
                absorb_l[i] = absorb
            freqs = (c.c/freqs[::-1])*1e9
            refl_l = refl_l[::-1]
            absorb_l = absorb_l[::-1]
            trans_l = trans_l[::-1]
            plt.figure()
            if absorbance:
                self.log.info('Plotting absorbance')
                plt.plot(freqs, absorb_l, '-o', label='Absorption')
            if reflectance:
                plt.plot(freqs, refl_l, '-o', label='Reflectance')
            if transmission:
                plt.plot(freqs, trans_l, '-o', label='Transmission')
            plt.legend(loc='best')
            figp = os.path.join(base, 'transmission_plots.pdf')
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
    parser.add_argument('-gb','--group_by',help="""The parameter you
            would like to group simulations by, specified as a dot separated path 
            to the key in the config as: path.to.key.value""")
    parser.add_argument('-ga','--group_against',help="""The parameter
            you would like to group against, specified as a dot separated path 
            to the key in the config as: path.to.key.value""")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = Config(path=os.path.abspath(args.config_file))
    else:
        raise ValueError("The file you specified does not exist!")

    if not (args.group_by or args.group_against):
        raise ValueError('Need to group sims somehow. A sensible value would be'
                         ' by/against frequency')
    else:
        if args.group_by:
            group_by = args.group_by.split('.')
        else:
            group_ag = args.group_against.split('.')

    # Configure logger
    lfile = os.path.join(conf['General']['base_dir'],'logs/postprocess.log')
    logger = configure_logger(level=args.log_level,name='postprocess',
                              console=True,logfile=lfile)
                                

    # Collect the sims once up here and reuse them later
    proc = Processor(conf)
    sims, failed_sims = proc.collect_sims()
    # First we need to group against if specified. Grouping against corresponds
    # to "leaves" in the tree
    if args.group_against:
        sim_groups = proc.group_against(group_ag, conf.variable)
    # Next we group by. This corresponds to building the parent nodes for each
    # set of leaf groups
    if args.group_by:
        sim_groups = proc.group_by(group_by)
    # print(len(sim_groups))
    # print(group_ag)
    # print(conf.variable)
    # for group in proc.sim_groups:
    #     sim = group[0]
    #     try:
    #         os.makedirs(sim.conf['General']['results_dir'])
    #     except OSError:
    #         pass
    # quit()
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
        crunchr.process_all()
        # for sim in crunchr.sims:
        #     crunchr.transmissionData(sim)
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
