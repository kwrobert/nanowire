import shutil
import subprocess
import itertools
import argparse as ap
import os
import glob
import logging
import datetime
from collections import OrderedDict 
from profilehooks import profile
import multiprocessing as mp 
import pandas
import numpy as np
import scipy.optimize as optz
import postprocess as pp
import time
import yaml
import pprint
from copy import deepcopy 

class Config(object):

    def __init__(self,path=None,data=None):
        if path:
            self.data = self._parse_file(path)
        elif data:
            self.data = data
        else:
            raise ValueError('You must supply either a path to a YAML file or a python'
            ' dictionary')
        self._store_params()

    def _parse_file(self,path):
        """Parse the YAML file provided at the command line"""
         
        with open(path,'r') as cfile:
            text = cfile.read()
        conf = yaml.load(text)
        return conf

    def _store_params(self):
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
                    self.fixed.append(('Layer',layer,'params',par))
                elif data['type'] == 'variable':
                    self.variable.append(('Layer',layer,'params',par))
                elif data['type'] == 'sorting':
                    self.sorting.append(('Layer',layer,'params',par))
                elif data['type'] == 'evaluated':
                    self.evaluated.append(('Layer',layer,'params',par))
                elif data['type'] == 'optimized':
                    self.optimized.append(('Layer',layer,'params',par))
                else:
                    loc = '.'.join('Layer',layer,'params',par)
                    raise ValueError('Specified an invalid config type at {}'.format(loc))
    
    def get(self,keyset):
        """A convenience method to get the section of the config file located at the end
        of the sequence of keys"""
        section = self.data
        print(keyset)
        for key in keyset:
            section = section[key]
        return section

    def set(self,keyset,value):
        """A convenience method to set the a value in the config given a sequence of keys"""
        #loc = self.get(keyset[:-1])
        #print(loc)
        self.get(keyset[:-1])[keyset[-1]] = value
        #section = self.data
        #for key in keyset[:-1]:

    def copy(self):
        """Returns a copy of the current config object"""
        return deepcopy(self)

    def write(self,path):
        """Dumps this config object to its YAML representation given a path to a file"""
        with open(path,'w') as out:
            out.write(yaml.dump(self.data,default_flow_style=False))

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
    """Parse the YAML file provided at the command line"""
     
    with open(path,'r') as cfile:
        text = cfile.read()
    conf = yaml.load(text)
    return conf

def get_combos(conf):
    """Given a config object, return two lists. The first list contains the
    names of all the variable parameters in the config object. The second is a
    list of lists, where the inner list contains all the unique combinations of
    this config object's non-fixed parameters. The elements of the inner list
    of value correspond to the elements of the key list"""


    log = logging.getLogger('sim_wrapper') 
    log.info("Constructing dictionary of options and their values ...")
    # Get the list of values from all our variable keysets
    optionValues = OrderedDict()
    for keyset in conf.variable:
        par = keyset[-1]
        pdict = conf.get(keyset)
        print(pdict)
        if pdict['itertype'] == 'numsteps':
            values = np.linspace(pdict['start'],pdict['end'],pdict['step'])
        elif pdict['itertype'] == 'stepsize':
            value = np.arange(pdict['start'],pdict['end']+pdict['step'],pdict['step'])
        else:
            raise ValueError('Invalid itertype specified at {}'.format(str(keyset)))
        optionValues[par] = values
    log.debug("Option values dict after processing: %s"%str(optionValues))
    valuelist = list(optionValues.values())
    keys = list(optionValues.keys())
    # Consuming a list of lists/tuples where each inner list/tuple contains all the values
    # for a particular parameter, returns a list of tuples containing all the unique
    # combos for that set of parameters 
    combos=list(itertools.product(*valuelist))
    log.debug('The list of parameter combos: %s',str(combos))
    return keys,combos 

def make_leaves(nodes):
    """Accepts a list of tuples, where the first element of the tuple is the full path to
    the node under which the leaves will be created, and the second element is
    un-finalized config object for that node"""


    log = logging.getLogger('sim_wrapper') 
    log.info("Building all leaves for each node...")
    # Get a list of the parameters names (keys) and a list of lists of values where the
    # ith value of the inner lists corresponds to the ith param name in keys. The inner
    # list consists of a unique combination of variable parameters. Note all nodes sweep
    # through the same parameters so we only need to compute this once. 
    keys,combos = get_combos(nodes[0][1])
    # Build all the leaves at every node in the directory tree 
    leaves = []
    for nodepath,conf in nodes:
        # Loop through each unique combination of parameters
        for combo in combos:
            # Make a unique working directory based on parameter combo
            workdir=''
            for i in range(len(combo)):
                substr = '{}_{:G}__'.format(keys[i],combo[i])
                workdir += substr 
            workdir=workdir.rstrip('__')
            fullpath = os.path.join(nodepath,workdir)
            try:
                os.makedirs(fullpath)
            except OSError:
                log.info("Working directory already exists, appending new to name")
                workdir = workdir+"__new"
                fullpath = os.path.join(nodepath,workdir)
                os.makedirs(fullpath)
            log.info('Created leaf %s',fullpath)
            # Write new config file to sim dir. We make a copy then remove the postprocessing
            # section
            sim_conf = conf.copy()
            del sim_conf.data['Postprocessing']
            sim_conf.data['General']['sim_dir'] = fullpath
            # Now we just overwrite all the variable parameters with their new
            # fixed values. Note that itertools.product is so wonderful and
            # nice that it preserves the order of the values in every combo
            # such that the combo values always line up with the proper
            # parameter name
            for i in range(len(combo)):
                sim_conf.set(conf.variable[i],{'type':'fixed','value':float(combo[i])})
            # Now that all the parameters have been defined, we can add the evaluated parameters that
            # depend on the values of some other parameters
            #for par, val in conf.items('Evaluated Parameters'):
            #    # Wrap the value in back ticks so we know to evaluate the results of the interpolated
            #    # expression in the simulation script
            #    wrapped = '`'+str(val)+'`'
            #    sim_conf.set('Parameters',par,wrapped)
            # Now write the config file to the appropriate subdir
            out = os.path.join(fullpath,'sim_conf.yml') 
            sim_conf.write(out)
            base_script = sim_conf.data['General']['sim_script']
            script = os.path.join(nodepath,os.path.basename(base_script))
            if not os.path.exists(script):
                shutil.copy(base_script,script)
            leaves.append((fullpath,sim_conf))
    log.debug('Here are the leaves: %s',str(leaves))
    log.info('Finished building leaves!')
    return leaves

def make_single_sim(conf):
    """Create a configuration file for a single simulation, such that it is ready to be
    consumed by the Lua script. Return the path to the config file and the config object
    itself"""

    log = logging.getLogger('sim_wrapper')
    log.info("Running single sim")
    # Make the simulation dir
    basedir = conf.data['General']['base_dir']
    # The dir is already made when the logger is configured but this is a safety measure i guess?
    try:
        path = os.path.join(basedir,'data')
        os.makedirs(path)
    except OSError:
        log.info('Data directory already exists, appending timestamp')
        stamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        path = os.path.join(basedir,'data__'+stamp)
        os.makedirs(path)
    # Write new config file to sim dir. We make a copy then remove the postprocessing
    # section
    sim_conf = conf.copy()
    del sim_conf.data['Postprocessing']
    sim_conf.data['General']['sim_dir'] = path
    # Now write the config file to the data subdir
    out = os.path.join(path,'sim_conf.yml') 
    sim_conf.write(out)
    # Copy sim script to sim dir
    base_script = sim_conf.data['General']['sim_script']
    script = os.path.join(path,os.path.basename(base_script))
    shutil.copy(base_script,script)
    return (path,sim_conf)

def convert_data(path,conf):
    """Converts text file spit out by S4 into npz format for faster loading during postprocessing"""
    # Have we decided to ignore and remove H field files?
    ignore = conf.data['General']['ignore_h']
    # Convert e field data files
    bname = conf.data['General']['base_name']
    efile = os.path.join(path,bname+'.E')
    d = pandas.read_csv(efile,delim_whitespace=True,header=None,skip_blank_lines=True)
    edata = d.as_matrix()
    econv = efile+'.raw'
    np.savez(econv,data=edata,headers=[None])
    hfile = os.path.join(path,bname+'.H')
    if not ignore:
        # Convert h field files
        d = pandas.read_csv(hfile,delim_whitespace=True,header=None,skip_blank_lines=True)
        hdata = d.as_matrix()
        hconv = hfile+'.raw'
        np.savez(hconv,data=hdata,headers=[None])
    # Remove the old text files
    os.remove(efile)
    os.remove(hfile)
    return None

def run_sim(jobtup):
    """Actually runs simulation in a given directory using subprocess.call. Expects a tuple
    containing the absolute path to the job directory as the first element and
    the configuration object for the job as the second element"""

    log = logging.getLogger('sim_wrapper') 
    jobpath,jobconf = jobtup
    timed = jobconf.data['General']['save_time']
    tout = os.path.join(jobpath,'timing.dat')
    simlog = os.path.join(jobpath,'sim.log')
    script = jobconf.data['General']['sim_script']
    ini_file = os.path.join(jobpath,'sim_conf.ini')
    if timed:
        log.debug('Executing script with timing wrapper ...')
        cmd = 'command time -vv -o %s lua %s %s 2>&1 | tee %s'%(tout,script,ini_file,simlog)
        #cmd = '/usr/bin/perf stat -o timing.dat -r 5 /usr/bin/lua %s %s'%(script,ini_file)
    else:
        cmd = 'command lua %s %s 2>&1 | tee %s'%(script,ini_file,simlog)
    log.debug('Subprocess command: %s',cmd)
    relpath = os.path.relpath(jobpath,jobconf.data['General']['treebase'])
    log.info("Starting simulation for %s ...",relpath)
    completed = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #log.info('Simulation stderr: %s',completed.stderr)
    log.info("Simulation stderr: %s",completed.stderr)
    log.debug('Simulation stdout: %s',completed.stdout)
    # Convert text data files to npz binary file
    if jobconf.data['General']['save_as'] == 'npz':
        log.info('Converting data at %s to npz format',relpath)
        convert_data(jobpath,jobconf)
    log.info("Finished simulation for %s!",relpath)
    return None

def execute_jobs(gconf,jobs):

    """Given a list of length 2 tuples containing job directories and their
    configuration objects (in that order), executes the jobs. If specified, a
    multiprocessing pool is used to run the jobs in parallel. If not, jobs are
    executed serially"""
    
    log=logging.getLogger('sim_wrapper')
    if not gconf.data['General']['parallel']:
        log.info('Executing jobs serially')
        for job in jobs:
            run_sim(job)
    else:
        num_procs = mp.cpu_count() - gconf.data['General']['reserved_cores']
        log.info('Executing jobs in parallel using %s cores ...',str(num_procs))
        with mp.Pool(processes=num_procs) as pool:
            result = pool.map(run_sim,jobs)

def run(conf,log):
    """The main run methods that decides what kind of simulation to run based on the
    provided config object"""

    basedir = conf.data['General']['base_dir']
    logdir = os.path.join(basedir,'logs')
    # Configure logger
    logger = configure_logger(log,'sim_wrapper',logdir,'sim_wrapper.log')
    # Just a simple single simulation
    if conf.fixed and not conf.variable and not conf.sorting and not conf.optimized:
        logger.debug('Entering single sim function from main')
        job = make_single_sim(conf)
        #run_sim(job)
    # If we have variable params, do a parameter sweep
    elif conf.variable and not conf.optimized:
        # If we have sorting parameters we need to make the nodes first
        if conf.sorting:
            logger.debug('Creating nodes and leaves')
            jobs = make_nodes(conf)
        else:
            logger.debug('Creating leaves')
            # make_leaves expects a list of tuples with contents (jobpath,jobconfobject)
            jobs = make_leaves([(basedir,conf)])
        execute_jobs(conf,jobs)
    # Looks like we need to run an optimization
    elif conf.optimized:
        logger.debug('Entering optimization function from main')
        run_optimization(conf)
    else:
        logger.error('Unsupported configuration for a simulation run. Not a single sim, sweep, or \
        optimization')

def main():

    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('--log_level',type=str,default='info',choices=['debug','info','warning','error','critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        conf = Config(path=os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    #pre_check(os.path.abspath(args.config_file),conf)
    run(conf,args.log_level)

if __name__ == '__main__':
    main()
