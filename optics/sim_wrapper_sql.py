import shutil
import subprocess
import itertools
import os
import glob
import datetime
import copy
import hashlib
import multiprocessing as mp
import pandas
import numpy as np
import scipy.optimize as optz
#  import postprocess as pp
import time
import pprint
import argparse as ap
import ruamel.yaml as yaml
import logging

# get our custom config object and the logger function
#from utils.simulation import *
from utils.config import Config
from utils.simulator import Simulator
from collections import MutableMapping,OrderedDict
from functools import wraps
from contextlib import contextmanager
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#__DB_ENGINE__ = None
#__SESSION_FACTORY__ = None
#Base = declarative_base()
#
#def setup_db(name,verbose=False):
#    """Sets up the module-scope database engine and the session factory"""
#    global __DB_ENGINE__
#    global __SESSION_FACTORY__
#    if not __DB_ENGINE__ and not __SESSION_FACTORY__:
#        __DB_ENGINE__ = create_engine('sqlite:///{}'.format(name),echo=verbose)
#        __SESSION_FACTORY__ = sessionmaker(bind=__DB_ENGINE__)
#        Base.metadata.create_all(__DB_ENGINE__)
#        Simulation.metadata.create_all(__DB_ENGINE__)
#    else:
#        raise RuntimeError('DB Engine already set')
#
#@contextmanager
#def session_scope():
#    """Provide a transactional scope around a series of operations."""
#    session = __SESSION_FACTORY__()
#    try:
#        yield session
#        session.commit()
#    except:
#        session.rollback()
#        raise
#    finally:
#        session.close()

#def dbconnect(func):
#    @wraps(func)
#    def wrapper(*args,**kwargs):
#        session = __SESSION_FACTORY__() 
#        print('Setting up session scope')
#        print('Here is session in wrapper')
#        print(session)
#        return func(*args,**kwargs)
#    return wrapper

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

def parse_file(path):
    with open(path,'r') as cfile:
        text = cfile.read()
    conf = yaml.load(text,Loader=yaml.Loader)
    return conf

def get_combos(conf,keysets):
    """Given a config object, return two lists. The first list contains the
    names of all the variable parameters in the config object. The second is a
    list of lists, where the inner list contains all the unique combinations of
    this config object's non-fixed parameters. The elements of the inner list
    of value correspond to the elements of the key list"""

    log = logging.getLogger('sim_wrapper')
    log.info("Constructing dictionary of options and their values ...")
    # Get the list of values from all our variable keysets
    optionValues = OrderedDict()
    for keyset in keysets:
        par = keyset[-1]
        pdict = conf[keyset]
        if pdict['itertype'] == 'numsteps':
            values = np.linspace(pdict['start'],pdict['end'],pdict['step'])
        elif pdict['itertype'] == 'stepsize':
            values = np.arange(pdict['start'],pdict['end']+pdict['step'],pdict['step'])
        else:
            raise ValueError('Invalid itertype specified at {}'.format(str(keyset)))
        optionValues[par] = values
    log.debug("Option values dict after processing: %s"%str(optionValues))
    valuelist = list(optionValues.values())
    keys = list(optionValues.keys())
    # Consuming a list of lists/tuples where each inner list/tuple contains all the values
    # for a particular parameter, returns a list of tuples containing all the unique
    # combos for that set of parameters
    combos = list(itertools.product(*valuelist))
    log.debug('The list of parameter combos: %s',str(combos))
    return keys,combos

def make_sims(global_conf):
    """Make all the individual simulations for each parameter combination"""
   
    locs,combos = get_combos(global_conf,global_conf.variable)
    sims = []
    for combo in combos:
        # Make a copy of the global config for this parameter combos. This copy
        # represents an individual simulation
        sim_conf = global_conf.copy()
        del sim_conf['Postprocessing']
        # Now we just overwrite all the variable parameters with their new
        # fixed values. Note that itertools.product is so wonderful and
        # nice that it preserves the order of the values in every combo
        # such that the combo values always line up with the proper
        # parameter name
        for i in range(len(combo)):
            sim_conf[global_conf.variable[i]] = {'type':'fixed','value':float(combo[i])}
        sims.append(sim_conf)
    return sims
        
#      with session_scope() as sess:
#          log = logging.getLogger('sim_wrapper')
#          log.info("Running single sim")
#          del sim['Postprocessing']
#          # Instantiate simulation object
#          #sim = Simulation(sim_conf)
#          path = os.path.join(sim['General']['base_dir'],sim.id[0:10])
#          print(sim.id)
#          sim['General']['sim_dir'] = path
#          print(sim.id)
#          sess.add(sim)
#          # The dir is already made when the logger is configured but this is a safety measure i guess?
#          try:
#              os.makedirs(path)
#          except OSError:
#              log.info('Data directory already exists, appending timestamp')
#              stamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
#              path = path+'__{}'.format(stamp)
#              os.makedirs(path)
#          # Now write the config file to the data subdir
#          out = os.path.join(path,'sim_conf.yml')
#          sim.write(out)
#          # Copy sim script to sim dir
#          base_script = sim['General']['sim_script']
#          script = os.path.join(path,os.path.basename(base_script))
#          shutil.copy(base_script,script)
    #  return (path,sim)

def run_sim(sim_conf):
    """Actually runs simulation in a given directory using subprocess.call. Expects a tuple
    containing the absolute path to the job directory as the first element and
    the configuration object for the job as the second element"""
    log = logging.getLogger('sim_wrapper')
    if not sim_conf.variable_thickness:
        sim = Simulator(sim_conf)
        log.info('Executing sim %s'%sim.id[0:10])
        os.makedirs(sim.dir)
        sim.conf.write(os.path.join(sim.dir,'sim_conf.yml'))
        sim.configure()
        sim.build_device()
        sim.set_excitation()
        sim.get_fields()
        sim.get_fluxes()
    else:
        raise NotImplementedError('Need to figure out thicknesses')
    return
    #  with session_scope() as session:
        #  timed = sim['General']['save_time']
        #  tout = os.path.join(jobpath,'timing.dat')
        #  simlog = os.path.join(jobpath,'sim.log')
        #  script = sim['General']['sim_script']
        #  ini_file = os.path.join(jobpath,'sim_conf.yml')
        #  if timed:
            #  log.debug('Executing script with timing wrapper ...')
            #  cmd = 'command time -vv -o %s lua %s %s 2>&1 | tee %s'%(tout,script,ini_file,simlog)
            #  #cmd = '/usr/bin/perf stat -o timing.dat -r 5 /usr/bin/lua %s %s'%(script,ini_file)
        #  else:
            #  cmd = 'command lua %s %s 2>&1 | tee %s'%(script,ini_file,simlog)
        #  log.debug('Subprocess command: %s',cmd)
        #  relpath = os.path.relpath(jobpath,sim['General']['treebase'])
        #  log.info("Starting simulation for %s ...",relpath)
        #  completed = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #  #log.info('Simulation stderr: %s',completed.stderr)
        #  log.info("Simulation stderr: %s",completed.stderr)
        #  log.debug('Simulation stdout: %s',completed.stdout)
        #  # Convert text data files to npz binary file
        #  if sim['General']['save_as'] == 'npz':
            #  log.info('Converting data at %s to npz format',relpath)
            #  convert_data(jobpath,sim)
        #  log.info("Finished simulation for %s!",relpath)
    #  return None

def execute_jobs(gconf,sims):
    """Given a list of simulation objects, run them either serially or in
    parallel"""

    log=logging.getLogger('sim_wrapper')
    if not gconf['General']['parallel']:
        log.info('Executing sims serially')
        for sim in sims:
            run_sim(sim)
              
    else:
        num_procs = mp.cpu_count() - gconf['General']['reserved_cores']
        log.info('Executing sims in parallel using %s cores ...',str(num_procs))
        #  with mp.Pool(processes=num_procs) as pool:
            #  result = pool.map(run_sim,sims)
        pool = mp.Pool(processes=num_procs)
        res = pool.map(run_sim,sims)
        pool.close()

def run(conf,log):
    """The main run methods that decides what kind of simulation to run based on the
    provided config object"""

    basedir = conf['General']['base_dir']
    logdir = os.path.join(basedir,'logs')
    # Configure logger
    logger = configure_logger(log,'sim_wrapper',logdir,'sim_wrapper.log')
    # Get all the sims
    sims = make_sims(conf)
    # Just a simple single simulation
    if not conf.optimized:
        logger.info("Executing job campaign")
        execute_jobs(conf,sims)
    # If we have variable params, do a parameter sweep
    elif conf.optimized:
        raise NotImplementedError('Havent figured out optimizations yet')
    else:
        logger.error('Unsupported configuration for a simulation run. Not a '
        'single sim, sweep, or optimization. Make sure your sweeps are '
        'configured correctly, and if you are running an optimization '
        'make sure you do not have any sorting parameters specified')

def pre_check(conf_path,conf):
    """Checks conf file for invalid values"""
    if not os.path.isfile(conf['General']['sim_script']):
        print('You need to change the sim_script entry in the [General] section of your config \
        file')
        quit()

    # Warn user if they are about to dump a bunch of simulation data and directories into a
    # directory that already exists
    base = conf["General"]["base_dir"]
    if os.path.isdir(base):
        print('WARNING!!! You are about to start a simulation in a directory that already exists')
        print('WARNING!!! This will dump a whole bunch of crap into that directory and possibly')
        print('WARNING!!! overwrite old simulation data.')
        ans = input('Would you like to continue? CTRL-C to exit, any other key to continue: ')
    # Copy provided config file to basedir
    try:
        os.makedirs(base)
    except OSError:
        pass
    fname = os.path.basename(conf_path)
    new_path = os.path.join(base,fname)
    try:
        shutil.copy(conf_path,new_path)
    except shutil.SameFileError:
        pass
    # TODO: Add checks between specified x,y,z samples and plane vals in plotting section

def main():

    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('--log_level',type=str,default='info',choices=['debug','info','warning','error','critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        global_conf = Config(path=os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()
    
    #setup_db(sim['General']['db_name'])
    ##pre_check(os.path.abspath(args.config_file),conf)
    run(global_conf,args.log_level)

#def db_test():
#    with session_scope() as session:
#        print(session)
#
#def main():
#    setup_db('new_test.db')
#    db_test()
#    #Session = sessionmaker(bind=engine)
#    #session = Session()
#    #conf = {'some_stuff':'wtih a value'}
#    #sim = Simulation(conf)
#    #print(sim.__table__)
#    #print(sim.id)
#    #session.add(sim)
#    #session.commit()

if __name__ == '__main__':
    main()
