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
import postprocess as pp
import time
import pprint
import argparse as ap
import ruamel.yaml as yaml
import logging

# get our custom config object and the logger function
from utils.simulation import *
from collections import MutableMapping,OrderedDict
from functools import wraps
from contextlib import contextmanager
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

__DB_ENGINE__ = None
__SESSION_FACTORY__ = None
Base = declarative_base()

def setup_db(name,verbose=False):
    """Sets up the module-scope database engine and the session factory"""
    global __DB_ENGINE__
    global __SESSION_FACTORY__
    if not __DB_ENGINE__ and not __SESSION_FACTORY__:
        __DB_ENGINE__ = create_engine('sqlite:///{}'.format(name),echo=verbose)
        __SESSION_FACTORY__ = sessionmaker(bind=__DB_ENGINE__)
        Base.metadata.create_all(__DB_ENGINE__)
        Simulation.metadata.create_all(__DB_ENGINE__)
    else:
        raise RuntimeError('DB Engine already set')

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = __SESSION_FACTORY__()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

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
        

def make_single_sim(sim):
    """Create a configuration file for a single simulation, such that it is ready to be
    consumed by the Lua script. Return the path to the config file and the
    Simulation object itself"""
    with session_scope() as sess:
        log = logging.getLogger('sim_wrapper')
        log.info("Running single sim")
        del sim['Postprocessing']
        # Instantiate simulation object
        #sim = Simulation(sim_conf)
        path = os.path.join(sim['General']['base_dir'],sim.id[0:10])
        print(sim.id)
        sim['General']['sim_dir'] = path
        print(sim.id)
        sess.add(sim)
        # The dir is already made when the logger is configured but this is a safety measure i guess?
        try:
            os.makedirs(path)
        except OSError:
            log.info('Data directory already exists, appending timestamp')
            stamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
            path = path+'__{}'.format(stamp)
            os.makedirs(path)
        # Now write the config file to the data subdir
        out = os.path.join(path,'sim_conf.yml')
        sim.write(out)
        # Copy sim script to sim dir
        base_script = sim['General']['sim_script']
        script = os.path.join(path,os.path.basename(base_script))
        shutil.copy(base_script,script)
    return (path,sim)
    
def run_sim(jobtup):
    """Actually runs simulation in a given directory using subprocess.call. Expects a tuple
    containing the absolute path to the job directory as the first element and
    the configuration object for the job as the second element"""

    log = logging.getLogger('sim_wrapper')
    jobpath,sim = jobtup
    with session_scope() as session:     
        timed = sim['General']['save_time']
        tout = os.path.join(jobpath,'timing.dat')
        simlog = os.path.join(jobpath,'sim.log')
        script = sim['General']['sim_script']
        ini_file = os.path.join(jobpath,'sim_conf.yml')
        if timed:
            log.debug('Executing script with timing wrapper ...')
            cmd = 'command time -vv -o %s lua %s %s 2>&1 | tee %s'%(tout,script,ini_file,simlog)
            #cmd = '/usr/bin/perf stat -o timing.dat -r 5 /usr/bin/lua %s %s'%(script,ini_file)
        else:
            cmd = 'command lua %s %s 2>&1 | tee %s'%(script,ini_file,simlog)
        log.debug('Subprocess command: %s',cmd)
        relpath = os.path.relpath(jobpath,sim['General']['treebase'])
        log.info("Starting simulation for %s ...",relpath)
        completed = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #log.info('Simulation stderr: %s',completed.stderr)
        log.info("Simulation stderr: %s",completed.stderr)
        log.debug('Simulation stdout: %s',completed.stdout)
        # Convert text data files to npz binary file
        if sim['General']['save_as'] == 'npz':
            log.info('Converting data at %s to npz format',relpath)
            convert_data(jobpath,sim)
        log.info("Finished simulation for %s!",relpath)
    return None

def run(conf,log):
    """The main run methods that decides what kind of simulation to run based on the
    provided config object"""

    basedir = conf['General']['base_dir']
    logdir = os.path.join(basedir,'logs')
    # Configure logger
    logger = configure_logger(log,'sim_wrapper',logdir,'sim_wrapper.log')
    # Just a simple single simulation
    if conf.fixed and not conf.variable and not conf.sorting and not conf.optimized:
        logger.debug('Entering single sim function from main')
        job = make_single_sim(conf)
        run_sim(job)
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
    elif conf.optimized and not conf.sorting:
        logger.debug('Entering optimization function from main')
        run_optimization(conf)
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
        sim = Simulation(path=os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()
    
    setup_db(sim['General']['db_name'])
    #pre_check(os.path.abspath(args.config_file),conf)
    run(sim,args.log_level)

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
