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

class Config(object):

    def __init__(self,path=None,data=None):
        if path:
            self.data = self._parse_file(path)
        elif data:
            self.data = data
        else:
            raise ValueError('You must supply either a path to a YAML file or a python'
            ' dictionary')
        pprint.pprint(self.data)
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

    def copy(self):
        """Returns a copy of the current config object"""
        return Config(data=self.data)

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
