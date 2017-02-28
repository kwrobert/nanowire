import shutil
#import subprocess
import itertools
import os
#import glob
#  import datetime
import copy
#  import hashlib
import multiprocessing as mp
#import pandas
import numpy as np
import scipy.optimize as optz
import postprocess as pp
import time
#import pprint
import argparse as ap
import ruamel.yaml as yaml
import logging

# get our custom config object and the logger function
#from utils.simulation import *
#from utils.config import Config
from utils.simulator import Simulator
from utils.config import Config
from utils.utils import configure_logger
from collections import OrderedDict
#  from functools import wraps
#  from contextlib import contextmanager
#  from sqlalchemy.ext.declarative import declarative_base
#  from sqlalchemy.orm import sessionmaker

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

    log = logging.getLogger()
    log.info("Constructing dictionary of options and their values ...")
    # Get the list of values from all our variable keysets
    optionValues = OrderedDict()
    for keyset in keysets:
        par = '.'.join(keyset)
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

def make_confs(global_conf):
    """Make all the configuration dicts for each parameter combination"""
    log = logging.getLogger()
    log.info('Constructing simulator objects ...')
    locs,combos = get_combos(global_conf,global_conf.variable)
    confs = []
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
        confs.append(sim_conf)
    return confs
#      with session_scope() as sess:
#          log = logging.getLogger()
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

def _get_data(sim,update=False):
    log = logging.getLogger()
    sim.conf.write(os.path.join(sim.dir,'sim_conf.yml'))
    start = time.time()
    if not update:
        sim.configure()
        sim.build_device()
        sim.set_excitation()
    else:
        sim.update_thicknesses()
    sim.save_field()
    sim.save_fluxes()
    end = time.time()
    runtime = end-start
    time_file = os.path.join(sim.dir,'time.dat')
    with open(time_file,'w') as out:
        out.write('{}\n'.format(runtime))
    log.info('Simulation {} completed in {:.2}'
             ' seconds!'.format(sim.id[0:10],runtime))

def run_sim(conf):
    """Actually runs simulation in a given directory using subprocess.call. Expects a tuple
    containing the absolute path to the job directory as the first element and
    the configuration object for the job as the second element"""
    log = logging.getLogger()
    sim = Simulator(copy.deepcopy(conf))
    try:
        os.makedirs(sim.dir)
    except OSError:
        pass
    if not sim.conf.variable_thickness:
        log.info('Executing sim %s'%sim.id[0:10])
        _get_data(sim)
    else:
        log.info('Computing a thickness sweep at %s'%sim.id[0:10])
        orig_id = sim.id[0:10]
        # Get all combinations of layer thicknesses
        keys,combos = get_combos(sim.conf,sim.conf.variable_thickness)
        # Update base directory to new sub directory
        sim.conf['General']['base_dir'] = sim.dir
        # Set things up for the first combo
        combo = combos.pop()
        # First update all the thicknesses in the config. We make a copy of the
        # list because it gets continually updated in the config object
        var_thickness = sim.conf.variable_thickness
        for i in range(len(combo)):
            keyseq = var_thickness[i]
            sim.conf[keyseq] = {'type':'fixed','value':float(combo[i])}
        # With all the params updated we can now make the subdir from the
        # sim id and get the data
        sim.update_id()
        os.makedirs(sim.dir)
        subpath = os.path.join(orig_id,sim.id[0:10])
        log.info('Computing initial thickness at %s'%subpath)
        _get_data(sim)
        # Now we can repeat the same exact process, but instead of rebuilding
        # the device we just update the thicknesses
        for combo in combos:
            for i in range(len(combo)):
                keyseq = var_thickness[i]
                sim.conf[keyseq] = {'type':'fixed','value':float(combo[i])}
            sim.update_id()
            subpath = os.path.join(orig_id,sim.id[0:10])
            log.info('Computing additional thickness at %s'%subpath)
            os.makedirs(sim.dir)
            _get_data(sim,update=True)
    return
    #  with session_scope() as session:
    #      timed = sim['General']['save_time']
    #      tout = os.path.join(jobpath,'timing.dat')
    #      simlog = os.path.join(jobpath,'sim.log')
    #      script = sim['General']['sim_script']
    #      ini_file = os.path.join(jobpath,'sim_conf.yml')
    #      if timed:
    #          log.debug('Executing script with timing wrapper ...')
    #          cmd = 'command time -vv -o %s lua %s %s 2>&1 | tee %s'%(tout,script,ini_file,simlog)
    #          #cmd = '/usr/bin/perf stat -o timing.dat -r 5 /usr/bin/lua %s %s'%(script,ini_file)
    #      else:
    #          cmd = 'command lua %s %s 2>&1 | tee %s'%(script,ini_file,simlog)
    #      log.debug('Subprocess command: %s',cmd)
    #      relpath = os.path.relpath(jobpath,sim['General']['treebase'])
    #      log.info("Starting simulation for %s ...",relpath)
    #      completed = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #      #log.info('Simulation stderr: %s',completed.stderr)
    #      log.info("Simulation stderr: %s",completed.stderr)
    #      log.debug('Simulation stdout: %s',completed.stdout)
    #      # Convert text data files to npz binary file
    #      if sim['General']['save_as'] == 'npz':
    #          log.info('Converting data at %s to npz format',relpath)
    #          convert_data(jobpath,sim)
    #      log.info("Finished simulation for %s!",relpath)
    #  return None

def execute_jobs(gconf,confs):
    """Given a list of configuration dictionaries, run them either serially or in
    parallel by applying run_sim to each dict. We do this instead of applying
    to an actual Simulator object because the Simulator objects are not
    pickeable and thus cannot be parallelized by the multiprocessing lib"""

    log = logging.getLogger()
    if not gconf['General']['parallel']:
        log.info('Executing sims serially')
        for conf in confs:
            run_sim(conf)
    else:
        num_procs = mp.cpu_count() - gconf['General']['reserved_cores']
        log.info('Executing sims in parallel using %s cores ...',str(num_procs))
        pool = mp.Pool(processes=num_procs)
        try:
            res = pool.map_async(run_sim,confs)
            res.get(999999999) 
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
        pool.join()

def spectral_wrapper(opt_pars,baseconf):
    """A wrapper function to handle spectral sweeps and postprocessing for the scipy minimizer. It
    accepts the initial guess as a vector, the base config file, and the keys that map to the
    initial guesses. It runs a spectral sweep (in parallel if specified), postprocesses the results,
    and returns the spectrally weighted reflection"""
    # TODO: Pass in the quantity we want to optimize as a parameter, then compute and return that
    # instead of just assuming reflection

    ## Optimizing shell thickness could result is negative thickness so we need to take absolute
    ## value here
    #opt_pars[0] = abs(opt_pars[0])
    log = logging.getLogger()
    log.info('Param keys: %s'%str(baseconf.optimized))
    log.info('Current values %s'%str(opt_pars))
    # Clean up old data unless the user asked us not to. We do this first so on the last
    # iteration all our data is left intact
    basedir = baseconf['General']['base_dir']
    if not baseconf['General']['opt_keep_intermediates']:
        for item in os.listdir(basedir):
            if os.path.isdir(item) and item != 'logs':
                shutil.rmtree(item)
    # Set the value key of all the params we are optimizing over to the current
    # guess
    for i in range(len(baseconf.optimized)):
        keyseq = baseconf.optimized[i]
        print(keyseq)
        valseq = list(keyseq)+['value']
        print(valseq)
        baseconf[valseq] = float(opt_pars[i])
    # Make all the sim objects
    sims = make_confs(baseconf)
    # Let's reuse the convergence information from the previous iteration if it exists
    # NOTE: This kind of assumes your initial guess was somewhat decent with regards to the in plane
    # geometric variables and the optimizer is staying relatively close to that initial guess. If
    # the optimizer is moving far away from its previous guess at each step, then the fact that a
    # specific frequency may have been converged previously does not mean it will still be converged
    # with this new set of variables.
    info_file = os.path.join(basedir,'conv_info.txt')
    if os.path.isfile(info_file):
        conv_dict = {}
        with open(info_file,'r') as info:
            for line in info:
                freq,numbasis,conv_status = line.strip().split(',')
                if conv_status == 'converged':
                    conv_dict[freq] = (True,numbasis)
                elif conv_status == 'unconverged':
                    conv_dict[freq] = (False,numbasis)
        for sim in sims:
            freq = str(sim.conf['Simulation']['params']['frequency']['value'])
            conv,numbasis = conv_dict[freq]
            # Turn off adaptive convergence and update the number of basis
            # terms
            if conv:
                log.info('Frequency %s converged at %s basis'
                         ' terms'%(freq,numbasis))
                sim.conf['General']['adaptive_convergence'] = False
                sim.conf['Simulation']['params']['numbasis']['value'] = int(numbasis)
            # For sims that haven't converged, set the number of basis terms to the last
            # tested value so we're closer to our goal of convergence
            else:
                log.info('Frequency %s unconverged at %s basis'
                         ' terms'%(freq,numbasis))
                sim.conf['Simulation']['params']['numbasis']['value'] = int(numbasis)
    # With the leaf directories made and the number of basis terms adjusted, we can now kick off our frequency sweep
    execute_jobs(baseconf,sims)

    #####
    # TODO: This needs to be generalized. The user could pass in the name of
    # a postprocessing function in the config file. The function will be called
    # and used as the quantity for optimization
    #####

    # With our frequency sweep done, we now need to postprocess the results.
    # Configure logger
    #log = configure_logger('error','postprocess',
    #                          os.path.join(baseconf['General']['base_dir'],'logs'),
    #                          'postprocess.log')
    # Compute transmission data for each individual sim
    cruncher = pp.Cruncher(baseconf)
    cruncher.collect_sims()
    for sim in cruncher.sims:
        cruncher.transmissionData(sim)
    # Now get the fraction of photons absorbed
    gcruncher = pp.Global_Cruncher(baseconf,cruncher.sims,cruncher.sim_groups,cruncher.failed_sims)
    photon_fraction = gcruncher.Jsc()[0]
    # Lets store information we discovered from our adaptive convergence procedure so we can resue
    # it in the next iteration.
    if baseconf['General']['adaptive_convergence']:
        log.info('Storing adaptive convergence results ...')
        with open(info_file,'w') as info:
            for sim in sims:
                freq = sim.conf['Simulation']['params']['frequency']['value']
                conv_path = os.path.join(sim.dir,'converged_at.txt')
                nconv_path = os.path.join(sim.dir,'not_converged_at.txt')
                if os.path.isfile(conv_path):
                    conv_f = open(conv_path,'r')
                    numbasis = conv_f.readline().strip()
                    conv_f.close()
                    conv = 'converged'
                elif os.path.isfile(nconv_path):
                    conv_f = open(nconv_path,'r')
                    numbasis = conv_f.readline().strip()
                    conv_f.close()
                    conv = 'unconverged'
                else:
                    # If we were converged on a previous iteration, adaptive
                    # convergence was switched off and there will be no file to
                    # read from
                    conv = 'converged'
                    numbasis = sim.conf['Simulation']['params']['numbasis']['value']
                info.write('%s,%s,%s\n'%(str(freq),numbasis,conv))
        log.info('Finished storing convergence results!')
    #print('Total time = %f'%delta)
    #print('Num calls after = %i'%gcruncher.weighted_transmissionData.called)
    # This is a minimizer, we want to maximize the fraction of photons absorbed and thus minimize
    # 1 minus that fraction
    return 1-photon_fraction

def run_optimization(conf):
    """Runs an optimization on a given set of parameters"""
    log = logging.getLogger()
    log.info("Running optimization")
    print(conf.optimized)
    # Make sure the only variable parameter we have is a sweep through
    # frequency
    for keyseq in conf.variable:
        if keyseq[-1] != 'frequency':
            log.error('You should only be sweep through frequency during an '
                      'optimization')
            quit()
    # Collect all the guesses
    guess = np.zeros(len(conf.optimized))
    for i in range(len(conf.optimized)):
        keyseq = conf.optimized[i]
        par_data = conf.get(keyseq)
        guess[i] = par_data['guess']
    # Max iterations and tolerance
    tol = conf['General']['opt_tol']
    max_iter = conf['General']['opt_max_iter']
    os.makedirs(os.path.join(conf['General']['base_dir'],'opt_dir'))
    # Run the simplex optimizer
    opt_val = optz.minimize(spectral_wrapper,
                            guess,
                            args=(conf,),
                            method='Nelder-Mead',
                            options={'maxiter':max_iter,'xatol':tol,'disp':True})
    log.info(opt_val.message)
    log.info('Optimal values')
    log.info(conf.optimized)
    log.info(opt_val.x)
    # Write out the results to a file
    out_file = os.path.join(conf['General']['base_dir'],'optimization_results.txt')
    with open(out_file,'w') as out:
        out.write('# Param name, value\n')
        for key, value in zip(conf.optimized,opt_val.x):
            out.write('%s: %f\n'%(str(key),value))
    return opt_val.x
    
def run(conf,log):
    """The main run methods that decides what kind of simulation to run based on the
    provided config object"""

    basedir = conf['General']['base_dir']
    lfile = os.path.join(basedir,'logs/sim_wrapper.log')
    # Configure logger
    logger = configure_logger(level=log,console=True,logfile=lfile)
    # Just a simple single simulation
    if not conf.optimized:
        # Get all the sims
        sims = make_confs(conf)
        logger.info("Executing job campaign")
        execute_jobs(conf,sims)
    # If we have variable params, do a parameter sweep
    elif conf.optimized:
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
