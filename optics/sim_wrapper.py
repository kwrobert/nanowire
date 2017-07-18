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
# import gc3libs
# from gc3libs.core import Core, Engine

# get our custom config object and the logger function
from utils.simulator import Simulator
from utils.config import Config
from utils.utils import configure_logger, make_hash, get_combos
# from rcwa_app import RCWA_App
from collections import OrderedDict

def parse_file(path):
    """Super simple utility to parse a yaml file given a path"""
    with open(path, 'r') as cfile:
        text = cfile.read()
    conf = yaml.load(text, Loader=yaml.Loader)
    return conf


def make_confs(global_conf):
    """Make all the configuration dicts for each parameter combination"""
    log = logging.getLogger()
    log.info('Constructing simulator objects ...')
    locs, combos, bin_size = get_combos(global_conf, global_conf.variable)
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
            if 'frequency' in locs[i]:
                sim_conf[global_conf.variable[i]] = {'type': 'fixed',
                                                     'value': float(combo[i]),
                                                     'bin_size': bin_size}
            else:
                sim_conf[global_conf.variable[i]] = {
                    'type': 'fixed', 'value': float(combo[i])}
        confs.append(sim_conf)
    return confs


def run_sim(conf):
    """
    Actually runs simulation in a given directory.  Expects a tuple containing
    the absolute path to the job directory as the first element and the
    configuration object for the job as the second element. 
    """
    log = logging.getLogger()
    start = time.time()
    sim = Simulator(copy.deepcopy(conf))
    if not sim.conf.variable_thickness:
        sim.conf.interpolate()
        sim.conf.evaluate()
        sim.update_id()
        sim.make_logger()
        try:
            os.makedirs(sim.dir)
        except OSError:
            pass
        period = sim.conf['Simulation']['params']['array_period']['value']
        try:
            rad = sim.conf['Layers']['NW_AlShell']['params']['shell_radius']['value']
        except KeyError:
            rad = sim.conf['Layers']['NW_AlShell']['params']['core_radius']['value']

        if rad > period/2.0:
            log.info('sim %s has shell radius larger than half the period,'
                     ' returning'%sim.id)
            return
        log.info('Executing sim %s'%sim.id[0:10])
        sim.get_data()
        # sim.mode_solve()
    else:
        log.info('Computing a thickness sweep at %s' % sim.id[0:10])
        orig_id = sim.id[0:10]
        # Get all combinations of layer thicknesses
        keys, combos, bin_size = get_combos(sim.conf, sim.conf.variable_thickness)
        # Update base directory to new sub directory
        sim.conf['General']['base_dir'] = sim.dir
        # Set things up for the first combo
        combo = combos.pop()
        # First update all the thicknesses in the config. We make a copy of the
        # list because it gets continually updated in the config object
        var_thickness = sim.conf.variable_thickness
        for i in range(len(combo)):
            keyseq = var_thickness[i]
            sim.conf[keyseq] = {'type': 'fixed', 'value': float(combo[i])}
        # With all the params updated we can now run substutions and
        # evaluations in the config that make have referred to some thickness
        # data, then make the subdir from the sim id and get the data
        sim.conf.interpolate()
        sim.conf.evaluate()
        sim.update_id()
        sim.make_logger()
        try:
            os.makedirs(sim.dir)
        except OSError:
            pass
        subpath = os.path.join(orig_id, sim.id[0:10])
        log.info('Computing initial thickness at %s' % subpath)
        sim.get_data()
        # Now we can repeat the same exact process, but instead of rebuilding
        # the device we just update the thicknesses
        for combo in combos:
            for i in range(len(combo)):
                keyseq = var_thickness[i]
                sim.conf[keyseq] = {'type': 'fixed', 'value': float(combo[i])}
            sim.update_id()
            subpath = os.path.join(orig_id, sim.id[0:10])
            log.info('Computing additional thickness at %s' % subpath)
            os.makedirs(sim.dir)
            sim.get_data(update=True)
    end = time.time()
    runtime = end - start
    log.info('Simulation {} completed in {:.2}'
             ' seconds!'.format(sim.id[0:10], runtime))
    test = "hello"
    test2 = "world"
    sim.clean_sim()
    # del sim.log
    return sim 
    # return (test, test2) 


def gc3_submit(gconf, sim_confs):
    """
    This function runs jobs on a bunch of remote hosts via SSH
    and well as on the local machine using a library called gc3pie. Requires
    gc3pie to be installed and configured. Currently super slow and not really
    worth using.
    """
    log = logging.getLogger()
    log.info('GC3 FUNCTION')
    jobs = []
    for conf in sim_confs:
        # Set up the config object and make local simulation directory
        conf.interpolate()
        conf.evaluate()
        sim_id = make_hash(conf.data)
        sim_dir = os.path.join(conf['General']['base_dir'], sim_id[0:10])
        conf['General']['sim_dir'] = sim_dir
        try:
            os.makedirs(sim_dir)
        except OSError:
            pass
        conf.write(os.path.join(sim_dir, 'sim_conf.yml'))
        # Create the GC3 Application object and append to list of apps to run
        app = RCWA_App(conf)
        jobs.append(app)
    print(jobs)
    # Get the config for all resources, auth, etc.
    cfg = gc3libs.config.Configuration(*gc3libs.Default.CONFIG_FILE_LOCATIONS,
                                       auto_enable_auth=True)
    gcore = Core(cfg)
    eng = Engine(gcore, tasks=jobs, retrieve_overwrites=True)
    # eng = Engine(gcore, tasks=jobs)
    try:
        eng.progress()
        stats = eng.stats()
        while stats['TERMINATED'] < stats['total']:
            time.sleep(10)
            print('Checking jobs')
            eng.progress()
            stats = eng.stats()
            print(stats)
            states = eng.update_job_state()
            print(states)
    except KeyboardInterrupt:
        print('KILLING REMOTE JOBS BEFORE QUITTING')
        print('PLEASE BE PATIENT')
        # Kill all remote jobs
        for task in jobs:
            eng.kill(task)
        # update job states
        eng.progress()
        # free remote resources
        for task in jobs:
            eng.free(task)
        # now raise exception
        raise

def execute_jobs(gconf, confs):
    """Given a list of configuration dictionaries, run them either serially or in
    parallel by applying run_sim to each dict. We do this instead of applying
    to an actual Simulator object because the Simulator objects are not
    pickeable and thus cannot be parallelized by the multiprocessing lib"""

    log = logging.getLogger()
    if gconf['General']['execution'] == 'serial':
        log.info('Executing sims serially')
        for conf in confs:
            run_sim(conf)
    elif gconf['General']['execution'] == 'parallel':
        # All this crap is necessary for killing the parent and all child
        # processes with CTRL-C
        num_procs = gconf['General']['num_cores']
        log.info('Executing sims in parallel using %s cores ...',str(num_procs))
        pool = mp.Pool(processes=num_procs)
        total_sims = len(confs)
        remaining_sims = len(confs)
        def callback(completed_sim):
            # print('Thing passed to callback')
            # print(completed_sim)
            # print(isinstance(completed_sim, Simulator))
            callback.remaining_sims -= 1
            # callback.log.info('%i out of %i simulations remaining'%(callback.remaining_sims,
            #                                                 callback.total_sims))
            print('%i out of %i simulations remaining'%(callback.remaining_sims,
                                                            callback.total_sims))
        callback.remaining_sims = remaining_sims
        callback.total_sims = total_sims
        callback.log = log
        results = []
        try:
            # res = pool.map_async(run_sim, confs, callback=callback)
            # res.get(999999999)
            # pool.close()
            for conf in confs:
                res = pool.apply_async(run_sim, (conf,), callback=callback)
                results.append(res)
            for r in results:
                # We need to add this really long timeout so that subprocesses
                # receive keyboard interrupts. If our simulations take longer
                # than this timeout, an exception would be raised but that
                # should never happen
                r.wait(99999999)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
        pool.join()
        # print(results)
    elif gconf['General']['execution'] == 'gc3':
        log.info('Executing jobs using gc3 submission tools')
        gc3_submit(gconf, confs)


def spectral_wrapper(opt_pars, baseconf):
    """A wrapper function to handle spectral sweeps and postprocessing for the scipy minimizer. It
    accepts the initial guess as a vector, the base config file, and the keys that map to the
    initial guesses. It runs a spectral sweep (in parallel if specified), postprocesses the results,
    and returns the spectrally weighted reflection"""
    # TODO: Pass in the quantity we want to optimize as a parameter, then compute and return that
    # instead of just assuming reflection

    # Optimizing shell thickness could result is negative thickness so we need to take absolute
    # value here
    #opt_pars[0] = abs(opt_pars[0])
    log = logging.getLogger()
    log.info('Param keys: %s' % str(baseconf.optimized))
    log.info('Current values %s' % str(opt_pars))
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
        valseq = list(keyseq) + ['value']
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
    info_file = os.path.join(basedir, 'conv_info.txt')
    if os.path.isfile(info_file):
        conv_dict = {}
        with open(info_file, 'r') as info:
            for line in info:
                freq, numbasis, conv_status = line.strip().split(',')
                if conv_status == 'converged':
                    conv_dict[freq] = (True, numbasis)
                elif conv_status == 'unconverged':
                    conv_dict[freq] = (False, numbasis)
        for sim in sims:
            freq = str(sim.conf['Simulation']['params']['frequency']['value'])
            conv, numbasis = conv_dict[freq]
            # Turn off adaptive convergence and update the number of basis
            # terms
            if conv:
                log.info('Frequency %s converged at %s basis'
                         ' terms' % (freq, numbasis))
                sim.conf['General']['adaptive_convergence'] = False
                sim.conf['Simulation']['params'][
                    'numbasis']['value'] = int(numbasis)
            # For sims that haven't converged, set the number of basis terms to the last
            # tested value so we're closer to our goal of convergence
            else:
                log.info('Frequency %s unconverged at %s basis'
                         ' terms' % (freq, numbasis))
                sim.conf['Simulation']['params'][
                    'numbasis']['value'] = int(numbasis)
    # With the leaf directories made and the number of basis terms adjusted,
    # we can now kick off our frequency sweep
    execute_jobs(baseconf, sims)

    #####
    # TODO: This needs to be generalized. The user could pass in the name of
    # a postprocessing function in the config file. The function will be called
    # and used as the quantity for optimization
    #####

    # With our frequency sweep done, we now need to postprocess the results.
    # Configure logger
    # log = configure_logger('error','postprocess',
    #                          os.path.join(baseconf['General']['base_dir'],'logs'),
    #                          'postprocess.log')
    # Compute transmission data for each individual sim
    cruncher = pp.Cruncher(baseconf)
    cruncher.collect_sims()
    for sim in cruncher.sims:
        cruncher.transmissionData(sim)
    # Now get the fraction of photons absorbed
    gcruncher = pp.Global_Cruncher(
        baseconf, cruncher.sims, cruncher.sim_groups, cruncher.failed_sims)
    gcruncher.group_against(
        ['Simulation', 'params', 'frequency', 'value'], baseconf.variable)
    photon_fraction = gcruncher.fractional_absorbtion()[0]
    # Lets store information we discovered from our adaptive convergence procedure so we can resue
    # it in the next iteration.
    if baseconf['General']['adaptive_convergence']:
        log.info('Storing adaptive convergence results ...')
        with open(info_file, 'w') as info:
            for sim in sims:
                freq = sim.conf['Simulation']['params']['frequency']['value']
                conv_path = os.path.join(sim.dir, 'converged_at.txt')
                nconv_path = os.path.join(sim.dir, 'not_converged_at.txt')
                if os.path.isfile(conv_path):
                    conv_f = open(conv_path, 'r')
                    numbasis = conv_f.readline().strip()
                    conv_f.close()
                    conv = 'converged'
                elif os.path.isfile(nconv_path):
                    conv_f = open(nconv_path, 'r')
                    numbasis = conv_f.readline().strip()
                    conv_f.close()
                    conv = 'unconverged'
                else:
                    # If we were converged on a previous iteration, adaptive
                    # convergence was switched off and there will be no file to
                    # read from
                    conv = 'converged'
                    numbasis = sim.conf['Simulation'][
                        'params']['numbasis']['value']
                info.write('%s,%s,%s\n' % (str(freq), numbasis, conv))
        log.info('Finished storing convergence results!')
    #print('Total time = %f'%delta)
    #print('Num calls after = %i'%gcruncher.weighted_transmissionData.called)
    # This is a minimizer, we want to maximize the fraction of photons absorbed and thus minimize
    # 1 minus that fraction
    return 1 - photon_fraction


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
    ftol = conf['General']['func_opt_tol']
    max_iter = conf['General']['opt_max_iter']
    os.makedirs(os.path.join(conf['General']['base_dir'], 'opt_dir'))
    # Run the simplex optimizer
    opt_val = optz.minimize(spectral_wrapper,
                            guess,
                            args=(conf,),
                            method='Nelder-Mead',
                            options={'maxiter': max_iter, 'xatol': tol,
                                     'fatol': ftol, 'disp': True})
    log.info(opt_val.message)
    log.info('Optimal values')
    log.info(conf.optimized)
    log.info(opt_val.x)
    # Write out the results to a file
    out_file = os.path.join(
        conf['General']['base_dir'], 'optimization_results.txt')
    with open(out_file, 'w') as out:
        out.write('# Param name, value\n')
        for key, value in zip(conf.optimized, opt_val.x):
            out.write('%s: %f\n' % (str(key), value))
    return opt_val.x


def run(conf, log):
    """The main run methods that decides what kind of simulation to run based on the
    provided config object"""

    basedir = conf['General']['base_dir']
    lfile = os.path.join(basedir, 'logs/sim_wrapper.log')
    # Configure logger
    logger = configure_logger(level=log, console=True, logfile=lfile)
    # Just a simple single simulation
    if not conf.optimized:
        # Get all the sims
        sims = make_confs(conf)
        logger.info("Executing job campaign")
        execute_jobs(conf, sims)
    # If we have variable params, do a parameter sweep
    elif conf.optimized:
        run_optimization(conf)
    else:
        logger.error('Unsupported configuration for a simulation run. Not a '
                     'single sim, sweep, or optimization. Make sure your sweeps are '
                     'configured correctly, and if you are running an optimization '
                     'make sure you do not have any sorting parameters specified')


def pre_check(conf_path, conf):
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
        input('Would you like to continue? CTRL-C to exit, any other key to continue: ')
    # Copy provided config file to basedir
    try:
        os.makedirs(base)
    except OSError:
        pass
    fname = os.path.basename(conf_path)
    new_path = os.path.join(base, fname)
    try:
        shutil.copy(conf_path, new_path)
    except shutil.SameFileError:
        pass
    # TODO: Add checks between specified x,y,z samples and plane vals in
    # plotting section


def main():

    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file', type=str, help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('--log_level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        global_conf = Config(path=os.path.abspath(args.config_file))
        # Expand all the environment variables in the global config but
        # otherwise don't modify anything
        global_conf.expand_vars()
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    # setup_db(sim['General']['db_name'])
    # pre_check(os.path.abspath(args.config_file),conf)
    run(global_conf, args.log_level)

# def db_test():
#    with session_scope() as session:
#        print(session)
#
# def main():
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
