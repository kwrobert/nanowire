import shutil
import subprocess
import itertools
import argparse as ap
import os
import glob
import datetime
from collections import OrderedDict
import multiprocessing as mp
import pandas
import numpy as np
import scipy.optimize as optz
import postprocess_yaml as pp
import time
import pprint
# Get our custom config object and the logger function
from utils.config import *


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
        print(pdict)
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
    keys,combos = get_combos(nodes[0][1],nodes[0][1].variable)
    # Build all the leaves at every node in the directory tree
    leaves = []
    for nodepath,conf in nodes:
        # Loop through each unique combination of parameters
        for combo in combos:
            # Make a unique working directory based on parameter combo
            workdir = ''
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
            del sim_conf['Postprocessing']
            sim_conf['General']['sim_dir'] = fullpath
            # Now we just overwrite all the variable parameters with their new
            # fixed values. Note that itertools.product is so wonderful and
            # nice that it preserves the order of the values in every combo
            # such that the combo values always line up with the proper
            # parameter name
            for i in range(len(combo)):
                sim_conf[conf.variable[i]] = {'type':'fixed','value':float(combo[i])}
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
            base_script = sim_conf['General']['sim_script']
            script = os.path.join(nodepath,os.path.basename(base_script))
            if not os.path.exists(script):
                shutil.copy(base_script,script)
            leaves.append((fullpath,sim_conf))
    log.debug('Here are the leaves: %s',str(leaves))
    log.info('Finished building leaves!')
    return leaves

def make_nodes(conf):
    """Given a global config file that specifies all the different sorted and
    variable sweeps, create all the nodes and all the leaves in the directory
    tree. First, the nodes are built. Then, the list of nodes is passed to
    make_leaves to create all the leaves"""

    # Get access to logger
    log = logging.getLogger('sim_wrapper')
    log.info("Constructing all nodes in sorted directory tree ...")
    # Get a list of the parameters names (keys) and a list of lists of values where the ith value
    # of the inner lists corresponds to the ith param name in keys. The inner list consists of
    # a unique combination of variable parameters
    keys,combos = get_combos(conf,conf.sorting)
    # Make all the nodes in the directory tree
    nodes = []
    # Loop through all the combos. Each combo corresponds to a unique path down
    # the directory tree
    for combo in combos:
        # Copy global config object
        sub_conf = conf.copy()
        # Build each node in the path and add params to new config object
        path = conf['General']['base_dir']
        for i in range(len(combo)):
            # Now we just overwrite all the sorting parameters with their new
            # fixed values. Note that itertools.product is so wonderful and
            # nice that it preserves the order of the values in every combo
            # such that the combo values always line up with the proper
            # parameter name
            sub_conf[conf.sorting[i]] = {'type':'fixed','value':float(combo[i])}
            subdir = '{}_{:G}'.format(keys[i],combo[i])
            path = os.path.join(path,subdir)
            try:
                os.makedirs(path)
            except OSError:
                pass
            log.info("Created node %s",path)
            sub_conf['General']['base_dir'] = path
            # At each node in the tree we write out a complete config file that
            # could be used to rerun all jobs in a certain section of the tree,
            # or postprocess only that section of the tree
            conf_file = os.path.join(path,'sorted_sweep_conf_%s.yml'%keys[i])
            sub_conf.write(conf_file)
        nodes.append((path,sub_conf))
    log.debug('Here are the nodes: %s',str(nodes))
    log.info('Finished building nodes!')
    leaves = make_leaves(nodes)
    return leaves

def make_single_sim(conf):
    """Create a configuration file for a single simulation, such that it is ready to be
    consumed by the Lua script. Return the path to the config file and the config object
    itself"""

    log = logging.getLogger('sim_wrapper')
    log.info("Running single sim")
    # Make the simulation dir
    basedir = conf['General']['base_dir']
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
    del sim_conf['Postprocessing']
    sim_conf['General']['sim_dir'] = path
    # Now write the config file to the data subdir
    out = os.path.join(path,'sim_conf.yml')
    sim_conf.write(out)
    # Copy sim script to sim dir
    base_script = sim_conf['General']['sim_script']
    script = os.path.join(path,os.path.basename(base_script))
    shutil.copy(base_script,script)
    return (path,sim_conf)

def convert_data(path,conf):
    """Converts text file spit out by S4 into npz format for faster loading during postprocessing"""
    # Have we decided to ignore and remove H field files?
    ignore = conf['General']['ignore_h']
    # Convert e field data files
    bname = conf['General']['base_name']
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
    timed = jobconf['General']['save_time']
    tout = os.path.join(jobpath,'timing.dat')
    simlog = os.path.join(jobpath,'sim.log')
    script = jobconf['General']['sim_script']
    ini_file = os.path.join(jobpath,'sim_conf.yml')
    if timed:
        log.debug('Executing script with timing wrapper ...')
        cmd = 'command time -vv -o %s lua %s %s 2>&1 | tee %s'%(tout,script,ini_file,simlog)
        #cmd = '/usr/bin/perf stat -o timing.dat -r 5 /usr/bin/lua %s %s'%(script,ini_file)
    else:
        cmd = 'command lua %s %s 2>&1 | tee %s'%(script,ini_file,simlog)
    log.debug('Subprocess command: %s',cmd)
    relpath = os.path.relpath(jobpath,jobconf['General']['treebase'])
    log.info("Starting simulation for %s ...",relpath)
    completed = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #log.info('Simulation stderr: %s',completed.stderr)
    log.info("Simulation stderr: %s",completed.stderr)
    log.debug('Simulation stdout: %s',completed.stdout)
    # Convert text data files to npz binary file
    if jobconf['General']['save_as'] == 'npz':
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
    if not gconf['General']['parallel']:
        log.info('Executing jobs serially')
        for job in jobs:
            run_sim(job)
    else:
        num_procs = mp.cpu_count() - gconf['General']['reserved_cores']
        log.info('Executing jobs in parallel using %s cores ...',str(num_procs))
        with mp.Pool(processes=num_procs) as pool:
            result = pool.map(run_sim,jobs)

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
    log=logging.getLogger('sim_wrapper')
    # Make sure we don't have any preexisting frequency dirs. We do this first so on the last
    # iteration all our data is left intact
    log.info('Param keys: %s'%str(baseconf.optimized))
    log.info('Current values %s'%str(opt_pars))
    basedir = baseconf['General']['base_dir']
    globstr = os.path.join(basedir,'frequency*')
    dirs = glob.glob(globstr)
    for adir in dirs:
        shutil.rmtree(adir)
    # Set the value key of all the params we are optimizing over to the current
    # guess
    for i in range(len(baseconf.optimized)):
        keyseq = baseconf.optimized[i]
        valseq = (*keyseq,'value')
        baseconf[valseq] = float(opt_pars[i])
    # With the parameters set, we can now make the leaf directories, which should only contain
    # frequency values
    node = (basedir,baseconf)
    leaves = make_leaves([node])
    # Let's reuse the convergence information from the previous iteration if it exists
    # NOTE: This kind of assumes your initial guess was somewhat decent with regards to the in plane
    # geometric variables and the optimizer is staying relatively close to that initial guess. If
    # the optimizer is moving far away from its previous guess at each step, then the fact that a
    # specific frequency may have been converged previously does not mean it will still be converged
    # with this new set of variables.
    info_file = os.path.join(basedir,'conv_info.txt')
    if os.path.isfile(info_file):
        with open(info_file,'r') as info:
            for line in info:
                conf_path,numbasis,conv_status = line.strip().split(',')
                sim_conf = Config(conf_path)
                rel = os.path.relpath(conf_path,basedir)
                log.info('Simulation at %s is %s at %s basis terms'%(rel,conv_status,numbasis))
                # Turn off adaptive convergence for all the sims that have converged and set their number of
                # basis terms to the proper value so we continue to use that value
                if conv_status == 'converged':
                    sim_conf[('General','adaptive_convergence')] = 'False'
                    sim_conf[('Simulation','params','numbasis','value')] = numbasis
                    with open(conf_path,'w') as configfile:
                        sim_conf.write(configfile)
                    # We have to recreate the converged_at.txt file because it gets deleted every
                    # iteration and we need it to create the info file later on
                    fpath = os.path.join(os.path.dirname(conf_path),'converged_at.txt')
                    with open(fpath,'w') as conv_at:
                        conv_at.write('%s\n'%numbasis)
                # For sims that haven't converged, lets at least set the number of basis terms to the last
                # tested value so we're closer to our goal of convergence
                elif conv_status == 'unconverged':
                    sim_conf.set(('Simulation','params','numbasis','value'),numbasis)
                    with open(conf_path,'w') as configfile:
                        sim_conf.write(configfile)
    # With the leaf directories made and the number of basis terms adjusted, we can now kick off our frequency sweep
    execute_jobs(baseconf,leaves)


    #####
    # TODO: This needs to be generalized. The user could pass in the name of
    # a postprocessing function in the config file. The function will be called
    # and used as the quantity for optimization
    #####

    # With our frequency sweep done, we now need to postprocess the results.
    # Configure logger
    logger = configure_logger('error','postprocess',
                              os.path.join(baseconf['General']['base_dir'],'logs'),
                              'postprocess.log')
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
            conv_files = glob.glob(os.path.join(basedir,'**/converged_at.txt'))
            for convf in conv_files:
                simpath = os.path.join(os.path.dirname(convf),'sim_conf.ini')
                with open(convf,'r') as cfile:
                    numbasis = cfile.readline().strip()
                info.write('%s,%s,%s\n'%(simpath,numbasis,'converged'))
            nconv_files = glob.glob(os.path.join(basedir,'**/not_converged_at.txt'))
            for nconvf in nconv_files:
                simpath = os.path.join(os.path.dirname(nconvf),'sim_conf.ini')
                with open(nconvf,'r') as cfile:
                    numbasis = cfile.readline().strip()
                info.write('%s,%s,%s\n'%(simpath,numbasis,'unconverged'))

    #print('Total time = %f'%delta)
    #print('Num calls after = %i'%gcruncher.weighted_transmissionData.called)
    # This is a minimizer, we want to maximize the fraction of photons absorbed and thus minimize
    # 1 minus that fraction
    return 1-photon_fraction

def run_optimization(conf):
    log = logging.getLogger('sim_wrapper')
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
    base = conf["General"]["basedir"]
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
        conf = Config(path=os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    #pre_check(os.path.abspath(args.config_file),conf)
    run(conf,args.log_level)

if __name__ == '__main__':
    main()
