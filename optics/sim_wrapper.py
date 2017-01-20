import shutil
import subprocess
import itertools
import argparse as ap
import os
import configparser as confp 
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

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.ConfigParser(interpolation=None)
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

def copy_conf_obj(old):
    from io import StringIO
    config_string = StringIO()
    old.write(config_string)
    # We must reset the buffer ready for reading.
    config_string.seek(0) 
    new_config = confp.ConfigParser(interpolation=None)
    new_config.optionxform = str
    new_config.read_file(config_string)
    return new_config

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

def make_single_sim(conf):
    log = logging.getLogger('sim_wrapper')
    log.info("Running single sim")
    # Make the simulation dir
    basedir = conf.get('General','basedir')
    # The dir is already made when the logger is configured but this is a safety measure i guess?
    try:
        path = os.path.join(basedir,'data')
        os.makedirs(path)
    except OSError:
        log.info('Data directory already exists, appending timestamp')
        stamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        path = os.path.join(basedir,'data__'+stamp)
        os.makedirs(path)
    # Write new config file to sim dir
    # Make a new configuration object for this specific sim
    sim_conf = confp.SafeConfigParser()
    sim_conf.optionxform = str
    for section in ['General','Simulation','Parameters','Materials']:
        sim_conf.add_section(section)
    # Add the general stuff for the sim
    for section in ['General','Simulation']:
        for item, val in conf.items(section):
            sim_conf.set(section,item,val)
    sim_conf.set('General','sim_dir',path)
    # Add all the fixed parameters from global conf
    for name,param in conf.items('Fixed Parameters'):
        sim_conf.set('Parameters',name,str(param))
    # Now just add the materials
    for name,matpath in conf.items('Materials'):
        sim_conf.set('Materials',name,matpath)
    # Now write the config file to the data subdir
    with open(os.path.join(path,'sim_conf.ini'),'w+') as new_conf:
        sim_conf.write(new_conf)
    # Copy sim script to sim dir
    base_script = sim_conf.get('General','sim_script')
    script = os.path.join(path,os.path.basename(base_script))
    shutil.copy(base_script,script)
    return (path,sim_conf)
    
def get_combos(tuplist):
    log = logging.getLogger('sim_wrapper') 
    log.info("Constructing dictionary of options and their values ...")
    # Get our variable parameters by splitting string in conf file and explanding into list
    optionValues = OrderedDict(tuplist)
    log.debug("Option values dict before processing: %s",str(optionValues))
    for key, value in optionValues.items():
        # determine if we have floats or ints
        if value.find('.') != -1 or value.find('E') != -1 or value.find('e') != -1:
            type_converter = lambda x: float(x)
        else:
            type_converter = lambda x: int(x)
        
        # Now we determine if we have a range or if we have actually included the values
        if value.find(':') != -1:
            # First find whether we want number of steps or size of steps
            if value.find(';') != -1:
                data,itertype = value.split(';')
            else:
                raise ValueError('You need to specify an iteration type in all your variable \
                parameters')

            # Parse the range string
            dataRange = data.split(':')
            dataMin,dataMax,dataIter = list(map(type_converter,dataRange))
            if itertype == 'numsteps':
                vals = np.linspace(dataMin,dataMax,dataIter)
            elif itertype == 'stepsize':
                # construct the option list (handles ints and floats)
                vals = [dataMin]
                while vals[-1] < dataMax:
                    vals.append(vals[-1]+dataIter)
            else:
                raise ValueError('Incorrent iteration type specified in one of your variable \
                parameters')

            # assign the option list
            optionValues[key] = vals
        else:
            # store the typed out values
            dataList = value.split(',')
            dataList = list(map(type_converter,dataList))
            optionValues[key] = dataList
    log.debug("Option values dict after processing: %s"%str(optionValues))
    valuelist = list(optionValues.values())
    keys = list(optionValues.keys())
    # Consuming a list of lists/tuples where each inner list/tuple contains all the values for a
    # particular parameter, returns a list of tuples containing all the unique combos for that
    # set of parameters 
    combos=list(itertools.product(*valuelist))
    log.debug('The list of parameter combos: %s',str(combos))
    return keys,combos 

def make_leaves(nodes):
    """Accepts a list of tuples, where the first element of the tuple is the full path to the node
    under which the leaves will be created, and the second element is un-finalized config object for
    that node"""
    log = logging.getLogger('sim_wrapper') 
    log.info("Building all leaves for each node...")
    # Get a list of the parameters names (keys) and a list of lists of values where the ith value 
    # of the inner lists corresponds to the ith param name in keys. The inner list consists of 
    # a unique combination of variable parameters. Note all nodes sweep through the same parameters
    # so we only need to compute this once. 
    keys,combos = get_combos(nodes[0][1].items("Variable Parameters"))
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
            # Make a new configuration object for this specific sim
            sim_conf = confp.ConfigParser(interpolation=None)
            sim_conf.optionxform = str
            for section in ['General','Simulation','Parameters','Materials']:
                sim_conf.add_section(section)
            # Add the general stuff for the sim
            for section in ['General','Simulation']:
                for item, val in conf.items(section):
                    sim_conf.set(section,item,val)
            sim_conf.set('General','sim_dir',fullpath)
            # Add all the fixed parameters from global conf
            for name,param in conf.items('Fixed Parameters'):
                sim_conf.set('Parameters',name,str(param))
            # Now add this specific combo of variable params. Note that itertools.product is so
            # wonderful and nice that it preserves the order of the values in every combo such that the
            # combo values always line up with the proper parameter name
            for i in range(len(combo)):
                sim_conf.set('Parameters',keys[i],str(combo[i]))
            # Now that all the parameters have been defined, we can add the evaluated parameters that
            # depend on the values of some other parameters
            for par, val in conf.items('Evaluated Parameters'):
                # Wrap the value in back ticks so we know to evaluate the results of the interpolated
                # expression in the simulation script
                wrapped = '`'+str(val)+'`'
                sim_conf.set('Parameters',par,wrapped)
            # Now just add the materials
            for name,path in conf.items('Materials'):
                sim_conf.set('Materials',name,path)
            # Now write the config file to the appropriate subdir
            with open(os.path.join(fullpath,'sim_conf.ini'),'w+') as new_conf:
                sim_conf.write(new_conf)
            base_script = sim_conf.get('General','sim_script')
            script = os.path.join(nodepath,os.path.basename(base_script))
            if not os.path.exists(script):
                shutil.copy(base_script,script)
            leaves.append((fullpath,sim_conf))
    log.debug('Here are the leaves: %s',str(leaves))
    log.info('Finished building leaves!')
    return leaves


def make_nodes(conf):
    # Get access to logger
    log = logging.getLogger('sim_wrapper')
    log.info("Constructing all nodes in sorted directory tree ...")
    opts = conf.items("Sorting Parameters")
    log.debug('Opts before sorting: %s',str(opts))
    # TODO: This should really be in pre_check()
    if not all([len(x[1].split(';')[-1].split(',')) == 2 for x in opts]):
        raise ValueError("You forgot to add a sorting key to one of your sorting parameters")
    # Sort options by key in config file
    sort_opts = sorted(opts,key = lambda tup: tup[-1].split(';')[-1].split(',')[1])
    log.debug('Opts after sorting: %s',str(sort_opts))
    # Done with the sorting keys so we can discard them
    sort_opts = [(tup[0],tup[1][0:tup[1].rfind(',')]) for tup in sort_opts]
    # Get a list of the parameters names (keys) and a list of lists of values where the ith value 
    # of the inner lists corresponds to the ith param name in keys. The inner list consists of 
    # a unique combination of variable parameters
    keys,combos = get_combos(sort_opts)
    # Make all the nodes in the directory tree
    nodes = []
    for combo in combos:
        # Copy global config object
        sub_conf = copy_conf_obj(conf)
        # Build path and add params to new config object
        path = conf.get('General','basedir')
        for i in range(len(combo)):
            sub_conf.set('Fixed Parameters',keys[i],str(combo[i]))
            sub_conf.remove_option("Sorting Parameters",keys[i])
            subdir = '{}_{:G}'.format(keys[i],combo[i])
            path = os.path.join(path,subdir)
            #if isinstance(combo[i],float) and combo[i] >= 10000:
            #    subdir = '{}_{:.4E}'.format(keys[i],combo[i])
            #    path = os.path.join(path,subdir)
            #else:
            #    subdir = '{}_{:.2f}'.format(keys[i],combo[i])
            #    path = os.path.join(path,subdir)
            try:
                os.makedirs(path)
            except OSError:
                pass
            log.info("Created node %s",path)
            sub_conf.set('General','basedir',path)
            if not i == len(combo)-1:
                with open(os.path.join(path,'sorted_sweep_conf_%s.ini'%keys[i]),'w') as conf_file:
                    sub_conf.write(conf_file) 
            else:
                # Remove sorting param section
                sub_conf.remove_section('Sorting Parameters')
                with open(os.path.join(path,'sorted_sweep_conf.ini'),'w') as conf_file:
                    sub_conf.write(conf_file)
        nodes.append((path,sub_conf))
    log.debug('Here are the nodes: %s',str(nodes))
    log.info('Finished building nodes!')
    leaves = make_leaves(nodes)
    return leaves

def convert_data(path,conf):
    """Converts text file spit out by S4 into npz format for faster loading during postprocessing"""
    # Have we decided to ignore and remove H field files?
    ignore = conf.getboolean('General','ignore_h')
    # Convert e field data files
    bname = conf.get('General','base_name')
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
    containing the absolute path to the job directory as the first element and the configuration
    object for the job as the second element"""

    log = logging.getLogger('sim_wrapper') 
    jobpath,jobconf = jobtup
    timed = jobconf.getboolean('General','save_time')
    tout = os.path.join(jobpath,'timing.dat')
    simlog = os.path.join(jobpath,'sim.log')
    script = jobconf.get('General','sim_script')
    ini_file = os.path.join(jobpath,'sim_conf.ini')
    if timed:
        log.debug('Executing script with timing wrapper ...')
        cmd = 'command time -vv -o %s lua %s %s 2>&1 | tee %s'%(tout,script,ini_file,simlog)
        #cmd = '/usr/bin/perf stat -o timing.dat -r 5 /usr/bin/lua %s %s'%(script,ini_file)
    else:
        cmd = 'command lua %s %s 2>&1 | tee %s'%(script,ini_file,simlog)
    log.debug('Subprocess command: %s',cmd)
    relpath = os.path.relpath(jobpath,jobconf.get('General','treebase'))
    log.info("Starting simulation for %s ...",relpath)
    completed = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #log.info('Simulation stderr: %s',completed.stderr)
    log.info("Simulation stderr: %s",completed.stderr)
    log.debug('Simulation stdout: %s',completed.stdout)
    # Convert text data files to npz binary files
    if jobconf.get('General','save_as') == 'npz':
        log.info('Converting data at %s to npz format',relpath)
        convert_data(jobpath,jobconf)
    log.info("Finished simulation for %s!",relpath)

def execute_jobs(gconf,jobs):
    """Given a list of length 2 tuples containing job directories and their configuration objects
    (in that order), executes the jobs. If specified, a multiprocessing pool is used to run the 
    jobs in parallel. If not, jobs are executed serially"""
    
    log=logging.getLogger('sim_wrapper')
    if not gconf.getboolean('General','parallel'):
        log.info('Executing jobs serially')
        for job in jobs:
            run_sim(job)
    else:
        num_procs = mp.cpu_count() - gconf.getint('General','reserved_cores')
        log.info('Executing jobs in parallel using %s cores ...',str(num_procs))
        with mp.Pool(processes=num_procs) as pool:
            result = pool.map(run_sim,jobs)

def spectral_wrapper(opt_pars,baseconf,opt_keys):
    """A wrapper function to handle spectral sweeps and postprocessing for the scipy minimizer. It
    accepts the initial guess as a vector, the base config file, and the keys that map to the
    initial guesses. It runs a spectral sweep (in parallel if specified), postprocesses the results,
    and returns the spectrally weighted reflection"""
    # TODO: Pass in the quantity we want to optimize as a parameter, then compute and return that
    # instead of just assuming reflection
    log=logging.getLogger('sim_wrapper')
    # Make sure we don't have any preexisting frequency dirs. We do this first so on the last
    # iteration all our data is left intact
    log.info('Param keys: %s'%str(opt_keys))
    log.info('Current values %s'%str(opt_pars))
    basedir = baseconf.get('General','basedir')
    globstr = os.path.join(basedir,'frequency*')
    dirs = glob.glob(globstr)
    for adir in dirs:
        shutil.rmtree(adir)
    # Set the parameters we are optimizing over to the current guess
    for i in range(len(opt_keys)):
        baseconf.set('Fixed Parameters',opt_keys[i],str(opt_pars[i]))
        baseconf.remove_option('Variable Parameters',opt_keys[i])
    # With the parameters set, we can now make the leaf directories, which should only contain
    # frequency values
    basepath = baseconf.get('General','basedir')
    node = (basepath,baseconf)
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
                sim_conf = parse_file(conf_path)
                rel = os.path.relpath(conf_path,basedir)
                log.info('Simulation at %s is %s at %s basis terms'%(rel,conv_status,numbasis))
                # Turn off adaptive convergence for all the sims that have converged and set their number of
                # basis terms to the proper value so we continue to use that value
                if conv_status == 'converged':
                    sim_conf.set('General','adaptive_convergence','False')
                    sim_conf.set('Parameters','numbasis',numbasis)
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
                    sim_conf.set('Parameters','numbasis',numbasis)
                    with open(conf_path,'w') as configfile:
                        sim_conf.write(configfile)
    # With the leaf directories made and the number of basis terms adjusted, we can now kick off our frequency sweep
    execute_jobs(baseconf,leaves)
    # With our frequency sweep done, we now need to postprocess the results. 
    # Configure logger
    logger = configure_logger('error','postprocess',
                              os.path.join(baseconf.get('General','basedir'),'logs'),
                              'postprocess.log')
    # Compute transmission data for each individual sim
    cruncher = pp.Cruncher(baseconf)
    cruncher.collect_sims()
    #print(len(cruncher.sims))
    for sim in cruncher.sims:
        cruncher.transmissionData(sim)
    # Now get the fraction of photons absorbed
    gcruncher = pp.Global_Cruncher(baseconf,cruncher.sims,cruncher.sim_groups,cruncher.failed_sims)
    #print(len(gcruncher.sim_groups[0]))
    #print(len(gcruncher.sim_groups))
    photon_fraction = gcruncher.Jsc()[0] 
    #print(opt_keys)
    #print(opt_pars)
    #print('Reflection value is: %f'%ref)
    # Lets store information we discovered from our adaptive convergence procedure so we can resue
    # it in the next iteration.
    if baseconf.getboolean('General','adaptive_convergence'):
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
    # Find the name of the parameters we are optimizing over, get the initial guesses for them, and
    # carry over only a sweep through frequency
    keys = []
    guess = np.array([])
    for name,param in conf.items('Variable Parameters'):
        if not param:
            guess_val = float(input("What is your guess for %s?: "%name))
            keys.append(name)
            guess = np.append(guess,guess_val)
        elif param and not name == 'frequency':
            log.error('You should only be sweeping through frequency')
            quit()
    # Make sure we don't have any sorted parameters
    if conf.items('Sorting Parameters'):
        log.error('You should not have any sorting parameters during an optimization')
        quit()
    else:
        conf.remove_section('Sorting Parameters')
    # Default tolerance in parameters is .0001, which is to the 10ths of nanometers. Pretty sure we
    # can't design nanowires with subnanometer precision so this tolerance should be sufficient
    # Same tolerance for the function value (fraction of absorbed photons)
    opt_val = optz.minimize(spectral_wrapper,
                            guess,
                            args=(conf,keys),
                            method='Nelder-Mead',
                            options={'maxiter':200,'disp':True})
    log.info(opt_val.message)
    log.info('Optimal values')
    log.info(keys)
    log.info(opt_val.x)
    return opt_val.x

def run(conf,log):
    basedir = conf.get('General','basedir')
    logdir = os.path.join(basedir,'logs')
    # Configure logger
    logger = configure_logger(log,'sim_wrapper',logdir,'sim_wrapper.log')
    if not conf.options("Variable Parameters"):
        # No variable params, single sim
        logger.debug('Entering single sim function from main')
        job = make_single_sim(conf)
        run_sim(job)
    # If all the variable params have ranges specified, do a parameter sweep
    elif all(list(zip(*conf.items("Variable Parameters")))[1]):
        if conf.items('Sorting Parameters'):
            logger.debug('Creating nodes and leaves')
            jobs = make_nodes(conf)
        else:
            logger.debug('Creating leaves')
            # make_leaves expects a list of tuples with contents (jobpath,jobconfobject)
            jobs = make_leaves([(basedir,conf)])
        execute_jobs(conf,jobs)
    # If we have specified variable params without ranges, we need to optimize them
    elif not all(list(zip(*conf.items("Variable Parameters")))[1]):
        logger.debug('Entering optimization function from main')
        run_optimization(conf)
    else:
        logger.error('Unsupported configuration for a simulation run. Not a single sim, sweep, or \
        optimization')

    # Need to fix this cuz its currently broken
    #if conf.getboolean('General','postprocess') == True:
    #    os.chdir(conf.get('General','basedir'))
    #    os.chdir('../')
    #    out,err = sh('python3 ./postprocess.py setup.ini')
    #    print(err) 

def pre_check(conf_path,conf):
    """Checks conf file for invalid values"""
    if not os.path.isfile(conf.get('General','sim_script')):
        print('You need to change the sim_script entry in the [General] section of your config \
        file')
        quit()
    
    # Warn user if they are about to dump a bunch of simulation data and directories into a
    # directory that already exists
    base = conf.get("General","basedir")
    #if os.path.isdir(base):
    #    print('WARNING!!! You are about to start a simulation in a directory that already exists')
    #    print('WARNING!!! This will dump a whole bunch of crap into that directory and possibly')
    #    print('WARNING!!! overwrite old simulation data.')
    #    ans = input('Would you like to continue? CTRL-C to exit, any other key to continue: ')
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
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    pre_check(os.path.abspath(args.config_file),conf)
    run(conf,args.log_level)

if __name__ == '__main__':
    main()
