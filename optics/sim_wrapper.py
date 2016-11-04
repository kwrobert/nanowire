import shutil
import subprocess
import itertools
import argparse as ap
import os
import configparser as confp 
import glob
import logging
import datetime
from multiprocessing import cpu_count 

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

def sh(cmd):
    return subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

def start_sim(script,ini_file):
    '''Executes commands directly to the native bash shell using subprocess.Popen, and retrieves
    stdout and stderr. 
    
    !!!IMPORTANT NOTE!!!: Passing unsanitized input into this function can be a huge
    security threat. Tread carefully. 
    !!!IMPORTANT NOTE!!!: Passing in a command that generates excessive output might
    freeze the program depending on buffer sizes
    Useage: out, err = sh(bash_command)
    Input: - bash_command: string
    Output: - out: string
            - err: string'''
    log = logging.getLogger('sim_wrapper') 
    log.info('Hit core limit, polling processes ...')
    cmd = '/usr/bin/lua %s %s'%(script,ini_file)
    log.debug('Subprocess command: %s',cmd)
    return subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def poll_procs(procs):
    log = logging.getLogger('sim_wrapper') 
    log.info('Hit core limit, polling processes ...')
    # While there are running processes
    while procs:
        # True if the process finished, false if not. This way we only actually poll processes once,
        # polling seperately for finished and running procs might cause inconsistencies 
        flags = [True if proc[0].poll() != None else False for proc in procs]
        # The processes which have terminated 
        fin_procs = [procs[i] for i in range(len(flags)) if flags[i]]
        # Running procs
        run_procs = [procs[i] for i in range(len(flags)) if not flags[i]]
        # Retrieve output from finished procs
        for proc in fin_procs:
            out,err = proc[0].communicate()
            log.debug('Simulation stdout for %s: %s',proc[1],str(out))
            log.info('Simulation stderr for %s: %s',proc[1],str(err))
            log.info("Finished simulation for %s!",str(proc[1]))
        procs = run_procs
    log.info('Batch complete!')
    return procs

def sim(script,ini_file):
    out,err = sh('python %s %s'%(script,ini_file))
    return out,err

def run_single_sim(conf):
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
    script = os.path.join(path,'s4_sim.py')
    if not os.path.exists(script):
        shutil.copyfile(conf.get('General','sim_script'),script)
    # Move into sim dir and run the sim
    os.chdir(path)
    workdir = os.path.basename(basedir)
    log.info("Starting simulation for %s ....",workdir)
    proc = start_sim(script,"sim_conf.ini")
    out,err = proc.communicate()
    log.debug('Simulation stdout: %s',out)
    log.info('Simulation stderr: %s',err)
    log.info("Finished simulation for %s!",str(workdir))

def run_sweep(conf):
    log = logging.getLogger('sim_wrapper') 
    log.info("Running parameter sweep ...")
    # Get our variable parameters by splitting string in conf file and explanding into list
    optionValues = dict(conf.items("Variable Parameters"))
    log.debug("Option values dict before processing: %s",str(optionValues))
    for key, value in optionValues.items():
        # determine if we have floats or ints
        if value.find('.') != -1 or value.find('E') != -1:
            type_converter = lambda x: float(x)
        else:
            type_converter = lambda x: int(x)
        
        # Now we determine if we have a range or if we have actually included the values
        if value.find(':') != -1:
            
            # Parse the range string
            dataRange = value.split(':')
            dataMin,dataMax,dataStep = list(map(type_converter,dataRange))

            # construct the option list (handles ints and floats)
            vals = [dataMin]
            while vals[-1] < dataMax:
                vals.append(vals[-1]+dataStep)

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
    # If we are running sims in parallel, need to store running Popen instances
    # If not running in parallel, this list will always be empty
    procs = []
    if conf.getboolean('General','parallel'):
        max_procs = cpu_count()-conf.getint('General','reserved_cores')
        log.info('Using %i cores ...'%max_procs)
    # Loop through each unique combination of parameters
    for combo in combos:
        # Make a unique working directory based on parameter combo
        basedir=conf.get('General','basedir')
        workdir=''
        for i in range(len(combo)):
            if isinstance(combo[i],float) and combo[i] >= 10000:
                substr = '{}_{:.4E}__'.format(keys[i],combo[i])
                workdir += substr 
            else:
                substr = '{}_{:.4f}__'.format(keys[i],combo[i])
                workdir += substr 
        workdir=workdir.rstrip('__')
        log.info('Preparing for simulation %s',str(workdir))
        fullpath = os.path.join(basedir,workdir)
        try:
            os.makedirs(fullpath)
        except OSError:
            print("Working directory already exists, appending new to name")
            workdir = workdir+"__new"
            fullpath = os.path.join(basedir,workdir)
            os.makedirs(fullpath)
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
       
        script = os.path.join(basedir,'s4_sim.py')
        if not os.path.exists(script):
            shutil.copyfile(sim_conf.get('General','sim_script'),script)
        os.chdir(fullpath)
        log.info("Starting simulation for %s ....",workdir)
        proc = start_sim(script,"sim_conf.ini")
        fpath = os.path.join(fullpath,sim_conf.get('General','base_name')+'.E')
        if not conf.getboolean('General','parallel'):
            out,err = proc.communicate()
            log.debug('Simulation stdout: %s',str(out))
            log.info('Simulation stderr: %s',str(err))
            log.info("Finished simulation for %s!",str(workdir))
        else:
            # The reason for not using multiprocessing is because each process outputs and
            # identically named file and multiprocessing pools would spit them all out into the same
            # directory
            if len(procs) < max_procs:
                procs.append((proc,os.path.relpath(fullpath,basedir),fpath))
            else:
                procs = poll_procs(procs)
        os.chdir(basedir)
    if procs:
        poll_procs(procs)

def run_sorted_sweep(conf):
    # Get access to logger
    log = logging.getLogger('sim_wrapper')
    log.info("Running sorted parameter sweep ...")
    opts = conf.items("Sorting Parameters")
    log.debug('Opts before sorting: %s',str(opts))
    # TODO: This should really be in pre_check()
    if not all([len(x[1].split(';')) == 2 for x in opts]):
        raise ValueError("You forgot to add a sorting key to one of your sorting parameters")
    sort_opts = sorted(opts,key = lambda tup: tup[-1].split(';')[-1])
    log.debug('Opts after sorting: %s',str(sort_opts))
    # Done with the sorting keys so we can discard them
    sort_opts = [(tup[0],tup[1][0:tup[1].find(';')]) for tup in sort_opts]
    
    optionValues = []
    for opt,values in sort_opts:
        # determine if we have floats or ints
        if values.find('.') != -1 or values.find('E') != -1:
            type_converter = lambda x: float(x)
        else:
            type_converter = lambda x: int(x)
        # Now we determine if we have a range or if we have actually included the values
        if values.find(':') != -1:
            
            # Parse the range string
            dataRange = values.split(':')
            dataMin,dataMax,dataStep = list(map(type_converter,dataRange))

            # construct the option list (handles ints and floats)
            vals = [dataMin]
            while vals[-1] < dataMax:
                vals.append(vals[-1]+dataStep)
            # assign the option list
            optionValues.append(vals)
        else:
            # store the typed out values
            vals = values.split(',')
            vals = list(map(type_converter,vals))
            optionValues.append(vals) 
    log.debug('Processed option values list: %s',str(optionValues))
    # Make all the leaf directories
    for combo in itertools.product(*optionValues):
        # Copy global config object
        sub_conf = copy_conf_obj(conf)
        # Remove sorting param section
        sub_conf.remove_section('Sorting Parameters')
        # Build path and add params to new config object
        path = conf.get('General','basedir')
        for i in range(len(combo)):
            sub_conf.set('Fixed Parameters',sort_opts[i][0],str(combo[i]))
            if isinstance(combo[i],float) and combo[i] >= 10000:
                subdir = '{}_{:.4E}'.format(sort_opts[i][0],combo[i])
                path = os.path.join(path,subdir)
            else:
                subdir = '{}_{:.2f}'.format(sort_opts[i][0],combo[i])
                path = os.path.join(path,subdir)
        try:
            os.makedirs(path)
        except OSError:
            pass
        log.info("Created subdir %s",path)
        sub_conf.set('General','basedir',path)
        with open(os.path.join(path,'sorted_sweep_conf.ini'),'w') as conf_file:
            sub_conf.write(conf_file) 
        log.info('Running variable param sweep in %s',path)
        run_sweep(sub_conf)

def run_optimization(conf):
    log = logging.getLogger('sim_wrapper')
    log.info("Running optimization")

def run(conf,log):
    # Configure logger
    logger = configure_logger(log,'sim_wrapper',
                              os.path.join(conf.get('General','basedir'),'logs'),
                              'sim_wrapper.log')
    
    if not conf.options("Variable Parameters"):
        # No variable params, single sim
        logger.debug('Entering single sim function from main')
        run_single_sim(conf)
    # If all the variable params have ranges specified, do a parameter sweep
    elif all(list(zip(*conf.items("Variable Parameters")))[1]):
        if conf.items('Sorting Parameters'):
            logger.debug('Entering sorted sweep function from main')
            run_sorted_sweep(conf)
        else:
            logger.debug('Entering sweep function from main')
            run_sweep(conf)
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
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    pre_check(os.path.abspath(args.config_file),conf)
    run(conf,args.log_level)

if __name__ == '__main__':
    main()
