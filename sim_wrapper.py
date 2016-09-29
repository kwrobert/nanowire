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
    
    parser = confp.SafeConfigParser()
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

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

    cmd = 'python %s %s'%(script,ini_file)
    return subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def poll_procs(procs):
    log = logging.getLogger('sim_wrapper') 
    log.info('Hit core limit, polling processes ...')
    # While there are running processes
    while procs:
        for proc in procs:
            if proc[0].poll() != None:
                out,err = proc[0].communicate()
                log.debug('Simulation stdout for %s: %s',proc[1],str(out))
                log.debug('Simulation stderr for %s: %s',proc[1],str(err))
                log.info("Finished simulation for %s!",str(proc[1]))
                procs.remove(proc)

def sim(script,ini_file):
    out,err = sh('python %s %s'%(script,ini_file))
    return out,err

def run_single_sim(conf,conf_path):
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
        shutil.copyfile('s4_sim.py',script)
    # Move into sim dir and run the sim
    os.chdir(path)
    workdir = os.path.basename(basedir)
    log.info("Starting simulation for %s ....",workdir)
    proc = start_sim(script,"sim_conf.ini")
    out,err = proc.communicate()
    log.debug('Simulation stdout: %s',out)
    log.debug('Simulation stderr: %s',err)
    log.info("Finished simulation for %s!",str(workdir))

def run_sweep(conf):
    log = logging.getLogger('sim_wrapper') 
    log.info("Running parameter sweep ...")
    # Get our variable parameters by splitting string in conf file and explanding into list
    optionValues = dict(conf.items("Variable Parameters"))
    log.debug("Option values dict before processing: %s",str(optionValues))
    for key, value in optionValues.items():
        # determine if we have floats or ints
        print(value)
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
    # parameter  
    combos=list(itertools.product(*valuelist))
    log.debug('The list of parameter combos: %s',str(combos))
    # If we are running sims in parallel, need to store running Popen instances
    # If not running in parallel, this list will always be empty
    procs = []
    num_procs = cpu_count()-conf.getint('General','reserved_cores')
    log.info('Using %i cores ...'%num_procs)
    # Loop through each unique combination of parameters
    for combo in combos:
        # Make a unique working directory based on parameter combo
        basedir=conf.get('General','basedir')
        workdir=''
        for i in range(len(combo)):
            if isinstance(combo[i],float) and combo[i] >= 10000:
                workdir += str(keys[i])+'%E__'%combo[i]
            else:
                workdir += str(keys[i])+str(combo[i])+'__'
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
        sim_conf = confp.SafeConfigParser()
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
        
        # Now just add the materials
        for name,path in conf.items('Materials'):
            sim_conf.set('Materials',name,path)

        # Now write the config file to the appropriate subdir
        with open(os.path.join(fullpath,'sim_conf.ini'),'w+') as new_conf:
            sim_conf.write(new_conf)
       
        script = os.path.join(basedir,'s4_sim.py')
        if not os.path.exists(script):
            shutil.copyfile('s4_sim.py',script)
        os.chdir(fullpath)
        log.info("Starting simulation for %s ....",workdir)
        proc = start_sim(script,"sim_conf.ini")
        if not conf.getboolean('General','parallel'):
            out,err = proc.communicate()
            log.debug('Simulation stdout: %s',str(out))
            log.debug('Simulation stderr: %s',str(err))
            log.info("Finished simulation for %s!",str(workdir))
        else:
            # The reason for not using multiprocessing is because each process outputs and
            # identically named file and multiprocessing pools would spit them all out into the same
            # directory
            if len(procs) < num_procs:
                procs.append((proc,workdir))
            else:
                poll_procs(procs)
        os.chdir(basedir)
    poll_procs(procs)
        
def run_optimization(conf):
    log = logging.getLogger('sim_wrapper')
    log.info("Running optimization")

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

    # Configure logger
    logger = configure_logger(args.log_level,'sim_wrapper',
                              os.path.join(conf.get('General','basedir'),'logs'),
                              'sim_wrapper.log')
    
    if not conf.options("Variable Parameters"):
        # No variable params, single sim
        logger.debug('Entering single sim function from main')
        run_single_sim(conf,os.path.abspath(args.config_file))
    # If all the variable params have ranges specified, do a parameter sweep
    elif all(list(zip(*conf.items("Variable Parameters")))[1]):
        logger.debug('Entering sweep function from main')
        run_sweep(conf)
    # If we have specified variable params without ranges, we need to optimize them
    elif not all(list(zip(*conf.items("Variable Parameters")))[1]):
        logger.debug('Entering optimization function from main')
        run_optimization(conf)
    else:
        logger.error('Unsupported configuration for a simulation run. Not a single sim, sweep, or \
        optimization')

    if conf.getboolean('General','postprocess') == True:
        os.chdir(conf.get('General','basedir'))
        os.chdir('../')
        out,err = sh('python3 ./postprocess.py setup.ini')
        print(err) 

if __name__ == '__main__':
    main()
