import shutil
import subprocess
import itertools
import argparse as ap
import os
import configparser as confp 
import glob

def sh(cmd):
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
    return subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

def sim(script,ini_file):
    out,err = sh('python %s %s'%(script,ini_file))
    return out,err

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

def run_single_sim():
    print("Running single sim")

def run_sweep(conf):
        
    print("Running sweep")
    # Get our variable parameters by splitting string in conf file and explanding into list
    optionValues = dict(conf.items("Variable Parameters"))
    print(optionValues)
    for key, value in optionValues.items():
        # determine if we have floats or ints
        if value.find('.') != -1:
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
    
    valuelist = list(optionValues.values())
    keys = list(optionValues.keys())
    # Consuming a list of lists/tuples where each inner list/tuple contains all the values for a
    # particular parameter, returns a list of tuples containing all the unique combos for that
    # parameter  
    combos=list(itertools.product(*valuelist))
    # Loop through each unique combination of parameters
    for combo in combos:
        # Make a unique working directory based on parameter combo
        basedir=conf.get('General','basedir')
        workdir=''
        for i in range(len(combo)):
            workdir += str(keys[i])+str(combo[i])+'__'
        workdir=workdir.rstrip('__')
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
        for section in ['General','Parameters','Materials']:
            sim_conf.add_section(section)
        # Add the general stuff for the sim
        for item, val in conf.items('General'):
            sim_conf.set('General',item,val)
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
        print("Starting simulation for %s ...."%workdir)
        out, err = sim(script,"sim_conf.ini")
        print(out)
        print(err)
        print("Finished simulation for %s!"%workdir)
        os.chdir(basedir)
            
def run_optimization(conf):
    print("Running optimization")

def main():

    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")

    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    if not conf.options("Variable Parameters"):
        # No variable params, single sim
        run_single_sim(conf)
    # If the variable params have ranges specified, do a parameter sweep
    elif all(list(zip(*conf.items("Variable Parameters")))[1]):
        run_sweep(conf)
    # If we have specified variable params without ranges, we need to optimize them
    elif not all(list(zip(*conf.items("Variable Parameters")))[1]):
        run_optimization(conf)
    else:
        print("Something isn't right")


if __name__ == '__main__':
    main()
