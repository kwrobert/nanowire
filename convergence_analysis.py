import os
import argparse as ap
import configparser as confp
import itertools 
import copy
import sim_wrapper

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
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
    new_config = confp.ConfigParser()
    new_config.optionxform = str
    new_config.read_file(config_string)
    return new_config

def build_dirs(conf):

    opts = conf.items("Sorting Parameters")
    if not all([len(x[1].split(';')) == 2 for x in opts]):
        print("You forgot to add a sorting key to one of your sorting parameters")
        quit()
    sort_opts = sorted(opts,key = lambda tup: tup[-1].split(';')[-1])
    

    print(sort_opts)
    # Done with the sorting keys so we can discard them
    sort_opts = [(tup[0],tup[1][0:tup[1].find(';')]) for tup in sort_opts]
    print(sort_opts)
    
    tree_base = conf.get('General','basedir')
    optionValues = []
    for opt,values in sort_opts:
        print(opt)
        print(values)
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
        sub_conf.set('General','basedir',path)
        with open(os.path.join(path,'sim_config.ini'),'w') as conf_file:
            sub_conf.write(conf_file) 
        sim_wrapper.run(sub_conf,'info') 
    quit()

def main():
    parser = ap.ArgumentParser(description="""Runs a convergence analysis but keeps
            directories properly organized""") 
    parser.add_argument('config_file',help="The configuration file with sorted parameters")
    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    if not conf.options('Sorting Parameters'):
        print('Your setup file needs to have a [Sorting Parameters] section')
        quit()

    build_dirs(conf)


if __name__ == '__main__':
    main()
