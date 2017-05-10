import os
import glob
import argparse
import configparser as confp
import ruamel.yaml as yaml


def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.ConfigParser(interpolation=None)
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

def fix_ini(confs):

    for confpath in confs:
        print('Modifying config at %s'%confpath)
        conf = parse_file(confpath)
        try:
            period = conf.getfloat('Parameters','array_period')
        except confp.NoSectionError:
            print("Path is not a sim config")
            continue
        val = str(int(period*500))
        conf.set(args.section,args.key,val)
        with open(confpath,'w') as configfile:
            conf.write(configfile)

def parse_yaml(path):
    """Parse the YAML file provided at the command line"""

    with open(path,'r') as cfile:
        text = cfile.read()
    conf = yaml.load(text,Loader=yaml.Loader)
    return conf

def fix_yaml(confs):

    for confpath in confs:
        print('Modifying config at %s'%confpath)
        conf = parse_yaml(confpath)
        conf['General']['ignore_h'] = True
        with open(confpath,'w') as configfile:
            configfile.write(yaml.dump(conf))

def main():
    parser = argparse.ArgumentParser(description="""Add a missing option to all the config files
    belows a node""")
    parser.add_argument('node',type=str,help="Path to node")
#    parser.add_argument('--key',required=True,type=str,help="Key name to be added to config file")
#    #parser.add_argument('--val',required=True,type=str,help="Value of key")
#    parser.add_argument('--section',required=True,type=str,help="""Section in which to add key value
#            pair""")
    parser.add_argument('--type',required=True,choices=['ini','yaml'],type=str,help="""Type of config file""")
    args = parser.parse_args()
   
    if not os.path.isdir(args.node):
        print("Node doesn't exist")
        quit()

    if args.type == 'ini':
        confs = glob.glob(os.path.join(args.node,'**/*.ini'),recursive=True)
        fix_ini(confs)
    else:
        confs = glob.glob(os.path.join(args.node,'**/sim_conf.yml'), recursive=True)
        fix_yaml(confs)


if __name__ == '__main__':
    main()
