import glob
import os
import os.path as osp
import argparse as ap
import configparser as confp 

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
    # This preserves case sensitivity
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

def fix_simconf(c,tp):
    o = parse_file(c)
    new_p = osp.dirname(c)
    old_p = o.get('General','sim_dir')
    base_p = osp.dirname(new_p)
    if not osp.isfile(old_p):
        print('Replacing following paths:')
        print('Old: %s'%old_p)
        print('New: %s'%new_p)
        o.set('General','sim_dir',new_p)
        o.set('General','tree_base',tp)
        o.set('General','basedir',base_p)
        with open(c,'w') as conf_file:
            o.write(conf_file,space_around_delimiters=True)

def fix_sortedconf(c,tp):
    o = parse_file(c)
    new_p = osp.dirname(c)
    old_p = o.get('General','basedir')
    if not osp.isfile(old_p):
        print('Replacing following paths:')
        print('Old: %s'%old_p)
        print('New: %s'%new_p)
        o.set('General','tree_base',tp)
        o.set('General','basedir',new_p)
        with open(c,'w') as conf_file:
            o.write(conf_file,space_around_delimiters=True)

def main():

    parser = ap.ArgumentParser(description="""Fixes the configuration files so the path names in them
    actually match their current location""")
    parser.add_argument('path',type=str,help="""Path to any node in the tree that needs fixing""")
    parser.add_argument('tree_base',type=str,help="""Path to the base of the tree for this sweep""")
    args = parser.parse_args()

    p = osp.abspath(args.path)
    tp = osp.abspath(args.tree_base)
    if not osp.isdir(p):
        raise ValueError("Specified directory %s doesn't exist"%p)
    if not osp.isdir(tp):
        raise ValueError("Specified directory %s doesn't exist"%tp)
    
    print('Fixing all conf files beneath %s'%p)

    sim_confs = glob.glob(osp.join(p,'**/sim_conf.ini'),recursive=True)
    sorted_confs = glob.glob(osp.join(p,'**/sorted_*.ini'),recursive=True)
    for c in sim_confs:
        fix_simconf(c,tp)
    for c in sorted_confs:
        fix_sortedconf(c,tp)

if __name__ == '__main__':
    main()
