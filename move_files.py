import os
import os.path as osp
import shutil
import glob
import re
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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_freq(path):
    with open(path,'r') as f:
        for i in range(0,10):
            line = f.readline()
    dat = line.split()
    freq = dat[3]
    return freq

def main():
    parser = ap.ArgumentParser(description="""Move COMSOL files with frequency equal to those present in a
    simulation config to a new directory""")
    parser.add_argument('comsol_dir',type=str,help="""The directory containing all the COMSOl data
    files""")
    parser.add_argument('conf',type=str,help="""The path to the COMSOL conf file""")
    parser.add_argument('-o','--output',type=str,default="keep_files",help="""The directory to put the
    matching COMSOL files in""")
    args = parser.parse_args()

    confpath = osp.abspath(args.conf)
    coms_dir = osp.abspath(args.comsol_dir)
    if not osp.isfile(confpath):
        print('S4 file doesnt exist')
        quit()
    if not osp.isdir(coms_dir):
        print('COMSOL dir doesnt exist')
        quit()
    try:
        os.mkdir(args.output)
    except FileExistsError:
        print('Output directory already exists')
        input('Continue?')

    cf = parse_file(confpath)    
    freqs = cf.get('Sorting Parameters','frequency').split(',')
    comsglob = osp.join(coms_dir,'frequency*')
    comsfiles = sorted(glob.glob(comsglob),key=natural_keys)
    freqs = []
    for f in comsfiles:
        freq = get_freq(f)
        freqs.append(freq)
    desired_freqs = freqs[0::2]
    print(desired_freqs)
    for dfreq in desired_freqs:
        for comsfile in comsfiles:
            freq = get_freq(comsfile)
            if freq == dfreq:
                found = True
                print("Found desired freq %s in file %s, copying now"%(dfreq,comsfile))
                shutil.copy(comsfile,args.output)
                break
        if not found:
            print('No comsol file with freq=%s'%str(dfreq))

if __name__ == '__main__':
    main()
