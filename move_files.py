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

def dir_keys(path):
    """A function to take a path, and return a list of all the numbers in the path. This is
    mainly used for sorting 
        by the parameters they contain"""

    regex = '[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?' # matching any floating point
    m = re.findall(regex, path)
    if(m): val = m
    else: raise ValueError('Your path does not contain any numbers')
    val = list(map(float,val))
    return val

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
    desired_freqs = cf.get('Sorting Parameters','frequency').split(',')
    desired_freqs[-1] = desired_freqs[-1][:-2]
    print(desired_freqs[-1])
    comsglob = osp.join(coms_dir,'frequency*')
    comsfiles = sorted(glob.glob(comsglob),key=dir_keys)
    freqs = []
    for f in comsfiles:
        freq = get_freq(f)
        freqs.append(freq)
    counter = 0
    for dfreq in desired_freqs:
        found = False
        print('Desired Frequency: %s'%str(dfreq))
        for comsfile in comsfiles:
            freq = get_freq(comsfile)
            print('COMSOL Frequency: %s'%str(freq))
            if freq == dfreq:
                found = True
                print('**************************************************************')
                print("Found desired freq %s in file %s, copying now"%(dfreq,comsfile))
                print('**************************************************************')
                shutil.copy(comsfile,args.output)
                counter += 1
                break
        if not found:
            print('**************************************************************')
            print('No comsol file with freq=%s'%str(dfreq))
            print('Seeking file with minimal difference')
            mindiff = 1E20
            minfile = None
            for comsfile in comsfiles:
                freq = get_freq(comsfile)
                diff = abs(float(dfreq)-float(freq))
                if diff < mindiff:
                    mindiff = diff
                    minfile = comsfile
                    minfreq = freq
            print('Using file %s with frequency %s'%(minfile,minfreq))
            print('**************************************************************')
            shutil.copy(minfile,args.output)
            counter += 1
    print('Number of COMSOL files moved: %i'%counter)

if __name__ == '__main__':
    main()
