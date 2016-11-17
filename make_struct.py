import os
import glob
import argparse as ap
import shutil as sh
import re

def main():
    parser = ap.ArgumentParser(description="""Uses minimum basis term file to extract the data for a
    simulation that used the minimum number of basis terms for each frequency""")
    parser.add_argument('min_file',type=str,help="""Path to file containing frequency in first
    column and minimum number of basis terms in second column. Comma separated""")
    parser.add_argument('s4_dir',type=str,help="""Path to top level dir containing all the frequency
    subdirectories""")
    parser.add_argument('dest_dir',type=str,help="""Path to top level dir which all files will be
    moved to""")
    args = parser.parse_args()
    s4_dir = os.path.abspath(args.s4_dir)
    min_file = os.path.abspath(args.min_file)
    dest_dir = os.path.abspath(args.dest_dir)
    if not os.path.isdir(args.s4_dir):
        print("S4 dir does not exist")
        quit()
    if not os.path.isfile(min_file):
        print('Min file does not exists')
        quit()
    try:
        os.makedirs(dest_dir)
    except OSError:
        pass

    with open(min_file,'r') as f:
        data = [('{:G}'.format(float(line.split(',')[0])),str(line.split(',')[1].strip('\n'))) for line in f.readlines()[1:]]
    print(data)
    
    dir_glob = os.path.join(s4_dir,"frequency*")
    freq_dirs = glob.glob(dir_glob)
    for fdir in freq_dirs:
        for freq, numbasis in data:
            print('Frequency {} has minimum basis of {}'.format(freq,numbasis))
            dat = freq.split('E')
            regex = dat[0][0:4]+"[0-9]+E\\"+dat[1]
            regex = regex.replace('.','\.')
            m = re.search(regex,fdir)
            if m:
                print(m.group(0))
                print('Frequency {} found in directory {}'.format(freq,fdir))
                basis_path = os.path.join(fdir,'numbasis_{}'.format(numbasis))
                if os.path.isdir(basis_path):
                    print('Found min basis path {}'.format(basis_path))
                    new_path = os.path.join(dest_dir,os.path.basename(fdir))
                    print('Copying {} to {}'.format(basis_path,new_path))
                    sh.copytree(basis_path,new_path)
                else:
                    print('Missing {} !!!!'.format(basis_path))
                break

if __name__ == '__main__':
    main()
