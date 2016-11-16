import os
import glob
import argparse as ap
import shutil as sh

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
        data = [('{:.4E}'.format(float(line.split(',')[0])),str(line.split(',')[1].strip('\n'))) for line in f.readlines()[1:]]
    print(data)
    
    dir_glob = os.path.join(s4_dir,"frequency*")
    freq_dirs = glob.glob(dir_glob)
    for fdir in freq_dirs:
        for freq, numbasis in data:
            if freq in fdir:
                print('Frequency {} found in directory {}'.format(freq,fdir))
                basis_path = os.path.join(fdir,'numbasis_{}.0000'.format(numbasis))
                if os.path.isdir(basis_path):
                    print('Found {}'.format(basis_path))
                    new_path = os.path.join(dest_dir,os.path.basename(fdir))
                    print('Copying {} to {}'.format(basis_path,new_path))
                    sh.copytree(basis_path,new_path)
                else:
                    print('Missing {} !!!!'.format(basis_path))

if __name__ == '__main__':
    main()
