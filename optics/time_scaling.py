import glob
import re
import os
import datetime
import argparse as ap
import matplotlib
# Enables saving plots over ssh
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter, methodcaller

def extract_time(path):
    # Get file contents
    with open(path,'r') as f:
        lines = f.readlines()
    # The line of interest is the 4th one
    for line in lines:
        if '(wall clock)' in line:
            data = line.strip('\n').strip()
            break
    print(data)
    m = re.search('([0-9]+:[0-9]+:[0-9]+[\.]?[0-9]*)|([0-9]*:[0-9]+[\.]?[0-9]*)',data)
    # Matched h:mm:ss.mm
    if m.group(1):
        string = m.group(0)
        m = re.search('([0-9]*):([0-9]*):([0-9]*)(\.?[0-9]*)',string)
        hours,minutes,seconds,mseconds = m.group(1),m.group(2),m.group(3),m.group(4).strip('.') 
        kwargs = {'hours':float(hours),'minutes':float(minutes),'seconds':float(seconds),
                  'milliseconds':float(mseconds)}
    # Matched m:ss.mm
    else:
        string = m.group(2)
        m = re.search('([0-9]*):([0-9]*)(\.?[0-9]*)',string)
        minutes,seconds,mseconds = m.group(1),m.group(2),m.group(3).strip('.')
        kwargs = {'minutes':float(minutes),'seconds':float(seconds),
                  'milliseconds':float(mseconds)}
    tot_time = datetime.timedelta(**kwargs)
    
    return tot_time.total_seconds()

def analyze(path,show):
    # Get all error files via recursive globbing
    glob_str = os.path.join(path,'**/timing.dat')
    time_files = glob.glob(glob_str,recursive=True)
    data_pairs = []
    for f in time_files:
        # Get the parameter value from the directory name
        dirpath = os.path.dirname(f)
        param_dir = os.path.split(dirpath)[-1]
        m = re.search('[0-9]*\.[0-9]*[eE]?[-+]?[0-9]*',param_dir)
        x_val = float(m.group(0))
        # Now parse the timing file to extract total run time
        time = extract_time(f)
        data_pairs.append((x_val,time))
    m = re.search('frequency_([^/]*)',path)
    freq = m.group(1)
    # Sort parameters
    data_pairs.sort(key=itemgetter(0))
    out = os.path.join(path,'time_scaling.dat')
    with open(out,'w') as outf:
        outf.write('basis_terms,time (seconds)\n')
        for pair in data_pairs:
            outf.write('%i,%f\n'%(int(pair[0]),int(pair[1]))) 
    basis_terms,time = zip(*data_pairs)
    plt.figure()
    plt.plot(basis_terms,time,'b-o')
    plt.xlabel('Number of Basis Terms')
    plt.ylabel('Time (seconds)')
    plt.title('Time Scaling for Frequency = %s'%freq)
    if show:
        plt.show()
    plt.savefig(os.path.join(path,'time_scaling.pdf'))

def main():
    parser = ap.ArgumentParser(description="""Plots the minimum required basis terms to achieve a
    desired accuracy against an independent parameter""")
    parser.add_argument('path',type=str,help="Base directory path for the sorted sweep")
    parser.add_argument('-s','--show',action='store_true',default=False,help="""Show plots
            interactively""")
    args = parser.parse_args()
    
    path = args.path
    if not os.path.isdir(path):
        print("Specified path does not exist")
        quit()

    excludes = [os.path.join(path,'comp_struct')]
    print('Beginning global time scaling analysis for %s'%path)
    for root,dirs,files in os.walk(path,topdown=False):
        # If we haven't already analyzed this node in the directory tree
        if os.path.split(root)[0] not in excludes:
            # Find all error files in current directory
            if 'timing.dat' in files:
                # If we found an time file, go up one directory and perform the analysis
                base = os.path.split(root)[0]
                print('Computing time scaling for subdir %s'%base)
                analyze(base,args.show)
                # Add the level above the basis term sweep to the excludes list so we don't perform
                # the analysis for every basis term dir we find an error file in
                excludes.append(base)

if __name__ == '__main__':
    main()
