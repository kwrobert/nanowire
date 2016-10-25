import glob
import re
import os
import argparse as ap
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter, methodcaller

def analyze(path,thresh):
    glob_str = os.path.join(path,'**/mse_*.dat')
    err_files = glob.glob(glob_str,recursive=True)
   
    # Get the first number of basis terms that is within our
    # error bound
    # NOTE: This does not handle multiple sorted parameters!!
    data_pairs = []
    for f in err_files:
        # Get the parameter value from the directory name
        dirpath = os.path.dirname(f)
        param_dir = os.path.split(dirpath)[-1]
        m = re.search('[0-9]*\.[0-9]*[eE]?[-+]?[0-9]*',param_dir)
        x_val = float(m.group(0))
        # Now find the first number of basis terms that is within our error threshold
        with open(f,'r') as err:
            for line in err.readlines():
                data = line.split(',')
                if float(data[-1]) < thresh:
                    tup = (x_val,int(data[0]))
                    data_pairs.append(tup)
                    break
    # Sort parameters
    data_pairs.sort(key=itemgetter(0))
    x_vals,min_terms = zip(*data_pairs)
    # Plot
    plt.figure()
    plt.plot(x_vals,min_terms)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Number of Fourier Terms')
    plt.title('Lower Bound on Basis Terms,Threshold = %s'%str(thresh))
    xlow,xhigh,ylow,yhigh = plt.axis()
    plt.ylim(ylow-10,yhigh+10)
    plt.savefig(os.path.join(path,'minimum_basis_terms_t%s.pdf'%str(thresh)))
    plt.show()   

def main():
    parser = ap.ArgumentParser(description="""Plots the minimum required basis terms to achieve a
    desired accuracy against an independent parameter""")
    parser.add_argument('path',type=str,help="Base directory path for the sorted sweep")
    parser.add_argument('-t','--threshold',type=float,default=.01,help="Maximum error threshold")
    args = parser.parse_args()
    
    path = args.path
    if not os.path.isdir(path):
        print("Specified path does not exist")

    excludes = []
    for root,dirs,files in os.walk(path,topdown=False):
        # If we haven't already analyzed this node in the directory tree
        if os.path.split(root)[0] not in excludes:
            # Find all error files in current directory
            err_files = [m for f in files for m in [re.search('mse_[a-zA-Z]+\.dat',f)] if m]
            # If we found an error file, go up one directory and perform the analysis
            if err_files:
                base = os.path.split(root)[0]
                analyze(base,args.threshold)
                # Add the level above the basis term sweep to the excludes list so we don't perform
                # the analysis for every basis term dir we find an error file in
                excludes.append(base)

if __name__ == '__main__':
    main()
