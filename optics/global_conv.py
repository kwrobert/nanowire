import glob
import re
import os
import argparse as ap
import matplotlib
# Enables saving plots over ssh
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter, methodcaller

def analyze(path,thresh,local,show):
    # Get all error files via recursive globbing
    if local:
        glob_str = os.path.join(path,'**/localerror_*.dat')
    else:
        glob_str = os.path.join(path,'**/globalerror_*.dat')
    err_files = glob.glob(glob_str,recursive=True)
   
    # Get the first number of basis terms that is within our
    # error bound
    data_pairs = []
    for f in err_files:
        # Get the parameter value from the directory name
        dirpath = os.path.dirname(f)
        param_dir = os.path.split(dirpath)[-1]
        m = re.search('[0-9]*\.[0-9]*[eE]?[-+]?[0-9]*',param_dir)
        x_val = float(m.group(0))
        # Now find the first number of basis terms that is within our error threshold
        # and store True because its converged
        with open(f,'r') as err:
            tup = None
            lines = err.readlines()
            for line in lines:
                data = line.split(',')
                if float(data[-1]) < thresh:
                    tup = (x_val,int(data[0]),True)
                    data_pairs.append(tup)
                    break
            # If the error is never within the threshold, use highest available # of terms
            # and store False because it is not converged
            if not tup:
                data = lines[-1].split(',') 
                data_pairs.append((x_val,int(data[0]),False))

    # Sort parameters
    data_pairs.sort(key=itemgetter(0))
    # Write out to file
    if local:
        out = 'minimum_basis_terms_local_t%s.dat'%str(thresh)
    else:
        out = 'minimum_basis_terms_global_t%s.dat'%str(thresh)

    with open(os.path.join(path,out),'w') as minf:
        minf.write('frequency,numterms\n')
        for pair in data_pairs:
            minf.write('%E,%i\n'%(pair[0],int(pair[1]))) 
    x_vals,min_terms,converged = zip(*data_pairs)
    conv_x = [x_vals[i] for i in range(len(x_vals)) if converged[i]]
    conv_minterms = [min_terms[i] for i in range(len(x_vals)) if converged[i]]
    noconv_x = [x_vals[i] for i in range(len(x_vals)) if not converged[i]]
    noconv_minterms = [min_terms[i] for i in range(len(x_vals)) if not converged[i]]
    # Plot
    plt.figure()
    plt.plot(x_vals,min_terms,'b-')
    plt.plot(conv_x,conv_minterms,'bo',label='Converged')
    plt.plot(noconv_x,noconv_minterms,'ro',label='Not Converged')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Number of Fourier Terms')
    plt.title('Lower Bound on Basis Terms,Threshold = %s'%str(thresh))
    xlow,xhigh,ylow,yhigh = plt.axis()
    plt.ylim(ylow-10,yhigh+10)
    plt.legend(loc='best')
    if local:
        plt.savefig(os.path.join(path,'minimum_basis_terms_localerror_t%s.pdf'%str(thresh)))
    else:
        plt.savefig(os.path.join(path,'minimum_basis_terms_globalerror_t%s.pdf'%str(thresh)))
    if show:
        plt.show()   

def main():
    parser = ap.ArgumentParser(description="""Plots the minimum required basis terms to achieve a
    desired accuracy against an independent parameter""")
    parser.add_argument('path',type=str,help="Base directory path for the sorted sweep")
    parser.add_argument('-t','--threshold',type=float,default=.01,help="Maximum error threshold")
    parser.add_argument('-l','--local',action='store_true',default=False,help="""Use local error""")
    parser.add_argument('-s','--show',action='store_true',default=False,help="""Show plots
            interactively""")
    args = parser.parse_args()
    
    path = args.path
    if not os.path.isdir(path):
        print("Specified path does not exist")
        quit()
    
    if args.local:
        print('Using local error')
        file_reg = 'localerror_[a-zA-Z]+\.dat'
    else:
        print('Assuming global error')
        file_reg = 'globalerror_[a-zA-Z]+\.dat'

    excludes = [os.path.join(path,'comp_struct')]
    for root,dirs,files in os.walk(path,topdown=False):
        # If we haven't already analyzed this node in the directory tree
        if os.path.split(root)[0] not in excludes:
            # Find all error files in current directory
            err_files = [m for f in files for m in [re.search(file_reg,f)] if m]
            # If we found an error file, go up one directory and perform the analysis
            if err_files:
                base = os.path.split(root)[0]
                analyze(base,args.threshold,args.local,args.show)
                # Add the level above the basis term sweep to the excludes list so we don't perform
                # the analysis for every basis term dir we find an error file in
                excludes.append(base)

if __name__ == '__main__':
    main()
