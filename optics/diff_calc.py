import numpy as np
import argparse as ap
import pandas 
import os

def diff_sq(x,y):
    """Returns the magnitude of the difference vector squared between two vector fields at each
    point in space"""
    if x.size != y.size:
        self.log.error("You have attempted to compare datasets with an unequal number of points!!!!")
        quit()
    # Calculate the magnitude of the difference vector SQUARED at each point in space
    # This is mag(vec(x) - vec(y))^2 at each point in space. This should be a 1D array
    # with # of elements = # sampling points
    mag_diff_vec = np.sum((x-y)**2,axis=1)
    return mag_diff_vec

def main():
    
    parser = ap.ArgumentParser(description="""Simple script to compare two vector fields so we can
    avoid doing it in Lua, which is painfully slow""")
    parser.add_argument('path1',help="Path to first data file")
    parser.add_argument('path2',help="Path to second data file")
    parser.add_argument('--start',type=int,default=0,help="Starting index for the comparison slice")
    parser.add_argument('--end',type=int,default=-1,help="Final index for comparison slice")
    args = parser.parse_args()

    # Load in the text files
    d1 = pandas.read_csv(args.path1,delim_whitespace=True,header=None,skip_blank_lines=True)
    data1 = d1.as_matrix()
    d2 = pandas.read_csv(args.path2,delim_whitespace=True,header=None,skip_blank_lines=True)
    data2 = d2.as_matrix()
    # Get the slices
    data1 = data1[args.start:args.end,3:9]
    data2 = data2[args.start:args.end,3:9]
    # Do the actual calculation 
    mag_diff_vec = diff_sq(data1,data2)
    # We need to get our normalization vector
    normvec = np.sum(data1**2,axis=1)
    error = np.sqrt(np.sum(mag_diff_vec)/np.sum(normvec))
    # The lua script we are calling this from expects the error on stdout, so just print it
    print(error)

if __name__ == '__main__':
    main()
