import numpy as np
import pandas
import os
import glob


def convert_data(path,bname):
    """Converts text file spit out by S4 into npz format for faster loading during postprocessing"""
    # Convert both data files
    efile = os.path.join(path,bname+'.E')
    hfile = os.path.join(path,bname+'.H')
    d = pandas.read_csv(efile,delim_whitespace=True,header=None,skip_blank_lines=True)
    edata = d.as_matrix()
    econv = efile+'.raw'
    np.savez(econv,data=edata,headers=[None])
    d = pandas.read_csv(hfile,delim_whitespace=True,header=None,skip_blank_lines=True)
    hdata = d.as_matrix()
    hconv = hfile+'.raw'
    np.savez(hconv,data=hdata,headers=[None])
    # Remove the old text files
    os.remove(efile)
    os.remove(hfile)
    return None


path = ''
bname = 'field_data'
convert_data(path,bname)
