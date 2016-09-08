import numpy as np
import matplotlib
# Enables saving plots over ssh
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import argparse as ap
import os
import re

def comp_conv(val):
    if str(val).strip("b'") == 'NaN':
        return float(0)
    else:
        try:
            return float(val)
        except ValueError:
            val = str(val).strip("b'")
            val = val.replace('i','j')
            return complex(val)

def parse_comsol(path):
    conv = {col: comp_conv for col in range(3,11)}
    raw = np.loadtxt(path,comments='%',converters=conv,dtype=complex)

    # Fix this horrendous matrix

    # Make the positions real, convert to micrometers and shift z up so 
    # everything starts at z = 0
    data = np.zeros((raw.shape[0],17))
    data[:,0:3] = np.real(raw[:,0:3])*1E6
    data[:,2] = np.absolute(data[:,2] + 1) 
    # Split the complex values of all the field components. Dict maps col
    # of raw data to the two new columns in the cleaned up matrix
    mapping = {4:(3,4),5:(5,6),6:(7,8),8:(10,11),9:(12,13),10:(14,15)}
    for key,value in mapping.items():
        data[:,value[0]] = np.real(raw[:,key])
        data[:,value[1]] = np.imag(raw[:,key])

    # Get the magnitudes
    data[:,9] = np.real(raw[:,7])
    data[:,16] = np.real(raw[:,-1])
    
    print('COMSOL SHAPE: ',data.shape)
    
    # We need to fix the order of the variation in spatial coordinates in the matrix
    zs = np.unique(data[:,2])
    xs = np.unique(data[:,0])
    rowlist = []
    for z in zs:
        for x in xs:
            for row in data:
                if row[0] == x and row[2] == z:
                    rowlist.append(row)
    data = np.vstack(rowlist)

    print('COMSOL SHAPE: ',data.shape)
    np.savetxt('comsol_test.txt',data)

    return data


def parse_s4(epath,hpath):
    emat = np.loadtxt(epath)
    hmat = np.loadtxt(hpath)
    xy_inds = emat[:,0:2]
    z_inds = emat[:,2]
    # Convert pos_inds to positions
    xy = xy_inds*(.25/50)
    # Join into 1 matrix
    joined = np.column_stack((xy,z_inds,emat[:,3:],hmat[:,3:]))
    print('S4 SHAPE: ',joined.shape)
    np.savetxt('s4_test.txt',joined)
    

def main():
    print('main')
    parser = ap.ArgumentParser(description="""Compares the data between a provided RCWA data file and COMSOL
    output file""")
    parser.add_argument('s4_dir',type=str,help="""The directory containing data from S4""")
    parser.add_argument('comsol_file',type=str,help="""The data from COMSOL""")
    args = parser.parse_args()

    e_file = os.path.join(os.path.abspath(args.s4_dir),"field_data.E")
    h_file = os.path.join(os.path.abspath(args.s4_dir),"field_data.H")

    if not os.path.isfile(e_file) or not os.path.isfile(h_file):
        print("\nOne of the S4 files you specified does not exist! \n")
        quit()
    else:
        s4data = parse_s4(e_file,h_file)

    if os.path.isfile(args.comsol_file):
        comsoldata = parse_comsol(os.path.abspath(args.comsol_file))
    else:
        print("\n The comsol file you specified does not exist! \n")
        quit()

    # The points extracted from COMSOL and S4 are not exactly the same, so we need to interpolate S4
    # data onto COMSOL grid 
    # So:
    # 1. Construct 3D interpolation of S4 data
    # 2. Get fields at COMSOL points using S4 interpolation. These are the fields S4 WOULD have
    # generated at the COMSOL points. 
    # 3. Now we can compare the two data sets 


if __name__ == '__main__':
    main()
