import numpy as np
import scipy.interpolate as spi
import matplotlib
# Enables saving plots over ssh
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import argparse as ap
import os
import configparser as confp
import shutil

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
    # This preserves case sensitivity
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

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

def parse_comsol(conf,compdir,path):
    print('Parsing new COMSOL data')
    # COpy the comsol file to the comparison dir for completeness
    shutil.copy(path,compdir)
    conv = {col: comp_conv for col in range(3,11)}
    raw = np.loadtxt(path,comments='%',converters=conv,dtype=complex)

    # Fix this horrendous matrix

    # Make the positions real, convert to micrometers and shift z up so 
    # everything starts at z = 0
    data = np.zeros((raw.shape[0],17))
    data[:,0:3] = np.real(raw[:,0:3])*1E6
    data[:,2] = np.absolute(data[:,2] + 1) 
    # We also need to shift the origin so its at the corner of the nanowire
    data[:,0] = data[:,0] + conf.getfloat('Parameters','array_period')/2
    data[:,1] = data[:,1] + conf.getfloat('Parameters','array_period')/2
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
    data = data.reshape(( z_samples, y_samples, x_samples, 3))
    # swap the "y" and "x" axes
    data = np.swapaxes(data, 1,2)
    # back to 2-D array
    data = data.reshape((x_samples*y_samples*z_samples,3))

    print('COMSOL SHAPE: ',data.shape)
    np.savetxt(os.path.join(compdir,'comsol_data_formatted.txt'),data)

    return data

def parse_s4(conf,compdir,epath,hpath):
    print('Parsing new S4 data')
    emat = np.loadtxt(epath)
    hmat = np.loadtxt(hpath)
    x_inds = emat[:,0]
    y_inds = emat[:,1]
    z_inds = emat[:,2]

    # Convert pos_inds to positions
    period = conf.getfloat('Parameters','array_period')
    x = x_inds*(period/conf.getfloat('General','x_samples'))
    y = y_inds*(period/conf.getfloat('General','y_samples'))

    # Join into 1 matrix
    joined = np.column_stack((x,y,z_inds,emat[:,3:],hmat[:,3:]))
    print('S4 SHAPE: ',joined.shape)
    np.savetxt(os.path.join(compdir,'s4_data_formatted.txt'),joined)
    return joined 

def relative_difference(x,y):
    """Return the mean squared error between two equally sized sets of data"""
    if x.size != y.size:
        self.log.error("You have attempted to compare datasets with an unequal number of points!!!!")
        quit()
    else:
        diff = np.abs(x-y)
        norm_diff = diff/np.amax(diff) 
        rel_diff  = np.sum(norm_diff)/norm_diff.size
        return rel_diff

def compare_data(s4data,comsoldata):
    """ The points extracted from COMSOL and S4 are not exactly the same, so we need to interpolate S4
        data onto COMSOL grid 
        So:
        1. Construct 3D interpolation of S4 data
        2. Get fields at COMSOL points using S4 interpolation. These are the fields S4 WOULD have
        generated at the COMSOL points. 
        3. Now we can compare the two data sets""" 

    comsol_pts = comsoldata[:,0:3]
    s4_points = s4data[:,0:3]
    s4_mag = s4data[:,9]
    comsol_mag = comsoldata[:,10]
    

    # Simple mean squared error for now

    # These should be zero
    print("These should be zero")
    interp_vals_s4 = spi.griddata(s4_points,s4_mag,s4_points)
    interp_vals_coms = spi.griddata(comsol_pts,comsol_mag,comsol_pts)
    err = relative_difference(interp_vals_s4,s4_mag)
    print("Error of interpolated and actual S4 mag = ",err)
    err = relative_difference(interp_vals_coms,comsol_mag)
    print("Error of interpolated and actual COMSOL mag = ",err)

    # These should be the same
    
    # This is the S4 data on the COMSOL points
    interp_vals_s4 = spi.griddata(s4_points,s4_mag,comsol_pts)
    # This is the COMSOL data on the S4 points
    interp_vals_coms = spi.griddata(comsol_pts,comsol_mag,s4_points)
    print(interp_vals_s4)
    print(interp_vals_coms)
    err = relative_difference(interp_vals_s4,comsol_mag) 
    print("The error between interpolated S4 and COMSOL = ",err)
    err = relative_difference(interp_vals_coms,s4_mag) 
    print("The error between interpolated COMSOL and S4 = ",err)

def main():
    parser = ap.ArgumentParser(description="""Compares the data between a provided RCWA data file and COMSOL
    output file""")
    parser.add_argument('s4_dir',type=str,help="""The directory containing data from S4""")
    parser.add_argument('comsol_file',type=str,help="""The data from COMSOL""")
    args = parser.parse_args()

    # First lets create a subdirectory within the S4 directory to store all our comparison results
    comsol = False
    s4 = False
    try:
        comp_dir = os.path.join(args.s4_dir,'comparison_results')
        os.mkdir(comp_dir)
    except OSError:
        print('Comparison directory already exists')
        # Since the directory already exists, check for formatted data files and reuse them to save
        # time
        if os.path.isfile(os.path.join(comp_dir,'comsol_data_formatted.txt')):
            print('Discovered formatted COMSOL data file')
            comsol = True
        if os.path.isfile(os.path.join(comp_dir,'s4_data_formatted.txt')):
            print('Discovered formatted S4 data file')
            s4 = True
    
    # Grab the sim config for the S4 sim to which we are comparing
    conf = parse_file(os.path.join(args.s4_dir,'sim_conf.ini'))

    # If the formatted file exists use it, otherwise parse the file provided at the command line
    if comsol:
        print('Reusing formatted COMSOL data file')
        comsoldata = np.loadtxt(os.path.join(comp_dir,'comsol_data_formatted.txt'))
    else:
        if os.path.isfile(args.comsol_file):
            comsoldata = parse_comsol(conf,comp_dir,os.path.abspath(args.comsol_file))
        else:
            print("\n The comsol file you specified does not exist! \n")
            quit()

    # Do the same for the S4 file
    if s4:
        print('Reusing formatted S4 data file')
        s4data = np.loadtxt(os.path.join(comp_dir,'s4_data_formatted.txt'))
    else:

        # Get the data files
        name = conf.get('General','base_name')
        e_file = os.path.join(os.path.abspath(args.s4_dir),name+'.E')
        h_file = os.path.join(os.path.abspath(args.s4_dir),name+'.H')
        
        if not os.path.isfile(e_file) or not os.path.isfile(h_file):
            print("\nOne of the S4 files you specified does not exist! \n")
            quit()
        else:
            s4data = parse_s4(conf,comp_dir,e_file,h_file)

    # Now do the actual comparison    
    compare_data(s4data,comsoldata)

if __name__ == '__main__':
    main()
