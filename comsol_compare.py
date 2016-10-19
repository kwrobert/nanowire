import numpy as np
import scipy.interpolate as spi
import os
import matplotlib
# Enables saving plots over ssh
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import argparse as ap
import configparser as confp
import shutil
import glob
import re

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
    # If the formatted file exists use it, otherwise parse the file provided at the command line
    fdir,fname = os.path.split(path)
    formatted = os.path.join(compdir,fname+'.formatted')
    if os.path.isfile(formatted):
        print('Reusing formatted COMSOL data file')
        comsoldata = np.loadtxt(formatted)
        return comsoldata
    else:
        # COpy the comsol file to the comparison dir for completeness
        shutil.copy(path,compdir)
        # Create dictionary for converting comsol values and load raw data
        conv = {col: comp_conv for col in range(3,8)}
        raw = np.loadtxt(path,comments='%',converters=conv,dtype=complex)

        # Fix this horrendous matrix

        # Make the positions real, convert to micrometers and shift z up so 
        # everything starts at z = 0
        data = np.zeros((raw.shape[0],11))
        data[:,0:3] = np.real(raw[:,0:3])*1E6
        data[:,2] = np.absolute(data[:,2] + 1) 
        # We also need to shift the origin so its at the corner of the nanowire
        data[:,0] = data[:,0] + conf.getfloat('Fixed Parameters','array_period')/2
        data[:,1] = data[:,1] + conf.getfloat('Fixed Parameters','array_period')/2
        # Grab the frequency column
        data[:,3] = np.real(raw[:,3])
        # Split the complex values of all the field components. Dict maps col
        # of raw data to the two new columns in the cleaned up matrix
        mapping = {4:(4,5),5:(6,7),6:(8,9)}
        for key,value in mapping.items():
            data[:,value[0]] = np.real(raw[:,key])
            data[:,value[1]] = np.imag(raw[:,key])

        # Get the magnitude
        data[:,10] = np.real(raw[:,-1])
        
        print('COMSOL SHAPE: ',data.shape)
        
        # We need to fix the order of the variation in spatial coordinates in the matrix
        # (No need to copy if you don't want to keep the given_dat ordering)
        cols = data.shape[-1]
        z_samples = conf.getint('General','z_samples')
        x_samples = conf.getint('General','x_samples')
        y_samples = conf.getint('General','y_samples')
        new_data = data.reshape(( z_samples, y_samples, x_samples, cols))
        # swap the "y" and "x" axes
        new_data = np.swapaxes(new_data, 1,2)
        # back to 2-D array
        new_data = new_data.reshape((x_samples*y_samples*z_samples,cols))

        print('COMSOL SHAPE: ',new_data.shape)
        np.savetxt(formatted,new_data)

        return new_data

def parse_s4(conf,compdir,path):
    print('Parsing new S4 data')
    # Do the same for the S4 file
    fdir,fname = os.path.split(path)
    freq = os.path.basename(fdir)
    formatted = os.path.join(compdir,freq+'.formatted')
    if os.path.isfile(formatted):
        print('Reusing formatted S4 data file')
        s4data = np.loadtxt(formatted)
        return s4data
    else:
        emat = np.loadtxt(path)

        # Convert pos_inds to positions
        period = conf.getfloat('Fixed Parameters','array_period')
        emat[:,0] = emat[:,0]*(period/conf.getfloat('General','x_samples'))
        emat[:,1] = emat[:,1]*(period/conf.getfloat('General','y_samples'))

        # Join into 1 matrix
        print('S4 SHAPE: ',emat.shape)
        np.savetxt(formatted,emat)
        return emat 

def heatmap2d(x,y,cs,labels,ptype,path,colorsMap='jet'):
    """A general utility method for plotting a 2D heat map"""
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=np.amin(cs), vmax=np.amax(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.pcolormesh(x, y, cs,cmap=cm,norm=cNorm,alpha=.5)
    tx = np.arange(0,.25,.01)
    ty = np.zeros_like(tx)
    ty[:] = .5
    ax.plot(tx,ty)
    scalarMap.set_array(cs)
    cb = fig.colorbar(scalarMap)
    cb.set_label(labels[-1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    fig.suptitle('comsol cut plane')
    name = path+'_'+labels[-1]+'_'+ptype+'.pdf'
    print('Saving file %s'%name)
    fig.savefig(name)
    plt.close(fig)

def plot_comsol(data,conf,data_file):
    xval = .12755
    mat = np.column_stack((data[:,0],data[:,1],data[:,2],data[:,-1]))
    planes = np.array([row for row in mat if np.abs(row[0]-.12755) < .00001])
    print(planes)
    x,y,z = np.unique(planes[:,0]),np.unique(planes[:,1]),np.unique(planes[:,2])
    cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
    labels = ('y [um]','z [um]', 'normE')
    heatmap2d(y,z,cs,labels,'plane_2d_x',data_file.rstrip('.txt'))

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

def compare_data(s4data,comsoldata,conf,interpolate=False,exclude=False):
    """ The points extracted from COMSOL and S4 are not exactly the same, so we need to interpolate S4
        data onto COMSOL grid 
        So:
        1. Construct 3D interpolation of S4 data
        2. Get fields at COMSOL points using S4 interpolation. These are the fields S4 WOULD have
        generated at the COMSOL points. 
        3. Now we can compare the two data sets""" 

    if exclude:
        # Exclude the air regions and substrate regions
        x_samples = conf.getint('General','x_samples')
        y_samples = conf.getint('General','y_samples')
        z_samples = conf.getint('General','z_samples')
        h = sum((conf.getfloat('Fixed Parameters','nw_height'),conf.getfloat('Fixed Parameters','substrate_t'),
                conf.getfloat('Fixed Parameters','air_t'),conf.getfloat('Fixed Parameters','ito_t')))
        arr = np.linspace(0,h,z_samples)
        dz = arr[1] - arr[0]
        #print('dz = ',dz)
        #print(round(conf.getfloat('Fixed Parameters','air_t')/dz))
        #print(round(sum((conf.getfloat('Fixed Parameters','nw_height'),conf.getfloat('Fixed Parameters','air_t'),
        #        conf.getfloat('Fixed Parameters','ito_t')))/dz))
        start = int(round(conf.getfloat('Fixed Parameters','air_t')/dz)*(x_samples*y_samples))
        end = int(round(sum((conf.getfloat('Fixed Parameters','nw_height'),conf.getfloat('Fixed Parameters','air_t'),
                conf.getfloat('Fixed Parameters','ito_t')))/dz))
        end = end*(x_samples*y_samples)
        print(start)
        print(end)
        comsol_pts = comsoldata[start:end,0:3]
        #comsol_pts[0,0], comsol_pts[0,1] = 0,0
        s4_points = s4data[start:end,0:3]
        s4_mag = s4data[start:end,-1]
        comsol_mag = comsoldata[start:end,-1]
    else:
        comsol_pts = comsoldata[:,0:3]
        #comsol_pts[0,0], comsol_pts[0,1] = 0,0
        s4_points = s4data[:,0:3]
        s4_mag = s4data[:,-1]
        comsol_mag = comsoldata[:,-1]
    

    # Simple mean squared error for now

    if interpolate:
        print('Interpolating data sets before comparison ...')
        # These should be zero
        #print("These should be zero")
        #xv,yv,zv = np.meshgrid(s4_points
        #interp_vals_s4 = spi.griddata(s4_points,s4_mag,
        #                 (s4_points[:,0],s4_points[:,1],s4_points[:,2]),method='linear')
        #interp_vals_coms = spi.griddata(comsol_pts,comsol_mag,comsol_pts,method='linear')
        #err = relative_difference(interp_vals_s4,s4_mag)
        #print("Error between interpolated and actual S4 mag = ",err)
        #err = relative_difference(interp_vals_coms,comsol_mag)
        #print("Error between interpolated and actual COMSOL mag = ",err)
        
        # Regular grid
        # Interpolating S4 points onto comsol data/grid
        for tup in zip(comsol_mag,s4_mag):
            print(tup)
        x = input('Continue?')
        print('Reg grid method')
        cx,cy,cz = np.unique(comsol_pts[:,0]),np.unique(comsol_pts[:,1]),np.unique(comsol_pts[:,2])
        points = (cx,cy,cz)
        print('len cx : ',len(cx))
        print('len cy : ',len(cy))
        print('len cz : ',len(cz))
        print('len s4 mag : ',len(s4_mag))
        print('len coms mag : ',len(comsol_mag))
        dat = np.column_stack((comsol_pts,comsol_mag))
        dat = dat.reshape((z_samples,y_samples,x_samples,4))
        dat = np.swapaxes(dat,0,2)
        dat.reshape((z_samples*x_samples*y_samples,4))
        values = dat[-1].reshape((len(cx),len(cy),len(cz)))
        print('comsol pts dim: ',comsol_pts.shape)
        print('comsol pts last: ',comsol_pts[-1,:])
        print('s4 pts last: ',s4_points[-1,:])
        print('comsol pts first: ',comsol_pts[0,:])
        print('s4 pts first: ',s4_points[0,:])
        print('Interpolating')
        interp_vals_coms = spi.interpn(points,values,s4_points,method='linear',bounds_error=False,fill_value=None)
        print('Interp vals: ')
        print(interp_vals_coms)
        print('s4 vals: ',s4_mag)
        print('s4 len: ',len(s4_mag))
        print('interp len: ',len(interp_vals_coms))
        print(relative_difference(interp_vals_coms,s4_mag))
        quit()


        # These should be the same

        # This is the S4 data on the COMSOL points
        cx,cy,cz = np.meshgrid(np.unique(comsol_pts[:,0]),
                               np.unique(comsol_pts[:,1]),np.unique(comsol_pts[:,2]))
        interp_vals_s4 = spi.griddata(s4_points,s4_mag,(cx,cy,cz),method='linear')
        # This is the COMSOL data on the S4 points
        interp_vals_coms = spi.griddata(comsol_pts,comsol_mag,s4_points,method='linear')
        print(interp_vals_s4)
        print(interp_vals_coms)
        print('Computing error')
        err = relative_difference(interp_vals_s4,comsol_mag) 
        print("The error between interpolated S4 and COMSOL = ",err)
        err = relative_difference(interp_vals_coms,s4_mag) 
        print("The error between interpolated COMSOL and S4 = ",err)
    else:
        # Just get error without interpolating 
        print('Not interpolating data sets before comparison ...')
        err = relative_difference(s4_mag,comsol_mag) 
        print("The error between interpolated COMSOL and S4 = ",err)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def main():
    parser = ap.ArgumentParser(description="""Compares the data between a provided RCWA data file and COMSOL
    output file""")
    parser.add_argument('s4_dir',type=str,help="""The directory containing data from S4""")
    parser.add_argument('coms_dir',type=str,help="""The directory containing data from COMSOL""")
    parser.add_argument('conf_file',type=str,help="""The config file for the S4 simulation""")
    parser.add_argument('-i','--interpolate',action='store_true',default=False,help="""Interpolate
    data sets?""")
    parser.add_argument('-e','--exclude',action='store_true',default=False,help="""Exclude substrate
    and air region from comparison?""")
    args = parser.parse_args()

    # First lets create a subdirectory within the S4 directory to store all our comparison results
    s4_dir = os.path.abspath(args.s4_dir)
    coms_dir = os.path.abspath(args.coms_dir)
    if not (os.path.exists(s4_dir) and os.path.exists(coms_dir)):
        print('One of your paths doesnt exist')
        quit()

    try:
        comp_dir = os.path.join(s4_dir,'comparison_results')
        os.mkdir(comp_dir)
    except OSError:
        pass 
    
    # Grab the sim config for the S4 sim to which we are comparing
    if os.path.isfile(os.path.abspath(args.conf_file)):
        conf = parse_file(os.path.abspath(args.conf_file))
    else:
        print('Conf file doesnt exist')
        quit()

    # Glob data files
    sg = os.path.join(s4_dir,'**/*.E')
    s4_files = glob.glob(sg,recursive=True)
    cg = os.path.join(coms_dir,'frequency*.txt')
    coms_files = glob.glob(cg)
    print('S4 len: ',len(s4_files))
    print('COMSOL len: ',len(coms_files))
    files = zip(sorted(s4_files,key=natural_keys),sorted(coms_files,key=natural_keys))
    for f in files:
        print('S4 file: ',f[0])
        print('COMSOL file: ',f[1])
        # Parse the files
        s4data = parse_s4(conf,comp_dir,f[0])
        comsoldata = parse_comsol(conf,comp_dir,f[1])
        plot_comsol(comsoldata,conf,f[1])
        # Now do the actual comparison    
        compare_data(s4data,comsoldata,conf,args.interpolate,args.exclude)

if __name__ == '__main__':
    main()
