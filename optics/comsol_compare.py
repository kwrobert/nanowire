import numpy as np
import scipy.interpolate as spi
import scipy.constants as constants
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
    # If the formatted file exists use it, otherwise parse the file provided at the command line
    fdir,fname = os.path.split(path)
    formatted = os.path.join(compdir,fname)
    if os.path.isfile(formatted+'.npz'):
        print('Reusing formatted COMSOL data file')
        with np.load(formatted+'.npz') as data:
            comsoldata = data['arr_0']
        return comsoldata
    else:
        # COpy the comsol file to the comparison dir for completeness
        #shutil.copy(path,compdir)
        # Make a softlink to the original comsol data file for completeness
        try:
            os.symlink(path,os.path.join(compdir,os.path.basename(path)))
        except FileExistsError:
            pass
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
        np.savez(formatted,new_data)

        return new_data

def parse_s4(conf,compdir,path):
    # Do the same for the S4 file
    fdir,fname = os.path.split(path)
    freq = os.path.basename(fdir)
    if conf.get('General','save_as') == 'text':
        emat = np.loadtxt(path)
        # Convert pos_inds to positions
        period = conf.getfloat('Fixed Parameters','array_period')
        emat[:,0] = emat[:,0]*(period/conf.getfloat('General','x_samples'))
        emat[:,1] = emat[:,1]*(period/conf.getfloat('General','y_samples'))

    elif conf.get('General','save_as') == 'npz': 
        with np.load(path) as data:
            emat = data['data']
            # Convert pos_inds to positions
            period = conf.getfloat('Fixed Parameters','array_period')
            emat[:,0] = emat[:,0]*(period/conf.getfloat('General','x_samples'))
            emat[:,1] = emat[:,1]*(period/conf.getfloat('General','y_samples'))
    
    print('S4 SHAPE: ',emat.shape)
    return emat[:,0:10] 

def heatmap2d(x,y,cs,labels,ptype,path=None,colorsMap='jet'):
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

def heatmap2dax(ax,x,y,cs,labels,cNorm,conf,colorsMap='jet'):
    """A general utility method for plotting a 2D heat map"""
    cm = plt.get_cmap(colorsMap)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    ax.pcolormesh(x, y, cs,cmap=cm,norm=cNorm,alpha=.5)
    scalarMap.set_array(cs)
    cb = plt.colorbar(scalarMap,ax=ax)
    cb.set_label(labels[2])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(labels[3])
    ax.axis('tight')
    ## Draw a line at the interface between each layer
    ito_line = conf.getfloat('Fixed Parameters','air_t')
    nw_line = conf.getfloat('Fixed Parameters','ito_t')+ito_line
    sub_line = conf.getfloat('Fixed Parameters','nw_height')+nw_line
    air_line = sub_line+conf.getfloat('Fixed Parameters','substrate_t')
    for line_h in [(ito_line,'ITO'),(nw_line,'NW'),(sub_line,'Substrate'),(air_line,'Air')]:
        x = [0,conf.getfloat('Fixed Parameters','array_period')]
        y = [line_h[0],line_h[0]]
        label_y = line_h[0] + 0.01
        label_x = x[-1]
        #ax.text(label_x,label_y,line_h[-1],ha='right',family='sans-serif',size=12)
        line = mlines.Line2D(x,y,linestyle='solid',linewidth=2.0,color='black')
        ax.add_line(line)
    # Draw two vertical lines to show the edges of the nanowire
    cent = conf.getfloat('Fixed Parameters','array_period')/2.0
    rad = conf.getfloat('Fixed Parameters','nw_radius')
    shell = conf.getfloat('Fixed Parameters','shell_t')
    bottom = conf.getfloat('Fixed Parameters','ito_t')+ito_line
    top = conf.getfloat('Fixed Parameters','nw_height')+nw_line
    for x in (cent-rad,cent+rad,cent-rad-shell,cent+rad+shell):
        xv = [x,x]
        yv = [bottom,top]
        line = mlines.Line2D(xv,yv,linestyle='solid',linewidth=2.0,color='black')
        ax.add_line(line)
    return ax

def plot_comsol(data,conf,data_file):
    xval = .12755
    mat = np.column_stack((data[:,0],data[:,1],data[:,2],data[:,-1]))
    planes = np.array([row for row in mat if np.abs(row[0]-.12755) < .00001])
    x,y,z = np.unique(planes[:,0]),np.unique(planes[:,1]),np.unique(planes[:,2])
    cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
    labels = ('y [um]','z [um]', 'normE')
    heatmap2d(y,z,cs,labels,'plane_2d_x',data_file.rstrip('.txt'))

def plot_diff(data,conf,files):
    #print(data.shape)
    #print(data)
    planes = np.array([row for row in data if np.abs(row[0]-.125) < .00001])
    print(planes)
    x,y,z = np.unique(planes[:,0]),np.unique(planes[:,1]),np.unique(planes[:,2])
    cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
    labels = ('y [um]','z [um]', 'normE')
    fname = os.path.split(os.path.dirname(files[0]))[-1]
    path = os.path.join(files[-1],fname)
    #print(path)
    heatmap2d(y,z,cs,labels,'difference_magnitude',path)

def gen_plots(s4dat,comsdat,diffdat,conf,files,log):
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(13,6),ncols=3)
    # Define normalization for colorbar,
    s4min,s4max = np.amin(s4dat[:,-1]),np.amax(s4dat[:,-1])
    comsmin,comsmax = np.amin(comsdat[:,-1]),np.amax(comsdat[:,-1])
    print('S4 min: ',s4min)
    print('S4 max: ',s4max)
    print('COMSOL min: ',comsmin)
    print('COMSOL max: ',comsmax)
    if comsmin < s4min:
        gmin = comsmin
    else:
        gmin = s4min
    if comsmax > s4max:
        gmax = comsmax
    else:
        gmax = s4max
    cNorm = matplotlib.colors.Normalize(vmin=gmin, vmax=gmax)
    cm = plt.get_cmap('jet')
    # S4 Data plot
    period = conf.getfloat('Fixed Parameters','array_period')
    x_samp = conf.getfloat('General','x_samples')
    y_samp = conf.getfloat('General','y_samples') 
    dx = period/x_samp
    dy = period/y_samp
    mat = np.column_stack((s4dat[:,0],s4dat[:,1],s4dat[:,2],s4dat[:,-1]))
    planes = np.array([row for row in mat if np.abs(row[0]-dx*(x_samp/2)) < .0001])
    x,y,z = np.unique(planes[:,0]),np.unique(planes[:,1]),np.unique(planes[:,2])
    cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
    labels = ('y [um]','z [um]', 'normE', 'S4 Data')
    ax1 = heatmap2dax(ax1,y,z,cs,labels,cNorm,conf)
    # Plots comsol data
    xval = .12755
    mat = np.column_stack((comsdat[:,0],comsdat[:,1],comsdat[:,2],comsdat[:,-1]))
    planes = np.array([row for row in mat if np.abs(row[0]-.12755) < .00001])
    x,y,z = np.unique(planes[:,0]),np.unique(planes[:,1]),np.unique(planes[:,2])
    cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
    labels = ('y [um]','z [um]', 'normE','COMSOL Data')
    ax2 = heatmap2dax(ax2,y,z,cs,labels,cNorm,conf)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    # Plots diff data
    planes = np.array([row for row in diffdat if np.abs(row[0]-.125) < .00001])
    x,y,z = np.unique(planes[:,0]),np.unique(planes[:,1]),np.unique(planes[:,2])
    cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
    labels = ('y [um]','z [um]', 'M.S.E','Difference Between Data Sets')
    if log:
        cNorm = matplotlib.colors.LogNorm(vmin=np.amin(cs), vmax=np.amax(cs))
    else:
        cNorm = matplotlib.colors.Normalize(vmin=np.amin(cs), vmax=np.amax(cs))
    ax3 = heatmap2dax(ax3,y,z,cs,labels,cNorm,conf)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    # Fix overlap
    fig.tight_layout()
    # Now save the plot
    fname = os.path.split(os.path.dirname(files[0]))[-1]
    path = os.path.join(files[-1],fname)
    name = path+'multiplot.pdf'
    print('Saving figure %s'%name)
    fig.suptitle('Frequency = %.4E, Wavelength = %.2f nm'%(comsdat[0,3],(constants.c/comsdat[0,3])*1E9))
    fig.subplots_adjust(top=0.85)
    plt.savefig(name)
    plt.close(fig)

def relative_difference(x,y):
    """Return the mean squared error between two equally sized sets of data"""
    if x.size != y.size:
        print("You have attempted to compare datasets with an unequal number of points!!!!")
        quit()
    else:
        diff = x-y
        rel_diff  = np.sum(diff**2)/np.sum(x**2)
        return np.abs(diff), rel_diff

def compare_data(s4data,comsoldata,conf,files,interpolate=False,exclude=False):
    """ The points extracted from COMSOL and S4 are not exactly the same, so we need to interpolate S4
        data onto COMSOL grid 
        So:
        1. Construct 3D interpolation of S4 data
        2. Get fields at COMSOL points using S4 interpolation. These are the fields S4 WOULD have
        generated at the COMSOL points. 
        3. Now we can compare the two data sets""" 

    x_samples = conf.getint('General','x_samples')
    y_samples = conf.getint('General','y_samples')
    z_samples = conf.getint('General','z_samples')
    h = sum((conf.getfloat('Fixed Parameters','nw_height'),conf.getfloat('Fixed Parameters','substrate_t'),
            conf.getfloat('Fixed Parameters','air_t'),conf.getfloat('Fixed Parameters','ito_t')))
    if exclude:
        # Exclude the air regions and substrate regions
        arr = np.linspace(0,h,z_samples)
        dz = arr[1] - arr[0]
        start_plane = int(round(conf.getfloat('Fixed Parameters','air_t')/dz))
        start = start_plane*(x_samples*y_samples)
        end_plane = int(round(sum((conf.getfloat('Fixed Parameters','nw_height'),conf.getfloat('Fixed Parameters','air_t'),
                conf.getfloat('Fixed Parameters','ito_t')))/dz))
        end = end_plane*(x_samples*y_samples)
        comsol_pts = comsoldata[start:end,0:3]
        s4_points = s4data[start:end,0:3]
        s4_mag = s4data[start:end,-1]
        comsol_mag = comsoldata[start:end,-1]
    else:
        comsol_pts = comsoldata[:,0:3]
        s4_points = s4data[:,0:3]
        s4_mag = s4data[:,-1]
        comsol_mag = comsoldata[:,-1]
    

    # Simple mean squared error for now

    if interpolate:
        print('Interpolating data sets before comparison ...')
        cx,cy,cz = np.unique(comsol_pts[:,0]),np.unique(comsol_pts[:,1]),np.unique(comsol_pts[:,2])
        points = (cx,cy,cz)
        dat = np.column_stack((comsol_pts,comsol_mag))
        if exclude:
            dat = dat.reshape((end_plane-start_plane,y_samples,x_samples,4))
            dat = np.swapaxes(dat,0,2)
            dat = dat.reshape(((end_plane-start_plane)*x_samples*y_samples,4))
        else:
            dat = dat.reshape((z_samples,y_samples,x_samples,4))
            dat = np.swapaxes(dat,0,2)
            dat = dat.reshape((z_samples*x_samples*y_samples,4))
        values = dat[:,-1].reshape((len(cx),len(cy),len(cz)))
        print('Interpolationg S4 points onto COMSOL data and grid')
        interp_vals_coms = spi.interpn(points,values,s4_points,method='linear',bounds_error=False,fill_value=None)
        diff_vec, err = relative_difference(interp_vals_coms,s4_mag)
        print("The error between interpolated COMSOL and S4 = ",err)
        diff_dat = np.column_stack((s4_points,diff_vec))
    else:
        # Just get error without interpolating 
        print('Not interpolating data sets before comparison ...')
        diff_vec, err = relative_difference(s4_mag,comsol_mag) 
        diff_dat = np.column_stack((s4_points,diff_vec))
        print("The error between COMSOL and S4 = ",err)
    return diff_dat, err

def dir_keys(path):
    """A function to take a path, and return a list of all the numbers in the path. This is
    mainly used for sorting 
        by the parameters they contain"""

    regex = '[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?' # matching any floating point
    m = re.findall(regex, path)
    if(m): val = m
    else: raise ValueError('Your path does not contain any numbers')
    val = list(map(float,val))
    return val

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
    parser.add_argument('-o','--output',type=str,default='comparison_results',help="""Name for the
    output directory of all the comparison results. Not an absolute path""")
    parser.add_argument('--log',action='store_true',default=False,help="""Plot difference data on a
    log scale""")
    args = parser.parse_args()

    # First lets create a subdirectory within the S4 directory to store all our comparison results
    s4_dir = os.path.abspath(args.s4_dir)
    coms_dir = os.path.abspath(args.coms_dir)
    if not (os.path.exists(s4_dir) and os.path.exists(coms_dir)):
        print('One of your paths doesnt exist')
        quit()

    try:
        comp_dir = os.path.join(s4_dir,args.output)
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
    if conf.get('General','save_as') == 'text':
        sg = os.path.join(s4_dir,'**/*.E.crnch')
    elif conf.get('General','save_as') == 'npz': 
        sg = os.path.join(s4_dir,'**/*.E.npz')

    s4_files = glob.glob(sg,recursive=True)
    cg = os.path.join(coms_dir,'frequency*.txt')
    coms_files = glob.glob(cg)
    print('Number of S4 files: %i'%len(s4_files))
    print('Number of COMSOL files: %i'%len(coms_files))
    files = zip(sorted(s4_files,key=dir_keys),sorted(coms_files,key=dir_keys))
    err_file = os.path.join(comp_dir,'errors.txt')
    freqs = []
    errs = []
    with open(err_file,'w') as efile:
        efile.write('# freq (Hz), error\n')
        for f in files:
            print('*'*50)
            print('Comparing following files: ')
            print('S4 File: %s'%f[0])
            print('COMSOL File: %s'%f[1])
            # Parse the files
            s4data = parse_s4(conf,comp_dir,f[0])
            comsoldata = parse_comsol(conf,comp_dir,f[1])
            # Get frequency for err file
            freq = comsoldata[0,3]
            print('COMSOL Frequency: {:G}'.format(freq))
            # We need to flip the comsol magnitude over before comparing
            mag = comsoldata[:,-1]
            comsoldata[:,-1] = mag[::-1]
            # Now do the actual comparison    
            diff, err = compare_data(s4data,comsoldata,conf,(f[0],f[1],comp_dir),args.interpolate,args.exclude)
            gen_plots(s4data,comsoldata,diff,conf,(f[0],f[1],comp_dir),args.log)
            efile.write('%f, %f\n'%(freq,err))
            #plot_comsol(comsoldata,conf,f[1])
            #plot_diff(diff_dat,conf,files)
            freqs.append(freq)
            errs.append(err)
    fig = plt.figure(3)
    plt.title("Errors between COMSOL and S4")
    plt.plot(freqs,errs,'b-o')
    plt.plot([freqs[0],freqs[-1]],[.01,.01],'k-',linewidth=2.0)
    plt.yticks(np.arange(min(errs),max(errs)+.01,.01))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('M.S.E of normE')
    eplot = os.path.join(comp_dir,'error_plot.pdf')
    plt.savefig(eplot)
    plt.close(fig)


if __name__ == '__main__':
    main()
