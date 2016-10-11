import numpy as np
import argparse as ap
import os
import configparser as confp 
import re
import logging
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

def configure_logger(level,logger_name,log_dir,logfile):
    # Get numeric level safely
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    # Set formatting
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
    # Get logger with name
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    # Set up file handler
    try:
        os.makedirs(log_dir)
    except OSError:
        # Log dir already exists
        pass
    output_file = os.path.join(log_dir,logfile)
    fhandler = logging.FileHandler(output_file)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
    # This preserves case sensitivity
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

class Processor(object):
    """Base data processor class that has some methods every other processor needs"""
    def __init__(self,global_conf):
        self.log = logging.getLogger('postprocess')
        self.log.debug("Processor base init")
        self.gconf = global_conf
        exclude = set(['logs'])
        #base = self.gconf.get('General','basedir')
        #simdirs = []
        #for adir in os.listdir(base):
        #    if os.path.isdir(os.path.join(base,adir)) and adir not in exclude:
        #        simdirs.append(os.path.join(base,adir))
        #self.sims = [parse_file(os.path.join(simdir,'sim_conf.ini')) for simdir in simdirs]
        self.collect_sims()
        self.log.debug('Sims before sorting: %s',str(self.sims))
        # Sort on the sim dir to prevent weirdness when calculating convergence. Sorts by ascending
        # param values and works for multiple variable params
        self.sort_sims()
        self.log.debug('Sims after sorting: %s',str(self.sims))
        self.sim = None
    
    def collect_sims(self):
        self.sims = []
        self.sim_groups = []
        for root,dirs,files in os.walk(self.gconf.get('General','basedir')):
            if 'sim_conf.ini' in files:
                obj = parse_file(os.path.join(root,'sim_conf.ini'))
                self.sims.append(obj)
            if 'sorted_sweep_conf.ini' in files:
                conf_paths = [os.path.join(root,simdir,'sim_conf.ini') for simdir in dirs]
                self.log.debug('Sim group confs: %s',str(conf_paths))
                self.sim_groups.append(list(map(parse_file,conf_paths)))
                self.log.debug('Sim groups: %s',str(self.sim_groups))

    def sort_sims(self):
        """Sorts simulations by their parameters the way a human would. Called human sorting or
        natural sorting. Thanks stackoverflow"""
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(sim):
            text = sim.get('General','sim_dir')
            return [ atoi(c) for c in re.split('(\d+)', text) ]

        self.sims.sort(key=natural_keys)
        for group in self.sim_groups:
            paths = [c.get('General','sim_dir') for c in group]
            self.log.debug('Group paths before sorting: %s',str(paths))
            group.sort(key=natural_keys)
            paths = [c.get('General','sim_dir') for c in group]
            self.log.debug('Group paths after sorting: %s',str(paths))

    def get_data(self):
        """Gets the E and H data for this particular sim"""
        sim_path = self.sim.get('General','sim_dir')
        base_name = self.gconf.get('General','base_name')
        e_path = os.path.join(sim_path,base_name+'.E')
        h_path = os.path.join(sim_path,base_name+'.H')
        e_data = np.loadtxt(e_path)
        h_data = np.loadtxt(h_path)
        self.pos_inds = np.array_split(e_data,[3],axis=1)[0]
        self.e_data = e_data
        self.h_data = h_data

class Cruncher(Processor):
    """Crunches all the raw data. Calculates quantities specified in the global config file and
    either appends them to the existing data files or creates new ones as needed"""

    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug("This is THE CRUNCHER!!!!!")
        
    def crunch(self):
        mse = False
        self.log.info('Beginning data crunch ...')
        # NOTE: Doesn't make sense to switch the order of these loops as in the Plotter class
        # because we only want to write out data matrix once
        for sim in self.sims:
            # Set it as the current sim and grab its data
            self.sim = sim
            self.get_data()
            self.log.info('Crunching data for %s',
                          os.path.basename(self.sim.get('General','sim_dir')))
            # For each quantity 
            for quant,args in self.gconf.items('Cruncher'):
                self.log.info('Computing %s with args %s',str(quant),str(args))
                for argset in args.split(';'):
                    if argset:
                        self.calculate(quant,argset.split(','))
                    else:
                        self.calculate(quant,[])
            self.write_data()
    
    def calculate(self,quantity,args):
        try:
            getattr(self,quantity)(*args)
        except KeyError:
            print() 
            print("You have attempted to calculate an unsupported quantity!!")
            quit()

    def write_data(self):
        """Writes the crunched data"""
        # Get the current path
        base = self.sim.get('General','sim_dir')
        fname = self.sim.get('General','base_name')
        epath = os.path.join(base,fname+'.E')
        hpath = os.path.join(base,fname+'.H')
        # Move the raw data
        if not os.path.isfile(os.path.join(epath,epath+'.raw')):
            os.rename(epath,epath+".raw")
        if not os.path.isfile(os.path.join(hpath,hpath+'.raw')):
            os.rename(hpath,hpath+".raw")
        # Build the full matrices
        #full_emat = np.column_stack((self.pos_inds,self.e_data))
        #full_hmat = np.column_stack((self.pos_inds,self.h_data))
        # Build the header strings for the matrices
        eheader = ['x','y','z','Ex_real','Ey_real','Ez_real','Ex_imag','Ey_imag','Ez_imag']
        hheader = ['x','y','z','Hx_real','Hy_real','Hz_real','Hx_imag','Hy_imag','Hz_imag']
        for quant,args in self.gconf.items('Cruncher'):
            # TODO: This is just terrible because it means any quantities NOT pertaining to the
            # efield are not allowed to have a captial E in their name. Same for the H field. 
            if 'E' in quant:
                eheader.append(quant)
            elif 'H' in quant:
                hheader.append(quant)
        # Write the matrices. TODO: Add a nice format string
        #formatter = lambda x: "%22s"%x
        #np.savetxt(epath,full_emat,header=''.join(map(formatter, eheader)))
        #np.savetxt(hpath,full_hmat,header=''.join(map(formatter, hheader)))
        self.log.debug('Here is the E matrix: \n %s',str(self.e_data))
        self.log.debug('Here is the H matrix: \n %s',str(self.h_data))
        np.savetxt(epath,self.e_data,header=','.join(eheader))
        np.savetxt(hpath,self.h_data,header=','.join(hheader))
    
    def normE(self):
        """Calculate and returns the norm of E"""
        
        if not hasattr(self,'e_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of E and add it to our data
        E_mag = np.zeros(self.e_data.shape[0])
        for i in range(3,9):
            E_mag += self.e_data[:,i]*self.e_data[:,i]
        E_mag = np.sqrt(E_mag)
        #Ex_r = self.e_data[:,3] 
        #Ey_r = self.e_data[:,4] 
        #Ez_r = self.e_data[:,5] 
        #Ex_i = self.e_data[:,6]
        #Ey_i = self.e_data[:,7]
        #Ez_i = self.e_data[:,8]
        #E_mag = np.sqrt(Ex_r*Ex_r+Ex_i*Ex_i+Ey_r*Ey_r+Ey_i*Ey_i+Ez_r*Ez_r+Ez_i*Ez_i))
        # The .real is super important or it ruins the entire array
        # Note that discarding imag parts is fine here because the
        # magnitude is strictly real and all imag parts are 0
        self.e_data = np.column_stack((self.e_data,E_mag)) 
        return E_mag

    def normEsquared(self):
        """Calculates and returns normE squared"""
        if not hasattr(self,'e_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of E and add it to our data
        E_magsq = np.zeros(self.e_data.shape[0])
        for i in range(3,9):
            E_magsq += self.e_data[:,i]*self.e_data[:,i]
        #Ex_r = self.e_data[:,3] 
        #Ey_r = self.e_data[:,4] 
        #Ez_r = self.e_data[:,5] 
        #Ex_i = self.e_data[:,6]
        #Ey_i = self.e_data[:,7]
        #Ez_i = self.e_data[:,8]
        #E_magsq = np.sqrt(Ex_r*Ex_r+Ex_i*Ex_i+Ey_r*Ey_r+Ey_i*Ey_i+Ez_r*Ez_r+Ez_i*Ez_i))
        # The .real is super important or it ruins the entire array
        # Note that discarding imag parts is fine here because the
        # magnitude is strictly real and all imag parts are 0
        self.e_data = np.column_stack((self.e_data,E_magsq)) 
        return E_magsq

    def normH(self):
        """Calculate and returns the norm of H"""
        
        if not hasattr(self,'h_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of H and add it to our data. This loops through each components real and
        # imaginary parts and squared it (which is what would happen if you took the complex number
        # for each component and multiplied it by its conjugate). 
        H_mag = np.zeros(self.h_data.shape[0])
        for i in range(3,9):
            H_mag += self.h_data[:,i]*self.h_data[:,i]
        H_mag = np.sqrt(H_mag)
        #Hx_r = self.h_data[:,3] 
        #Hy_r = self.h_data[:,4] 
        #Hz_r = self.h_data[:,5] 
        #Hx_i = self.h_data[:,6]
        #Hy_i = self.h_data[:,7]
        #Hz_i = self.h_data[:,8]
        #H_mag = np.sqrt(Hx_r*Hx_r+Hx_i*Hx_i+Hy_r*Hy_r+Hy_i*Hy_i+Hz_r*Hz_r+Hz_i*Hz_i))
        # The .real is super important or it ruins the entire array
        # Note that discarding imag parts is fine here because the
        # magnitude is strictly real and all imag parts are 0

        self.h_data = np.column_stack((self.h_data,H_mag))
        return H_mag
    
    def normHsquared(self):
        """Calculates and returns the norm of H squared"""
        
        if not hasattr(self,'h_data'):
            self.log.error("You need to get your data first!")
            quit()
        
        # Get the magnitude of H and add it to our data
        H_magsq = np.zeros(self.h_data.shape[0])
        for i in range(3,9):
            H_magsq += self.h_data[:,i]*self.h_data[:,i]
        #Hx_r = self.h_data[:,3] 
        #Hy_r = self.h_data[:,4] 
        #Hz_r = self.h_data[:,5] 
        #Hx_i = self.h_data[:,6]
        #Hy_i = self.h_data[:,7]
        #Hz_i = self.h_data[:,8]
        #H_mag = np.sqrt(Hx_r*Hx_r+Hx_i*Hx_i+Hy_r*Hy_r+Hy_i*Hy_i+Hz_r*Hz_r+Hz_i*Hz_i))
        # The .real is super important or it ruins the entire array
        # Note that discarding imag parts is fine here because the
        # magnitude is strictly real and all imag parts are 0

        self.h_data = np.column_stack((self.h_data,H_magsq))
        return H_magsq

class Global_Cruncher(Cruncher):
    """Computes global quantities for an entire run, instead of local quantities for an individual
    simulation"""
    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug('This is the global cruncher') 

    def crunch(self):
        self.log.info('Beginning global data crunch ...')
        # For each quantity 
        for quant,args in self.gconf.items('Global_Cruncher'):
            self.log.info('Computing %s with args %s',str(quant),str(args))
            for argset in args.split(';'): 
                if argset:
                    self.calculate(quant,argset.split(','))
                else:
                    self.calculate(quant,[])

    def mse(self,x,y):
        """Return the mean squared error between two equally sized sets of data"""
        if x.size != y.size:
            self.log.error("You have attempted to compare datasets with an unequal number of points!!!!")
            quit()
        else:
            mse = np.sum((x-y)**2)/x.size
            return mse

    def mean_squared_error(self,field):
        """A wrapper to calculate the mean squared error between the fields of adjacent simulations for a run
        and write the results to a file"""
        self.log.info('Running the mean squared error wrapper for quantity %s',field) 
        for group in self.sim_groups:
            base = group[0].get('General','basedir')
            with open(os.path.join(base,'mse_%s.dat'%field),'w') as errfile:
                # Compare all other sims to our best estimate, which is sim with highest number of
                # basis terms (last in list cuz sorting)
                self.sim = group[-1]
                self.get_data()
                # Get the proper file extension depending on the field.
                if field == 'E':
                    ext = '.E'
                    vec = self.normEsquared()
                    avg = np.mean(vec)
                    # Get the comparison vector
                    vec1 = self.e_data[:,3:9]
                elif field == 'H':
                    ext = '.H'
                    vec = self.normHsquared()
                    avg = np.mean(vec)
                    vec1 = self.h_data[:,3:9]
                else:
                    self.log.error('The quantity for which you want to compute the error has not yet been calculated')
                    quit()
                   
                # For all other sims in the groups, compare to best estimate and write to error file 
                for i in range(0,len(group)-1):
                    sim2 = self.sims[i]
                    path2 = os.path.join(sim2.get('General','sim_dir'),
                                         sim2.get('General','base_name')+ext)
                    vec2 = np.loadtxt(path2,usecols=range(3,9))
                    #self.log.info("%s \n %s",str(vec1),str(vec2))
                    self.log.info("Computing error between numbasis %i and numbasis %i",
                                  self.sim.getint('Parameters','numbasis'),
                                  sim2.getint('Parameters','numbasis'))
                    error = self.mse(vec1,vec2)/avg
                    self.log.info(str(error))
                    errfile.write('%i,%f\n'%(sim2.getint('Parameters','numbasis'),error))

class Plotter(Processor):
    """Plots all the things listed in the config file"""
    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug("This is the plotter")
    
    def get_data(self):
        super().get_data()
        # But also build the lookup table that maps quantity names to column numbers
        epath = os.path.join(self.sim.get('General','sim_dir'),
                             self.sim.get('General','base_name')+'.E')
        hpath = os.path.join(self.sim.get('General','sim_dir'),
                             self.sim.get('General','base_name')+'.H')
        with open(epath,'r') as efile:
            e_headers = efile.readlines()[0].strip('#\n').split(',')
        with open(hpath,'r') as hfile:
            h_headers = hfile.readlines()[0].strip('#\n').split(',')
        self.e_lookup = {e_headers[ind]:ind for ind in range(len(e_headers))}
        self.h_lookup = {h_headers[ind]:ind for ind in range(len(h_headers))}
        self.log.debug('Here is the E field header lookup: %s',str(self.e_lookup))
        self.log.debug('Here is the H field header lookup: %s',str(self.h_lookup))

    def plot(self):
        self.log.info("Beginning local plotter method ...")
        for plot,args in self.gconf.items('Plotter'):
            self.log.info('Plotting %s with args %s',str(plot),str(args))
            # This is a special case because we need to compute error between sims and we need
            # to make sure all our other quantities have been calculated before we can compare
            # them
            for sim in self.sims:
                # Set it as the current sim and grab its data
                self.sim = sim
                self.get_data()
                # For each plot 
                self.log.info('Plotting data for sim %s',
                              str(os.path.basename(self.sim.get('General','sim_dir'))))
                for argset in args.split(';'):
                    self.log.info('Passing following arg set to function %s: %s',str(plot),str(argset))
                    if argset:
                        self.gen_plot(plot,argset.split(','))
                    else:
                        self.gen_plot(plot,[])

    def gen_plot(self,plot,args):
        try:
            getattr(self,plot)(*args)
        except KeyError:
            self.log.error("You have attempted to calculate an unsupported quantity!!")
            quit()

    def get_scalar_quantity(self,quantity):
        self.log.debug('Retrieving scalar quantity %s',str(quantity))
        try:
            col = self.e_lookup[quantity]
            self.log.debug('Column of E field quantity %s is %s',str(quantity),str(col))
            return self.e_data[:,col]
        except KeyError:
            col = self.h_lookup[quantity]
            self.log.debug('Column of H field quantitty %s is %s',str(quantity),str(col))
            return self.h_data[:,col]
        except KeyError:
            self.log.error('You attempted to retrieve a quantity that does not exist in the e and h \
                    matrices')
            quit()

    def heatmap2d(self,x,y,cs,labels,ptype,draw=False,fixed=None,colorsMap='jet'):
        """A general utility method for plotting a 2D heat map"""
        cm = plt.get_cmap(colorsMap)
        if fixed:
            cNorm = matplotlib.colors.Normalize(vmin=np.amin(5.0), vmax=np.amax(100.0))
        else:
            cNorm = matplotlib.colors.Normalize(vmin=np.amin(cs), vmax=np.amax(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111)
        ax.pcolormesh(x, y, cs,cmap=cm,norm=cNorm,alpha=.5)
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[-1])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start,end,0.1))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start,end,0.1))
        fig.suptitle(os.path.basename(self.sim.get('General','sim_dir')))
        # Draw geometric indicators and labels
        if draw:
            if ptype[-1] == 'z':
                self.log.info('draw nanowire circle')
                cent = self.sim.getfloat('Parameters','array_period')/2.0
                rad = self.sim.getfloat('Parameters','nw_radius')
                circ = mpatches.Circle((cent,cent),radius=rad,fill=False)
                ax.add_artist(circ)
            elif ptype[-1] == 'y' or ptype[-1] == 'x':
                self.log.info('draw layers')
                # Draw a line at the interface between each layer
                ito_line = self.sim.getfloat('Parameters','air_t')
                nw_line = self.sim.getfloat('Parameters','ito_t')+ito_line
                sub_line = self.sim.getfloat('Parameters','nw_height')+nw_line
                air_line = sub_line+self.sim.getfloat('Parameters','substrate_t')
                for line_h in [(ito_line,'ITO'),(nw_line,'NW'),(sub_line,'Substrate'),(air_line,'Air')]:
                    x = [0,self.sim.getfloat('Parameters','array_period')]
                    y = [line_h[0],line_h[0]]
                    label_y = line_h[0] + 0.01
                    label_x = x[-1]
                    plt.text(label_x,label_y,line_h[-1],ha='right',family='sans-serif',size=12)
                    line = mlines.Line2D(x,y,linestyle='solid',linewidth=2.0,color='black')
                    ax.add_line(line)
                # Draw two vertical lines to show the edges of the nanowire
                cent = self.sim.getfloat('Parameters','array_period')/2.0
                rad = self.sim.getfloat('Parameters','nw_radius')
                shell = self.sim.getfloat('Parameters','shell_t')
                bottom = self.sim.getfloat('Parameters','ito_t')+ito_line
                top = self.sim.getfloat('Parameters','nw_height')+nw_line
                for x in (cent-rad,cent+rad,cent-rad-shell,cent+rad+shell):
                    xv = [x,x]
                    yv = [bottom,top]
                    line = mlines.Line2D(xv,yv,linestyle='solid',linewidth=2.0,color='black')
                    ax.add_line(line)
        if self.gconf.getboolean('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)

    def plane_2d(self,quantity,plane,pval,draw=False,fixed=None):
        """Plots a heatmap of a fixed 2D plane"""
        if plane == 'x' or plane == 'y':
            pval = int(pval)
        else:
            # Find the nearest zval to the one we want. This is necessary because comparing floats
            # rarely works
            height = self.sim.getfloat('Parameters','total_height')
            z_samples = self.sim.getint('General','z_samples')
            desired_val = (height/z_samples)*int(pval)
            ind = np.abs(self.pos_inds[:,2]-desired_val).argmin()
            pval = self.pos_inds[ind,2]
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # Maps planes to an integer for extracting data
        plane_table = {'x': 0,'y': 1,'z':2}
        # Get the scalar values
        scalar = self.get_scalar_quantity(quantity)
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((self.pos_inds[:,0],self.pos_inds[:,1],self.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[plane_table[plane]] == pval])
        # Get all unique values for x,y,z and convert them to actual values not indices
        x,y,z = np.unique(planes[:,0])*dx,np.unique(planes[:,1])*dy,np.unique(planes[:,2])
        # Super hacky and terrible way to fix the minimum and maximum values of the color bar
        # for a plot across all sims
        if fixed:
            fixed = tuple(fixed.split(':'))
        if plane == 'x':
            cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
            labels = ('y [um]','z [um]', quantity)
            self.heatmap2d(y,z,cs,labels,'plane_2d_x',draw,fixed)
        elif plane == 'y':
            cs = planes[:,-1].reshape(z.shape[0],x.shape[0])
            labels = ('x [um]','z [um]', quantity)
            self.heatmap2d(x,z,cs,labels,'plane_2d_y',draw,fixed)
        elif plane == 'z':
            cs = planes[:,-1].reshape(y.shape[0],x.shape[0])
            labels = ('y [um]','x [um]', quantity)
            self.heatmap2d(x,y,cs,labels,'plane_2d_z',draw,fixed)
    
    def scatter3d(self,x,y,z,cs,labels,ptype,colorsMap='jet'):
        """A general utility method for scatter plots in 3D"""
        #fig = plt.figure(figsize=(8,6)) 
        #ax = fig.add_subplot(111,projection='3d')
        #colors = cm.hsv(E_mag/max(E_mag))
        #colmap = cm.ScalarMappable(cmap=cm.hsv)
        #colmap.set_array(E_mag)
        #yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
        #cb = fig.colorbar(colmap)
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9,7))
              
        #ax = Axes3D(fig)
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs),edgecolor='none')
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[-1])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        fig.suptitle(os.path.basename(self.sim.get('General','sim_dir')))
        if self.gconf.getboolean('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)
    
    def full_3d(self,quantity):
        """Generates a full 3D plot of a specified scalar quantity"""
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # The data just tells you what integer grid point you are on. Not what actual x,y coordinate you
        # are at
        xpos = self.pos_inds[:,0]*dx
        ypos = self.pos_inds[:,1]*dy
        zpos = self.pos_inds[:,2] 
        # Get the scalar
        scalar = self.get_scalar_quantity(quantity)
        labels = ('X [um]','Y [um]','Z [um]',quantity)
        # Now plot! 
        self.scatter3d(xpos,ypos,zpos,scalar,labels,'full_3d')
        
    def planes_3d(self,quantity,xplane,yplane):
        """Plots some scalar quantity in 3D but only along specified x-z and y-z planes"""
        xplane = int(xplane)
        yplane = int(yplane)
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # Get the scalar values
        scalar = self.get_scalar_quantity(quantity) 
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((self.pos_inds[:,0],self.pos_inds[:,1],self.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[0] == xplane or row[1] == yplane])
        planes[:,0] = planes[:,0]*dx
        planes[:,1] = planes[:,1]*dy
        labels = ('X [um]','Y [um]','Z [um]',quantity)
        # Now plot!
        self.scatter3d(planes[:,0],planes[:,1],planes[:,2],planes[:,3],labels,'planes_3d')

    def line_plot(self,x,y,ptype,labels):
        """Make a simple line plot"""
        fig = plt.figure()
        plt.plot(x,y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(os.path.basename(self.sim.get('General','sim_dir')))
        if self.gconf.getboolean('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.getboolean('General','show_plots'):
            plt.show()
        plt.close(fig)

    def fixed_line(self,quantity,direction,coord1,coord2):
        """Plot a scalar quantity on a line along a the z direction at some pair of
        coordinates in the plane perpendicular to that direction"""
        coord1 = int(coord1)
        coord2 = int(coord2)
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # Get the scalar values
        scalar = self.get_scalar_quantity(quantity) 
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((self.pos_inds[:,0],self.pos_inds[:,1],self.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[0] == coord1 and row[1] == coord2])
        planes[:,0] = planes[:,0]*dx
        planes[:,1] = planes[:,1]*dy
        labels = ('Z [um]',quantity) 
        ptype = "%s_line_plot_%i_%i"%(direction,coord1,coord2)
        self.line_plot(planes[:,2],planes[:,3],ptype,labels)

class Global_Plotter(Plotter):
    """Plots global quantities for an entire run that are not specific to a single simulation"""
    def __init__(self,global_conf):
        super().__init__(global_conf)
        self.log.debug("Global plotter init")

    def plot(self):
        self.log.info('Beginning global plotter method ...')
        for plot,args in self.gconf.items('Global_Plotter'):
            self.log.info('Plotting %s with args %s',str(plot),str(args))
            for argset in args.split(';'):
                self.log.info('Passing following arg set to function %s: %s',str(plot),str(argset))
                if argset:
                    self.gen_plot(plot,argset.split(','))
                else:
                    self.gen_plot(plot,[])

    def convergence(self,quantity,scale='linear'):
        """Plots the convergence of a field across all available simulations"""
        self.log.info('Actually plotting convergence')
        for group in self.sim_groups:
            base = group[0].get('General','basedir')
            path = os.path.join(base,'mse_%s.dat'%quantity) 
            labels = []
            errors = []
            with open(path,'r') as datf:
                for line in datf.readlines():
                    lab, err = line.split(',')
                    labels.append(lab)
                    errors.append(err)
            x = range(len(errors))
            fig = plt.figure(figsize=(9,7))
            plt.ylabel('M.S.E of %s'%quantity)
            plt.plot(labels,errors,linestyle='-',marker='o',color='b')
            plt.yscale(scale)
            #plt.xticks(x,labels,rotation='vertical')
            plt.tight_layout()
            plt.title(os.path.basename(self.gconf.get('General','basedir')))
            if self.gconf.getboolean('General','save_plots'):
                name = os.path.basename(base)+'_convergence_'+quantity+'.pdf'
                path = os.path.join(base,name)
                fig.savefig(path)
            if self.gconf.getboolean('General','show_plots'):
                plt.show() 
            plt.close(fig)

def main():
    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('-nc','--no_crunch',action="store_true",default=False,help="""Do not perform crunching
            operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-ngc','--no_gcrunch',action="store_true",default=False,help="""Do not
            perform global crunching operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-np','--no_plot',action="store_true",default=False,help="""Do not perform plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('-ngp','--no_gplot',action="store_true",default=False,help="""Do not perform global plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('--log_level',type=str,default='info',choices=['debug','info','warning','error','critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()
    
    # Configure logger
    logger = configure_logger(args.log_level,'postprocess',
                              os.path.join(conf.get('General','basedir'),'logs'),
                              'postprocess.log')

    # Now do all the work
    if not args.no_crunch:
        crunchr = Cruncher(conf)
        crunchr.crunch()
    if not args.no_gcrunch:
        gcrunchr = Global_Cruncher(conf)
        gcrunchr.crunch()
    if not args.no_plot:
        pltr = Plotter(conf) 
        pltr.plot()
    if not args.no_gplot:
        gpltr = Global_Plotter(conf)
        gpltr.plot()

if __name__ == '__main__':
    main()
