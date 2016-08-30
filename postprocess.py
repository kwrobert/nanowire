import numpy as np
from scipy import interpolate
from scipy import constants
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import argparse as ap
import os
import configparser as confp 
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


class Cruncher():
    """Crunches all the raw data. Calculates quantities specified in the global config file and
    either appends them to the existing data files or creates new ones as needed"""

    def __init__(self,global_conf):
        print("This is THE CRUNCHER!!!!!")
        self.gconf = global_conf
        simdirs = [x[0] for x in os.walk(self.gconf.get('General','basedir'))]
        self.sims = [parse_file(os.path.join(simdir,'sim_conf.ini')) for simdir in simdirs[1:]]
        # Sort on the sim dir to prevent weirdness when calculating convergence. Sorts by ascending
        # param values and works for multiple variable params
        self.sort_sims()
        self.sim = None

    def sort_sims(self):
        """Sorts simulations by their parameters the way a human would. Called human sorting or
        natural sorting. Thanks stackoverflow"""
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(sim):
            text = sim.get('General','sim_dir')
            return [ atoi(c) for c in re.split('(\d+)', text) ]
        self.sims.sort(key=natural_keys)

    def crunch(self):
        mse = False
        for sim in self.sims:
            # Set it as the current sim and grab its data
            self.sim = sim
            self.get_data()
            # For each quantity 
            for quant,args in self.gconf.items('Cruncher'):
                # This is a special case because we need to compute error between sims and we need
                # to make sure all our other quantities have been calculated before we can compare
                # them
                if quant == 'mean_squared_error':
                    mse = True
                else:
                    if args:
                        self.calculate(quant,args.split(','))
                    else:
                        self.calculate(quant,[])
            self.write_data()
        if mse:
            self.mse_wrap(self.gconf.get('Cruncher','mean_squared_error'))
    
    def get_data(self):
        """Gets the E and H data for this particular sim"""
        sim_path = self.sim.get('General','sim_dir')
        base_name = self.gconf.get('General','base_name')
        e_path = os.path.join(sim_path,base_name+'.E')
        h_path = os.path.join(sim_path,base_name+'.H')
        e_data = np.loadtxt(e_path)
        h_data = np.loadtxt(h_path)
        self.pos_inds,self.e_data = np.array_split(e_data,[3],axis=1)
        self.h_data = np.array_split(h_data,[3],axis=1)[-1] 

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
        os.rename(epath,epath+".raw")
        os.rename(hpath,hpath+".raw")
        # Build the full matrices
        full_emat = np.column_stack((self.pos_inds,self.e_data))
        full_hmat = np.column_stack((self.pos_inds,self.h_data))
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
        np.savetxt(epath,full_emat,header=','.join(eheader))
        np.savetxt(hpath,full_hmat,header=','.join(hheader))
    
    def normE(self):
        """Calculate and returns the norm of E"""
        
        if not hasattr(self,'e_data'):
            print()
            print("You need to get your data first!")
            print()
            quit()
        
        # Get the magnitude of E and add it to our data
        #Ex = data[:,3] + 1j*data[:,6]
        #Ey = data[:,4] + 1j*data[:,7]
        #Ez = data[:,5] + 1j*data[:,8]
        #E_mag = np.sqrt(np.absolute(Ex)+np.absolute(Ey)+np.absolute(Ez))
        
        # Grab only the real parts. I think this is right. 
        Ex = self.e_data[:,0]
        Ey = self.e_data[:,1]
        Ez = self.e_data[:,2]
        E_mag = np.sqrt(Ex**2+Ey**2+Ez**2)
        self.e_data = np.column_stack((self.e_data,E_mag))
                                     
    def normH(self):
        """Calculate and returns the norm of H"""
        
        if not hasattr(self,'h_data'):
            print()
            print("You need to get your data first!")
            print()
            quit()
        
        # Get the magnitude of E and add it to our data
        #Ex = data[:,3] + 1j*data[:,6]
        #Ey = data[:,4] + 1j*data[:,7]
        #Ez = data[:,5] + 1j*data[:,8]
        #E_mag = np.sqrt(np.absolute(Ex)+np.absolute(Ey)+np.absolute(Ez))
        
        # Grab only the real parts. I think this is right. 
        Hx = self.h_data[:,0]
        Hy = self.h_data[:,1]
        Hz = self.h_data[:,2]
        H_mag = np.sqrt(Hx**2+Hy**2+Hz**2)
        self.h_data = np.column_stack((self.h_data,H_mag))

    def mse_wrap(self,quant):
        """A wrapper to calculate the mean squared error of a quantity between simulations for a run
        and write the results to a file"""
        
        with open(os.path.join(self.gconf.get('General','basedir'),'mse_%s.dat'%quant),'w') as errfile:
            errors = []
            for i in range(1,len(self.sims)):
                sim1 = self.sims[i-1]
                sim2 = self.sims[i]
                epath1 = os.path.join(sim1.get('General','sim_dir'),sim1.get('General','base_name')+'.E')
                hpath1 = os.path.join(sim1.get('General','sim_dir'),sim1.get('General','base_name')+'.H')
                epath2 = os.path.join(sim2.get('General','sim_dir'),sim2.get('General','base_name')+'.E')
                hpath2 = os.path.join(sim2.get('General','sim_dir'),sim2.get('General','base_name')+'.H')
                with open(epath1,'r') as efile:
                    eheads = efile.readlines()[0].strip('#\n').split(',')
                with open(hpath1,'r') as hfile:
                    hheads = hfile.readlines()[0].strip('#\n').split(',')
                if quant in eheads:
                    ind = eheads.index(quant)
                    dat1 = np.loadtxt(epath1)
                    dat2 = np.loadtxt(epath2)
                    vec1 = dat1[:,ind]
                    vec2 = dat2[:,ind]
                    error = self.mean_squared_error(vec1,vec2)
                    errors.append(error)
                elif quant in hheads:
                    ind = hheads.index(quant)
                    dat1 = np.loadtxt(hpath1)
                    dat2 = np.loadtxt(hpath2)
                    vec1 = dat1[:,ind]
                    vec2 = dat2[:,ind]
                    error = self.mean_squared_error(vec1,vec2)
                    errors.append(error)
                else:
                    print('The quantity for which you want to compute the error has not yet been \
                            calculated')
                    quit()
                errfile.write('%s-%s,%f\n'%(os.path.basename(sim1.get('General','sim_dir')),os.path.basename(sim2.get('General','sim_dir')),error))
        plt.figure()
        plt.plot(range(len(errors)),errors)
        plt.show()
                
    def mean_squared_error(self,x,y):
        """Return the mean squared error between two equally sized sets of data"""
        if x.size != y.size:
            print("You have attempted to compare datasets with an unequal number of points!!!!")
            quit()
        else:
            print(x)
            print(y)
            mse = sum((x-y)**2)/x.size
            print(mse)
            return mse

class Plotter():

    def __init__(self,global_conf):
        self.gconf = global_conf
        simdirs = [x[0] for x in os.walk(self.gconf.get('General','basedir'))]
        self.sims = [parse_file(os.path.join(simdir,'sim_conf.ini')) for simdir in simdirs[1:]]
        self.sim = None
        
    def convergence(self,quantity):
        """Calculates the convergence of a given quantity across all available simulations"""
        errors = []
        sims_srt = self.sims.sort()
        print(sims_srt)
        quit()
        for i in range(1,len(self.sims)):
            self.sim = self.sims[i-1]
            self.get_data()
            scalar1 = self.get_scalar_quantity(quantity)
            self.sim = self.sims[i]
            self.get_data()
            scalar2 = self.get_scalar_quantity(quantity)
            mse = self.mean_squared_error(scalar1,scalar2)
            errors.append(mse)
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111)
        plt.plot(range(len(errors)),errors)
        if self.gconf.get('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.get('General','show_plots'):
            plt.show() 

    def heatmap2d(self,x,y,cs,labels,ptype,colorsMap='jet'):
        """A general utility method for plotting a 2D heat map"""
        cm = plt.get_cmap(colorsMap)
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
        fig.suptitle(os.path.basename(self.sim.get('General','sim_dir')))
        if self.gconf.get('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.get('General','show_plots'):
            plt.show()

    def plane_2d(self,quantity,plane,pval):
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
        if plane == 'x':
            cs = planes[:,-1].reshape(z.shape[0],y.shape[0])
            labels = ('y [um]','z [um]', quantity)
            self.heatmap2d(y,z,cs,labels,'plane_2d')
        elif plane == 'y':
            cs = planes[:,-1].reshape(z.shape[0],x.shape[0])
            labels = ('x [um]','z [um]', quantity)
            self.heatmap2d(x,z,cs,labels,'plane_2d')
        elif plane == 'z':
            cs = planes[:,-1].reshape(y.shape[0],x.shape[0])
            labels = ('y [um]','x [um]', quantity)
            self.heatmap2d(x,y,cs,labels,'plane_2d')
    
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
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[-1])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        fig.suptitle(os.path.basename(self.sim.get('General','sim_dir')))
        if self.gconf.get('General','save_plots'):
            name = labels[-1]+'_'+ptype+'.pdf'
            path = os.path.join(self.sim.get('General','sim_dir'),name)
            fig.savefig(path)
        if self.gconf.get('General','show_plots'):
            plt.show()
    
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
    

def main():
    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")

    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()
   
    
    # Instantiate objects
    crunchr = Cruncher(conf)
    pltr = Plotter(conf) 
    crunchr.crunch()
    ## Loop through each individual simulation
    #for sim in pp.sims:
    #    # Set it as the current sim and grab its data
    #    pp.sim = sim
    #    pp.get_data()
    #    # For each plot we want, and the list of quantities to be plotted using that plot type
    #    for plot, quantities in pp.gconf.items('Postprocess'):
    #        if plot != 'convergence':
    #            print(quantities)
    #            # For each group of arguments, grab the proper plot method bound to this instance of
    #            # PostProcess using getattr() built-in. Unpack the arguments provided by the config
    #            # file, and pass them in to the plot method. 
    #            for args in quantities.split(';'):
    #                tmp = args.split(',')
    #                print(tmp)
    #                try:
    #                    getattr(pp,plot)(*tmp)
    #                except KeyError:
    #                    print()
    #                    print("You have attempted to plot an unsupported plot type!")
    #                    print()
    #                    quit()

    #if pp.gconf.has_option('Postprocess','convergence'):
    #    pp.convergence(pp.gconf.get('Postprocess','convergence'))

if __name__ == '__main__':
    main()
