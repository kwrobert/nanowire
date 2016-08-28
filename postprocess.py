import numpy as np
from scipy import interpolate
from scipy import constants
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import argparse as ap
import os
import ConfigParser as confp 
import glob

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    return parser

def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    print(cs)
    #fig = plt.figure(figsize=(8,6)) 
    #ax = fig.add_subplot(111,projection='3d')
    #colors = cm.hsv(E_mag/max(E_mag))
    #colmap = cm.ScalarMappable(cmap=cm.hsv)
    #colmap.set_array(E_mag)
    #yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
    #cb = fig.colorbar(colmap)
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    plt.show()

class PostProcess(object):

    def __init__(self,global_conf):
        self.gconf = global_conf
        simdirs = [x[0] for x in os.walk(self.gconf.get('General','basedir'))]
        self.sims = [parse_file(os.path.join(simdir,'sim_conf.ini')) for simdir in simdirs[1:]]
        

    def get_data(self):
        """Gets the E and H data for this particular sim"""
        sim_path = self.sim.get('General','sim_dir')
        base_name = self.gconf.get('General','base_name')
        e_path = os.path.join(sim_path,base_name+'.E')
        h_path = os.path.join(sim_path,base_name+'.H')
        e_data = np.loadtxt(e_path)
        h_data = np.loadtxt(h_path)
        self.pos_inds,self.e_data = np.array_split(e_data,[3],axis=1)
        self.h_data = np.array_split(e_data,[3],axis=1)[-1] 

    def get_scalar_quantity(self,quantity):
        try:
            val_vec = getattr(self,quantity)()
        except KeyError:
            print() 
            print("You have attempted to calculate and unsupported quantity!!")
            print("Please modify your global config file to choose from the following options")
            print(self.quantities.keys())
            quit()
        return val_vec 
    
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
        
        return E_mag
                                     
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
        
        return H_mag

    
    
    
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
        print(scalar)
        # Now plot! 
        scatter3d(xpos,ypos,zpos,scalar)
        
    def planes_3d(self,quantity,xplane,yplane):
        """Plots some scalar quantity in 3D but only along specified x-z and y-z planes"""
        xplane = int(xplane)
        yplane = int(yplane)
        period = self.sim.getfloat('Parameters','array_period')
        dx = period/self.gconf.getfloat('General','x_samples')
        dy = period/self.gconf.getfloat('General','y_samples')
        # Get the scalar values
        scalar = self.get_scalar_quantity(quantity) 
        print(scalar)
        # Filter out any undesired data that isn't on the planes
        mat = np.column_stack((self.pos_inds[:,0],self.pos_inds[:,1],self.pos_inds[:,2],scalar))
        planes = np.array([row for row in mat if row[0] == xplane or row[1] == yplane])
        print planes.shape
        print(planes)
        planes[:,0] = planes[:,0]*dx
        planes[:,1] = planes[:,1]*dy
        # Now plot!
        scatter3d(planes[:,0],planes[:,1],planes[:,2],planes[:,3])

#plots = {'full_threeD':PostProcess.full_threeD,'threeD_planes':PostProcess.threeD_planes}
#quantities = {'normE':PostProcess.normE,'normH':PostProcess.normH}

def main():
    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")

    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = parse_file(os.path.abspath(args.config_file))
    else:
        print "\n The file you specified does not exist! \n"
        quit()
   
    
    pp = PostProcess(conf) 
    for sim in pp.sims:
        pp.sim = sim
        pp.get_data()
        for plot, quantities in pp.gconf.items('Postprocess'):
            print(quantities)
            for args in quantities.split(';'):
                tmp = args.split(',')
                print(tmp)
                getattr(pp,plot)(*tmp)


if __name__ == '__main__':
    main()
