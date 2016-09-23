import S4
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

def calcnormE(data):
    Ex = data[:,3] + 1j*data[:,6]
    Ey = data[:,4] + 1j*data[:,7]
    Ez = data[:,5] + 1j*data[:,8]
    E_mag = np.sqrt(Ex*np.conj(Ex)+Ey*np.conj(Ey)+Ez*np.conj(Ez))
    data = np.column_stack((data,E_mag.real))
    return data, E_mag.real

def plot_sim():
    data = np.loadtxt('test_fields.E')
    data, normE = calcnormE(data)
    print(normE)
    np.savetxt('processed_data.txt',data)
    pval = 50
    mat = np.column_stack((data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]))
    planes = np.array([row for row in mat if row[0] == pval])
    print(planes.shape)
    dx = dy = .63/200
    x,y,z = np.unique(planes[:,0])*dx,np.unique(planes[:,1])*dy,np.unique(planes[:,2])
    print(z.shape)
    print(y.shape)
    print(planes[:,-1].size)
    normE = planes[:,-1].reshape(z.shape[0],y.shape[0])
    # Plot stuff
    colorsMap = 'jet' 
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=np.amin(normE), vmax=np.amax(normE))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.pcolormesh(y, z, normE,cmap=cm,norm=cNorm,alpha=.5)
    scalarMap.set_array(normE)
    cb = fig.colorbar(scalarMap)
    labels = ('y','z','normE')
    cb.set_label(labels[-1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start,end,0.1))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start,end,0.1))
    fig.suptitle('Electric field') 
    plt.show()
    
    line = np.array([row for row in mat if row[0] == pval and row[1] == pval])
    print(line)
    plt.plot(line[:,2],line[:,-2],'r-',label='Ex Real')
    plt.plot(line[:,2],line[:,-1],'b-',label='Ey Real')
    plt.legend()
    plt.show()

def run_air_sim():
    sim = S4.New(Lattice=((.63,0),(0,.63)),NumBasis=25)
    sim.SetOptions(Verbosity=2)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
    sim.AddLayer(Name='air1',Thickness=.5,Material='vacuum')
    sim.AddLayer(Name='air2',Thickness=.5,Material='vacuum')
    # Set frequency
    f_phys = 3E14 
    c_conv = constants.c/10E-6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('f_conv = ',f_conv)
    sim.SetFrequency(f_conv)
    E_mag = 1.0 
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0), pAmplitude=complex(0,E_mag))
    x_samp = 200 
    y_samp = 200 
    z_samp = 200
    height = 1.0   
    for z in np.linspace(0,height,z_samp):
        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
                            Format='FileAppend',BaseFilename='test_fields')

def run_airito_sim():
    sim = S4.New(Lattice=((.63,0),(0,.63)),NumBasis=25)
    sim.SetOptions(Verbosity=2)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
    sim.SetMaterial(Name='ito',Epsilon=complex(2.0766028416,0.100037324))
    sim.AddLayer(Name='air1',Thickness=.5,Material='vacuum')
    sim.AddLayer(Name='ito',Thickness=.5,Material='ito')
    # Set frequency
    f_phys = 3E14 
    c_conv = constants.c/10E-6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('f_conv = ',f_conv)
    sim.SetFrequency(f_conv)
    E_mag = 1.0 
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0), pAmplitude=complex(0,E_mag))
    x_samp = 200 
    y_samp = 200 
    z_samp = 200
    height = 1.0   
    for z in np.linspace(0,height,z_samp):
        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
                            Format='FileAppend',BaseFilename='test_fields')

def main():
    run_airito_sim()
    print('Finished sim')
    plot_sim()
    
if __name__ == '__main__':
    main()
