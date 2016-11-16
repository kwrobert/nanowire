#import S4
import numpy as np
from scipy import interpolate
from scipy import constants
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import argparse as ap
import os

def get_epsilon(freq,path):
    """Returns complex dielectric constant for a material by pulling in nk text file, interpolating,
    computing nk values at freq, and converting"""
    # Get data
    freq_vec,n_vec,k_vec = np.loadtxt(path,skiprows=1,unpack=True)
    # Get n and k at specified frequency via interpolation 
    f_n = interpolate.interp1d(freq_vec,n_vec)
    f_k = interpolate.interp1d(freq_vec,k_vec)
    n,k = f_n(freq),f_k(freq)
    # Convert to dielectric constant
    # NOTE!!: This assumes the relative magnetic permability (mew) is 1
    epsilon_real = n**2 - k**2
    epsilon_imag = 2*n*k
    epsilon = complex(epsilon_real,epsilon_imag)
    # NOTE: Include below in some sort of unit test later
    #print("n = %f"%n)
    #print("k = %f"%k)
    #print('epsilon = {:.2f}'.format(epsilon))
    #plt.plot(freq_vec,k_vec,'bs',freq_vec,f_k(freq_vec),'r--')
    #plt.show()
    return epsilon

def calcnormE(data):
    #Ex = data[:,3] + 1j*data[:,6]
    #Ey = data[:,4] + 1j*data[:,7]
    #Ez = data[:,5] + 1j*data[:,8]
    #E_mag = np.sqrt(Ex*np.conj(Ex)+Ey*np.conj(Ey)+Ez*np.conj(Ez))
    E_mag = np.zeros_like(data[:,3])
    for i in range(3,9):
        E_mag += data[:,i]*data[:,i]
    E_mag = np.sqrt(E_mag)
    new_data = np.zeros((data.shape[0],data.shape[1]+1))
    new_data[:,:-1] = data
    new_data[:,-1] = E_mag
    return new_data, E_mag

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

def get_comsol(path):
    conv = {col: comp_conv for col in range(0,2)}
    raw = np.loadtxt(path,comments='#',converters=conv,dtype=complex)
    return raw

def plot_sim(p,save):
    # Load the data
    coms_dat = get_comsol('/home/kyle_robertson/schoolwork/gradschool/nanowire/code/optics/comsol_data_fixed.txt')
    data = np.loadtxt('test_fields.E')
    mat, normE = calcnormE(data)
    np.savetxt('processed_data.txt',data)
    ## Planar plot of norm of E
    pval = p['plane']
    #mat = np.column_stack((data,normE))
    #planes = np.array([row for row in mat if row[0] == pval])
    #dx = p['L']/p['x_samp']
    #dy = p['L']/p['y_samp']
    #x,y,z = np.unique(planes[:,0])*dx,np.unique(planes[:,1])*dy,np.unique(planes[:,2])
    #normE = planes[:,-1].reshape(z.shape[0],y.shape[0])
    #colorsMap = 'jet' 
    #cm = plt.get_cmap(colorsMap)
    #cNorm = matplotlib.colors.Normalize(vmin=np.amin(normE), vmax=np.amax(normE))
    #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    #fig = plt.figure(1,figsize=(9,7))
    #ax = fig.add_subplot(111)
    #ax.pcolormesh(y, z, normE,cmap=cm,norm=cNorm,alpha=.5)
    #scalarMap.set_array(normE)
    #cb = fig.colorbar(scalarMap)
    #labels = ('y','z','normE')
    #cb.set_label(labels[-1])
    #ax.set_xlabel(labels[0])
    #ax.set_ylabel(labels[1])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start,end,0.1))
    #start, end = ax.get_ylim()
    #ax.yaxis.set_ticks(np.arange(start,end,0.1))
    #fig.suptitle('Electric field Norm') 
    #if save:
    #    plt.savefig('normE_plane_plot.pdf')
    #plt.show()
    # Plot along line 
    line = np.array([row for row in mat if row[0] == pval and row[1] == pval])
    #plt.figure(2)
    #plt.plot(line[:,2],line[:,-1],'b-')
    #plt.xlabel('z [um]')
    #plt.ylabel('norm E')
    #plt.title('Norm of Electric Field along Line')
    #plt.show()
    #if save:
    #    plt.savefig('normE_line_plot.pdf')
    # Plot real and imaginary components
    fig, axes = plt.subplots(2,2)
    axes[0,0].plot(line[:,2],line[:,3],'r-',label='Ex Real S4')
    axes[0,0].plot(line[:,2],line[:,5],'b-',label='Ey Real S4')
    axes[0,0].legend()
    axes[0,1].plot(line[:,2],line[:,4],'g-',label='Ex Imag S4')
    axes[0,1].plot(line[:,2],line[:,6],'m-',label='Ey Imag S4')
    axes[0,1].legend()
    axes[1,0].plot(range(coms_dat.shape[0]),np.real(coms_dat[:,0]),'r-',label='Ex Real COMS')
    axes[1,0].plot(range(coms_dat.shape[0]),np.real(coms_dat[:,1]),'b-',label='Ey Real COMS')
    axes[1,0].legend()
    axes[1,1].plot(range(coms_dat.shape[0]),np.imag(coms_dat[:,0]),'g-',label='Ex Imag COMS')
    axes[1,1].plot(range(coms_dat.shape[0]),np.imag(coms_dat[:,1]),'m-',label='Ey Imag COMS')
    axes[1,1].legend()
    fig.suptitle('Real and Imag Components on Line')
    if save:
        plt.savefig('real_imag_line.pdf')
    plt.show()
    # Plot components magnitudes 
    #plt.figure(3)
    #ex_mag = np.sqrt(line[:,3]*line[:,3]+line[:,4]*line[:,4])
    #ey_mag = np.sqrt(line[:,5]*line[:,5]+line[:,6]*line[:,6])
    #plt.plot(line[:,2],ex_mag,'r-',label='Ex Magnitude')
    #plt.plot(line[:,2],ey_mag,'b-',label='Ey Magnitude')
    #plt.legend()
    #plt.title('Magnitudes on Line')
    #if save:
    #    plt.savefig('real_imag_line.pdf')
    #plt.show()
def run_air_sim(p,m):
    sim = S4.New(Lattice=((p['L'],0),(0,p['L'])),NumBasis=p['numbasis'])
    sim.SetOptions(Verbosity=2)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
    sim.AddLayer(Name='air1',Thickness=p['layer_t'],Material='vacuum')
    sim.AddLayer(Name='air2',Thickness=p['layer_t'],Material='vacuum')
    # Set frequency
    f_phys = p['freq'] 
    c_conv = constants.c*1E6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('wavelength = ',(constants.c/f_phys)*1E6)
    print('f_conv = ',f_conv)
    sim.SetFrequency(f_conv)
    E_mag = 1.0 
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0),
            pAmplitude=complex(0,-E_mag))
    x_samp = p['x_samp'] 
    y_samp = p['y_samp'] 
    z_samp = p['z_samp']
    height = p['layer_t']*2   
    for z in np.linspace(0,height,z_samp):
        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
                            Format='FileAppend',BaseFilename='test_fields')

def run_airito_sim(p,m):
    sim = S4.New(Lattice=((p['L'],0),(0,p['L'])),NumBasis=p['numbasis'])
    sim.SetOptions(Verbosity=2)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
    eps = get_epsilon(p['freq'],m['ito'])
    sim.SetMaterial(Name='ito',Epsilon=eps)
    sim.AddLayer(Name='air1',Thickness=p['layer_t'],Material='vacuum')
    sim.AddLayer(Name='ito',Thickness=p['layer_t'],Material='ito')
    sim.AddLayerCopy('air2',Thickness=p['layer_t'],Layer='air1')
    # Set frequency
    f_phys = p['freq'] 
    c_conv = constants.c*1E6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('wavelength = ',(constants.c/f_phys)*1E6)
    print('f_conv = ',f_conv)
    sim.SetFrequency(f_conv)
    E_mag = 1.0 
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0),
            pAmplitude=complex(0,E_mag))
    x_samp = p['x_samp'] 
    y_samp = p['y_samp'] 
    z_samp = p['z_samp']
    height = p['layer_t']*3  
    for z in np.linspace(0,height,z_samp):
        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
                            Format='FileAppend',BaseFilename='test_fields')

def run_airitosub_sim(p,m):
    sim = S4.New(Lattice=((p['L'],0),(0,p['L'])),NumBasis=p['numbasis'])
    sim.SetOptions(Verbosity=2)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
    eps = get_epsilon(p['freq'],m['ito'])
    sim.SetMaterial(Name='ito',Epsilon=eps)
    eps = get_epsilon(p['freq'],m['gaas'])
    sim.SetMaterial(Name='gaas',Epsilon=eps)
    sim.AddLayer(Name='air1',Thickness=p['layer_t'],Material='vacuum')
    sim.AddLayer(Name='ito',Thickness=p['layer_t'],Material='ito')
    sim.AddLayer(Name='gaas',Thickness=p['layer_t'],Material='gaas')
    sim.AddLayerCopy('air2',Thickness=p['layer_t'],Layer='air1')
    # Set frequency
    f_phys = p['freq'] 
    c_conv = constants.c*1E6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('wavelength = ',(constants.c/f_phys)*1E6)
    print('f_conv = ',f_conv)
    sim.SetFrequency(f_conv)
    E_mag = 1.0 
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0),
            pAmplitude=complex(0,0))
    x_samp = p['x_samp'] 
    y_samp = p['y_samp'] 
    z_samp = p['z_samp']
    height = p['layer_t']*4 
    for z in np.linspace(0,height,z_samp):
        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
                            Format='FileAppend',BaseFilename='test_fields')

def run_airitowiresub_sim(params):
    sim = S4.New(Lattice=((.63,0),(0,.63)),NumBasis=25)
    sim.SetOptions(Verbosity=2)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
    sim.SetMaterial(Name='ito',Epsilon=complex(2.0766028416,0.100037324))
    sim.SetMaterial(Name='gaas',Epsilon=complex(3.5384,0.0))
    sim.SetMaterial(Name='cyc',Epsilon=complex(1.53531,1.44205E-6))
    sim.AddLayer(Name='air1',Thickness=.5,Material='vacuum')
    sim.AddLayer(Name='ito',Thickness=.5,Material='ito')
    sim.AddLayer(Name='wire',Thickness=.5,Material='cyc')
    sim.SetRegionCircle('wire','gaas',(0,0),.2)
    sim.AddLayer(Name='gaas',Thickness=.5,Material='gaas')
    # Set frequency
    f_phys = 3E14 
    c_conv = constants.c/10E6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('wavelength = ',(constants.c/f_phys)*10E6)
    print('f_conv = ',f_conv)
    sim.SetFrequency(f_conv)
    E_mag = 1.0 
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0), pAmplitude=complex(0,E_mag))
    x_samp = 200 
    y_samp = 200 
    z_samp = 200
    height = 2.0 
    for z in np.linspace(0,height,z_samp):
        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
                            Format='FileAppend',BaseFilename='test_fields')

def main():
    parser = ap.ArgumentParser(description="""Sanity check for S4 library""")
    parser.add_argument('--air',action='store_true',help="Run air simulation")
    parser.add_argument('--ito',action='store_true',help="Run air-ito simulation")
    parser.add_argument('--sub',action='store_true',help="Run air-ito-substrate simulation")
    parser.add_argument('--save_plots',action='store_true',help="Save all generated plots")
    parser.add_argument('--plot',type=str,default='air',help="Plot stuff")
    args = parser.parse_args()

    params = {'freq':5E14,'layer_t':.5,'L':.25,'x_samp':50,'y_samp':50,'z_samp':600,'numbasis':40}
    plane = params['x_samp']/2
    params['plane'] = plane
    mats = {'ito':'/home/kyle_robertson/schoolwork/gradschool/nanowire/code/NK/008_ITO_nk_Hz.txt',
            'gaas':'/home/kyle_robertson/schoolwork/gradschool/nanowire/code/NK/006_GaAs_nk_Walker_modified_Hz.txt'}
    #os.mkdir('sanity_check_run')
    basedir = os.path.join(os.getcwd(),'sanity_check_run')
    if args.air:
        path = os.path.join(basedir,'air_sim')
        os.mkdir(path)
        os.chdir(path)
        run_air_sim(params,mats)
        print('Finished air sim')
        plot_sim(params,args.save_plots)
        os.chdir(basedir)
    if args.ito:
        path = os.path.join(basedir,'ito_sim')
        os.mkdir(path)
        os.chdir(path)
        run_airito_sim(params,mats)
        print('Finished ito sim')
        plot_sim(params,args.save_plots)
    if args.sub:
        path = os.path.join(basedir,'sub_sim')
        os.mkdir(path)
        os.chdir(path)
        run_airitosub_sim(params,mats)
        print('Finished ito sim')
        plot_sim(params,args.save_plots)
    if args.plot:
        print('Plotting {}'.format(args.plot))
        path = os.path.join(basedir,args.plot+'_sim')
        os.chdir(path)
        plot_sim(params,args.save_plots)

if __name__ == '__main__':
    main()
