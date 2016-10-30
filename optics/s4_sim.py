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

def parse_file(path):
    """Parse the INI file provided at the command line"""
    
    parser = confp.SafeConfigParser()
    parser.optionxform = str
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    height = sum((parser.getfloat('Parameters','nw_height'),
                  parser.getfloat('Parameters','substrate_t'),
                  parser.getfloat('Parameters','ito_t'),
                  parser.getfloat('Parameters','air_t')))
    parser.set('Parameters','total_height',str(height))
    # If we have any evaluated parameters in the config, evaluate them all and set them to the
    # appropriate value
    for par, val in parser.items("Parameters"):
        if val[0] == '`' and val[-1] == '`':
            print('Found evaluated param')
            print(val)
            result = val.strip('`')
            result = eval(result)
            parser.set('Parameters',par,str(result))
            print(result)
    with open(path,'w') as conf_file:
        parser.write(conf_file)
    return parser
   
def get_epsilon(freq,path):
    """Returns complex dielectric constant for a material by pulling in nk text file, interpolating,
    computing nk values at freq, and converting"""
    # Get data
    freq_vec,n_vec,k_vec = np.loadtxt(path,skiprows=1,unpack=True)
    # Get n and k at specified frequency via interpolation 
    f_n = interpolate.interp1d(freq_vec,n_vec,kind='nearest',
                               bounds_error=False,fill_value='extrapolate')
    f_k = interpolate.interp1d(freq_vec,k_vec,kind='nearest',
                               bounds_error=False,fill_value='extrapolate')
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

def get_incident_amplitude(freq,period,path):
    """Returns the incident amplitude of a wave depending on frequency. NOTE!! The input power in
    the text file is in units of W/m^2*nm. Here we DO NOT multiply by unit cell area to get pure W before
    converting to amplitudes. This is because we would be multiplying by a factor of ~10^-12, which
    is super small and would be detrimental to the numerics. Thus, all electric field values have
    units of Volts/sqrt(period*period*conversion_factor^2)""" 
    # Get data
    freq_vec,p_vec = np.loadtxt(path,skiprows=1,unpack=True)
    # Get p at freq by interpolation 
    f_p = interpolate.interp1d(freq_vec,p_vec,kind='nearest',
                               bounds_error=False,fill_value='extrapolate')
    # NOTE: Include below in some sort of unit test later
    #print("p = %f"%f_p(freq))
    #plt.plot(freq_vec,p_vec,'bs',freq_vec,f_p(freq_vec),'r--')
    #plt.show()
    # Use the Poynting vector to go from power per unit area to E field mag per unit length    i
    ## NOTE!!!: S4 normalizes c and mu_0 to 1, so I think passing in this value for the E field 
    # amplitude is incorrect!!
    E = np.sqrt(constants.c*constants.mu_0*f_p(freq))
    #E = np.sqrt(2*np.sqrt(constants.epsilon_0/constants.mu_0)*f_p(freq))
    #E = np.sqrt(2*f_p(freq))
    return E



def build_sim(conf):
    """Define the materials and build the geometry for the simulation"""
    
    # Initialize simulation object
    num_basis = conf.getint("Parameters","numbasis") 
    # These lattice vectors can be a little confusing. Everything in S4 is normalized so that speed
    # of light, vacuum permittivity and permeability are all normalized to 1. This means frequencies
    # must be specified in units of inverse length. This can be clarified as follows 
    # 1. Define a base length unit (say 1 micron)
    # 2. Define your lattice vectors as fractions of that base unit 
    # 3. Whenever you enter a physical frequency (say in Hz), divide it by the speed of light,
    # where the speed of light has been converted to your base length unit of choice.
    # 4. Supply that value to the SetFrequency method
    vec_mag = conf.getfloat("Parameters","array_period")
    sim = S4.New(Lattice=((vec_mag,0),(0,vec_mag)),NumBasis=num_basis)
    # Collect, clean, and set the simulation config
    opts = dict(conf.items('Simulation'))
    for key,val in opts.items():
        if val.isdigit():
            opts[key] = int(val)
        elif val == 'True':
            opts[key] = True
        elif val == 'False':
            opts[key] = False
    sim.SetOptions(**opts)

    # Set up materials
    for mat in conf.options("Materials"):
        nk_path = conf.get("Materials",mat)
        epsilon = get_epsilon(conf.getfloat("Parameters","frequency"),nk_path)
        sim.SetMaterial(Name=mat,Epsilon=epsilon)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
    # Add layers. NOTE!!: Order here DOES MATTER, as incident light will be directed at the FIRST
    # LAYER SPECIFIED
    sim.AddLayer(Name='air',Thickness=conf.getfloat('Parameters','air_t'),Material='vacuum')
    sim.AddLayer(Name='ito',Thickness=conf.getfloat('Parameters','ito_t'),Material='ITO')
    sim.AddLayer(Name='nanowire_alshell',Thickness=conf.getfloat('Parameters','alinp_height'),Material='Cyclotene')
    # Add patterning to section with AlInP shell
    core_rad = conf.getfloat('Parameters','nw_radius')
    shell_rad = core_rad + conf.getfloat('Parameters','shell_t')
    sim.SetRegionCircle(Layer='nanowire_alshell',Material='AlInP',Center=(vec_mag/2,vec_mag/2),Radius=shell_rad)
    sim.SetRegionCircle(Layer='nanowire_alshell',Material='GaAs',Center=(vec_mag/2,vec_mag/2),Radius=core_rad)
    # Si layer and patterning 
    sim.AddLayer(Name='nanowire_sishell',Thickness=conf.getfloat('Parameters','sio2_height'),Material='Cyclotene')
    # Add patterning to layer with SiO2 shell 
    sim.SetRegionCircle(Layer='nanowire_sishell',Material='SiO2',Center=(vec_mag/2,vec_mag/2),Radius=shell_rad)
    sim.SetRegionCircle(Layer='nanowire_sishell',Material='GaAs',Center=(vec_mag/2,vec_mag/2),Radius=core_rad)
    # Substrate layer and air transmission region
    sim.AddLayer(Name='substrate',Thickness=conf.getfloat('Parameters','substrate_t'),Material='GaAs')
    #sim.AddLayerCopy('air_below',Thickness=conf.getfloat('Parameters','air_t'),Layer='air') 

    # Set frequency
    f_phys = conf.getfloat("Parameters","frequency")
    print('Physical Frequency = %E'%f_phys)
    c_conv = constants.c/conf.getfloat("General","base_unit")
    f_conv = f_phys/c_conv
    print('Converted Frequency = %f'%f_conv)
    sim.SetFrequency(f_conv)

    # Define incident light. Normally incident with frequency dependent amplitude
    E_mag = get_incident_amplitude(f_phys,vec_mag,conf.get("General","input_power"))
    # To define circularly polarized light, basically just stick a j (imaginary number) in front of
    # one of your components. The handedness is determined by the component you stick the j in front
    # of. From POV of source, looking away from source toward direction of propagation, right handed
    # circular polarization has j in front of y component. Magnitudes are the same. This means
    # E-field vector rotates clockwise when observed from POV of source. Left handed =
    # counterclockwise. 
    # In S4, if indicent angles are 0, p-polarization is along x-axis. The minus sign on front of the 
    # x magnitude is just to get things to look like Anna's simulations.
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0),
            pAmplitude=complex(0,-E_mag))
    #sim.OutputLayerPatternPostscript(Layer='ito',Filename='out.ps')
    #sim.OutputStructurePOVRay(Filename='out.pov')
    E_layers = []
    output_file = conf.get("General","base_name")
    existing_files = glob.glob(output_file+".*")
    if existing_files:
        for afile in existing_files:
            os.remove(afile)
    x_samp = conf.getint('General','x_samples')
    y_samp = conf.getint('General','y_samples')
    z_samp = conf.getint('General','z_samples')
    height = conf.getfloat('Parameters','total_height') 
    for z in np.linspace(0,height,z_samp):
        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
                            Format='FileAppend',BaseFilename=output_file)

def main():

    parser = ap.ArgumentParser(description="""A program that uses the S4 RCWA simulation library to
    simulate the optical properties of a single nanowire in a square lattice""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    containing your material, geometry, and other configuration parameters""")

    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf_obj = parse_file(os.path.abspath(args.config_file))
    else:
        print "\n The file you specified does not exist! \n"
        quit()

    build_sim(conf_obj)


if __name__ == '__main__':
    main()
