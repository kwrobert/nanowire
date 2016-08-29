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
    with open(path,'r') as config_file:
        parser.readfp(config_file)
    height = sum((parser.getfloat('Parameters','nw_height'),
                  parser.getfloat('Parameters','substrate_t'),
                  parser.getfloat('Parameters','ito_t')))
    parser.set('Parameters','total_height',str(height))
    with open(path,'w') as conf_file:
        parser.write(conf_file)
    return parser
   
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
    print(freq)
    #wvlength = (constants.c/freq)*10E9
    #print(wvlength)
    #lambda_vec,p_vec = np.loadtxt(path,skiprows=2,delimiter=",",usecols=(0,3),unpack=True)
    #print(lambda_vec)
    freq_vec,p_vec = np.loadtxt(path,skiprows=1,unpack=True)
    # Get p at freq by interpolation 
    f_p = interpolate.interp1d(freq_vec,p_vec)
    # NOTE: Include below in some sort of unit test later
    #print("p = %f"%f_p(freq))
    #plt.plot(freq_vec,p_vec,'bs',freq_vec,f_p(freq_vec),'r--')
    #plt.show()
    # Use the Poynting vector to go from power per unit area to E field mag per unit length    i
    E = np.sqrt(constants.c*constants.mu_0*f_p(freq))
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

    # Set up materials
    for mat in conf.options("Materials"):
        nk_path = conf.get("Materials",mat)
        epsilon = get_epsilon(conf.getfloat("Parameters","frequency"),nk_path)
        sim.SetMaterial(Name=mat,Epsilon=epsilon)
    sim.SetMaterial(Name='vacuum',Epsilon=complex(1,0))
    # Add layers. NOTE!!: Order here DOES MATTER, as incident light will be directed at the FIRST
    # LAYER SPECIFIED
    sim.AddLayer(Name='air',Thickness=conf.getfloat('Parameters','air_t'),Material='vacuum')
    sim.AddLayer(Name='ito',Thickness=conf.getfloat('Parameters','ito_t'),Material='ito')
    sim.AddLayer(Name='nanowire',Thickness=conf.getfloat('Parameters','nw_height'),Material='cyclotrene')
    sim.AddLayer(Name='substrate',Thickness=conf.getfloat('Parameters','substrate_t'),Material='gaas')
    
    # Add patterning to layers
    core_rad = conf.getfloat('Parameters','nw_radius')
    alinp_rad = core_rad + conf.getfloat('Parameters','shell_t')
    sim.SetRegionCircle(Layer='nanowire',Material='alinp',Center=(0,0),Radius=alinp_rad)
    sim.SetRegionCircle(Layer='nanowire',Material='gaas',Center=(0,0),Radius=core_rad)
   
    # Set frequency
    f_phys = conf.getfloat("Parameters","frequency")
    c_conv = constants.c/conf.getfloat("General","base_unit")
    f_conv = f_phys/c_conv
    print(f_conv)
    sim.SetFrequency(f_conv)

    # Define incident light. Normally incident with frequency dependent amplitude
    E_mag = get_incident_amplitude(f_phys,vec_mag,conf.get("General","input_power"))
    # To define circularly polarized light, basically just stick a j (imaginary number) in front of
    # one of your components. The handedness is determined by the component you stick the j in front
    # of. From POV of source, looking away from source toward direction of propagation, right handed
    # circular polarization has j in front of y component. Magnitudes are the same. This means
    # E-field vector rotates clockwise when observed from POV of source. Left handed =
    # counterclockwise. 
    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0), pAmplitude=complex(0,E_mag))
    #sim.OutputLayerPatternPostscript(Layer='nanowire',Filename='out.ps')
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

    #postprocess_planes(output_file,vec_mag,50,25)

    #for layer in E_layers:
    #    print("#"*30)
    #    print("Printing E field layer")
    #    print("#"*30)
    #    print()
    #    print(layer)
    #    for row in layer:
    #        print("#"*30)
    #        print("Printing E field layer row")
    #        print("#"*30)
    #        print()
    #        print(row)    
    #    raw_input("Press enter to continue: ")

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
