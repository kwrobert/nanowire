import os
import numpy as np
import scipy.interpolate as spi
import scipy.constants as constants
import S4
import hashlib

from collections import OrderedDict

def make_hash(o):
    """Makes a hash from the dict representing the Simulation config. It is
    consistent across runs and handles arbitrary nesting. Right now it ignores
    any settings in the General section because those aren't really important
    when it comes to differentiating simulations"""
    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        buf = repr(o).encode('utf-8')
        return hashlib.md5(buf).hexdigest()
    new_o = OrderedDict()
    for k, v in sorted(o.items(),key=lambda tup: tup[0]):
        if k == 'General':
            continue
        else:
            new_o[k] = make_hash(v)
    out = repr(tuple(frozenset(sorted(new_o.items())))).encode('utf-8')
    return hashlib.md5(out).hexdigest()

class Simulator():

    def __init__(self,conf):
        conf.interpolate()
        conf.evaluate()
        self.conf = conf
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        period = self.conf['Simulation']['params']['array_period']['value']
        self.id = make_hash(conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'],self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir 
        self.dir = sim_dir 
        self.s4 = S4.New(Lattice=((period,0),(0,period)),NumBasis=numbasis)

    def configure(self):
        """Configure options for the RCWA solver"""
        self.s4.SetOptions(**self.conf['Solver'])

    def _get_epsilon(self,path):
        """Returns complex dielectric constant for a material by pulling in nk text file, interpolating,
        computing nk values at freq, and converting"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        # Get data
        freq_vec,n_vec,k_vec = np.loadtxt(path,skiprows=1,unpack=True)
        # Get n and k at specified frequency via interpolation 
        f_n = spi.interp1d(freq_vec,n_vec,kind='nearest',
                           bounds_error=False,fill_value='extrapolate')
        f_k = spi.interp1d(freq_vec,k_vec,kind='nearest',
                           bounds_error=False,fill_value='extrapolate')
        n,k = f_n(freq),f_k(freq)
        # Convert to dielectric constant
        # NOTE!!: This assumes the relative magnetic permability (mew) is 1
        epsilon_real = n**2 - k**2
        epsilon_imag = 2*n*k
        epsilon = complex(epsilon_real,epsilon_imag)
        return epsilon
      
    def _get_incident_amplitude(self):
        """Returns the incident amplitude of a wave depending on frequency""" 
        freq = self.conf['Simulation']['params']['frequency']['value']
        polar_angle = self.conf['Simulation']['params']['polar_angle']['value']
        path = self.conf['Simulation']['input_power']
        # Get data
        freq_vec,p_vec = np.loadtxt(path,skiprows=1,unpack=True)
        # Get p at freq by interpolation 
        f_p = spi.interp1d(freq_vec,p_vec,kind='nearest',
                           bounds_error=False,fill_value='extrapolate')
        # We need to reduce total incident power depending on incident polar
        # angle
        E = np.sqrt(constants.c*constants.mu_0*f_p(freq))*np.cos(polar_angle)
        return E

    def set_excitation(self):
        """Sets the exciting plane wave for the simulation"""
        f_phys = self.conf['Simulation']['params']['frequency']['value']
        print('Physical Frequency = %E'%f_phys)
        c_conv = constants.c/self.conf['Simulation']['base_unit']
        f_conv = f_phys/c_conv  
        self.s4.SetFrequency(f_conv)
        E_mag = self._get_incident_amplitude()
        polar = self.conf['Simulation']['params']['polar_angle']['value']
        azimuth = self.conf['Simulation']['params']['azimuthal_angle']['value']
        # To define circularly polarized light, basically just stick a j (imaginary number) in front of
        # one of your components. The handedness is determined by the component you stick the j in front
        # of. From POV of source, looking away from source toward direction of propagation, right handed
        # circular polarization has j in front of y component. Magnitudes are the same. This means
        # E-field vector rotates clockwise when observed from POV of source. Left handed =
        # counterclockwise. 
        # In S4, if indicent angles are 0, p-polarization is along x-axis. The minus sign on front of the 
        # x magnitude is just to get things to look like Anna's simulations.
        polarization = self.conf['Simulation']['polarization']
        if polarization == 'rhcp':
            # Right hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(E_mag,0),
                                           pAmplitude=complex(0,E_mag))
        elif polarization == 'lhcp':
            # Left hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(0,E_mag),
                                           pAmplitude=complex(E_mag,0))
        elif polarization == 'lpx':
            # Linearly polarized along x axis (TM polarixation)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(0,0),
                                           pAmplitude=complex(E_mag,0))
        elif polarization == 'lpy':
            # Linearly polarized along y axis (TE polarization)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar,azimuth),
                                           sAmplitude=complex(E_mag,0),
                                           pAmplitude=complex(0,0))
        else:
            raise ValueError('Invalid polarization specification')

    def build_device(self):
        """Build the device geometry"""
        
        # First define all the materials
        for mat,mat_path in self.conf['Materials'].items():
            eps = self._get_epsilon(mat_path)
            self.s4.SetMaterial(Name=mat,Epsilon=eps)
        self.s4.SetMaterial(Name='vacuum',Epsilon=complex(1,0))
        # We need to properly sort our layers because order DOES matter. Light
        # will be incident upon the first layer specified
        for layer,ldata in sorted(self.conf['Layers'].items(),key=lambda tup: tup[1]['order']):
            print('Building layer:',layer)
            print(layer)
            print(ldata['order'])
            base_mat = ldata['base_material']
            layer_t = ldata['params']['thickness']['value']
            self.s4.AddLayer(Name=layer,Thickness=layer_t,Material=base_mat)
            if 'geometry' in ldata:
                for shape,sdata in sorted(ldata['geometry'].items(),key=lambda tup: tup[1]['order']): 
                    print(shape)
                    print(sdata['order'])
                    print(sdata['type'])
                    shape_mat = sdata['material']
                    if sdata['type'] == 'circle':
                        rad = sdata['radius']
                        cent = sdata['center']
                        coord = (cent['x'],cent['y'])
                        self.s4.SetRegionCircle(Layer=layer,Material=shape_mat,Center=coord,
                                                Radius=rad)
                    else:
                        raise NotImplementedError('Shape %s is not yet implemented'%sdata['type'])

    def get_height(self):
        """Get the total height of the device"""
        height = 0
        for layer,ldata in self.conf['Layers'].items():
            layer_t = ldata['params']['thickness']['value']
            height += layer_t
        return height

    def set_lattice(self,period):
        """Updates the S4 simulation object with a new array period"""
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        self.s4 = S4.New(Lattice=((period,0),(0,period)),NumBasis=numbasis)

    def set_basis(self,numbasis):
        """Updates the S4 simulation object with a new set of basis terms"""
        period = self.conf['Simulation']['params']['array_period']['value']
        self.s4 = S4.New(Lattice=((period,0),(0,period)),NumBasis=numbasis)

    def get_fields(self):
        # Gets the fields throughout the device
        output_file = os.path.join(self.dir,self.conf["General"]["base_name"])
        x_samp = self.conf['Simulation']['x_samples']
        y_samp = self.conf['Simulation']['y_samples']
        z_samp = self.conf['Simulation']['z_samples']
        height = self.get_height()
        dz = height/z_samp
        zvec = np.arange(0,height+dz,dz)
        if self.conf['General']['adaptive_convergence']:
            self.adaptive_convergence(x_samp,y_samp,zvec,output_file)
        else:
            #numbasis = self.conf:get({'Simulation','params','numbasis','value'})
            #self.set_basis(numbasis)
            for z in zvec: 
                self.s4.GetFieldsOnGrid(z=z,NumSamples=(x_samp,y_samp),
                                        Format='FileAppend',BaseFilename=output_file)

    def get_fluxes(self):
        # Gets the incident, reflected, and transmitted powers
        # Note: these all return real and imag components of forward and backward waves
        # as follows: forward_real,backward_real,forward_imaginary,backward_imaginary
        print('Computing fluxes ...')
        path = os.path.join(self.dir,'fluxes.dat')
        with open(path,'w') as out:
            out.write('# Layer,ForwardReal,BackwardReal,ForwardImag,BackwardImag\n')
            for layer,ldata in self.conf['Layers'].items():
                print('Computing fluxes through layer: %s'%layer)
                # This gets flux at top of layer
                forw,back = self.s4.GetPowerFlux(Layer=layer)
                row = '%s,%s,%s,%s,%s\n'%(layer,forw.real,back.real,
                                          forw.imag,back.imag)
                out.write(row)
                # This gets flux at the bottom
                offset = ldata['params']['thickness']['value']
                forw,backward = self.s4.GetPowerFlux(Layer=layer,zOffset=offset)
                row = '%s_bottom,%s,%s,%s,%s\n'%(layer,forw.real,back.real,
                                                 forw.imag,back.imag)
                out.write(row)
