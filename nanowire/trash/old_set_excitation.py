"""
This is the old set excitation function. I've decided to just always set the
incident power to 1 when running simulations, and then rescale the fields using
incident spectrum after the fact. This makes it much easier to swap different
spectra in and out at will
"""

def set_excitation(self):
    """Sets the exciting plane wave for the simulation"""
    f_phys = self.conf['Simulation']['frequency']
    self.log.debug('Physical Frequency = %E' % f_phys)
    c_conv = constants.c / self.conf['Simulation']['base_unit']
    f_conv = f_phys / c_conv
    self.s4.SetFrequency(f_conv)
    E_mag = get_incident_amplitude(self)
    # E_mag = self._get_incident_amplitude_anna()
    polar = self.conf['Simulation']['polar_angle']
    azimuth = self.conf['Simulation']['azimuthal_angle']
    # To define circularly polarized light from the point of view of the
    # source, basically just stick a j
    # (imaginary number) in front of one of your components. The component
    # you choose to stick the j in front of is a matter of convention. In
    # S4, if the incident azimuthal angle is 0, p-polarization is along
    # x-axis. Here, we choose to make the y-component imaginary. The
    # handedness is determined both by the component you stick the j in
    # front of and the sign of the imaginary component. In our convention,
    # minus sign means rhcp, plus sign means lhcp. To be circularly
    # polarized, the magnitudes of the two components must be the same.
    # This means E-field vector rotates clockwise when observed from POV of
    # source. Left handed = counterclockwise.
    # TODO: This might not be properly generalized to handle
    # polarized light if the azimuth angle IS NOT 0. Might need some extra
    # factors of cos/sin of azimuth to gen proper projections onto x/y axes
    polarization = self.conf['Simulation']['polarization']
    if polarization == 'rhcp':
        # Right hand circularly polarized
        self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                       sAmplitude=complex(0,
                                                          -E_mag/np.sqrt(2)),
                                       pAmplitude=complex(E_mag/np.sqrt(2), 0))
    elif polarization == 'lhcp':
        # Left hand circularly polarized
        self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                       sAmplitude=complex(0,
                                                          E_mag/np.sqrt(2)),
                                       pAmplitude=complex(E_mag/np.sqrt(2), 0))
    elif polarization == 'lpx':
        # Linearly polarized along x axis (TM polarixation)
        self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                       sAmplitude=complex(0, 0),
                                       pAmplitude=complex(E_mag, 0))
    elif polarization == 'lpy':
        # Linearly polarized along y axis (TE polarization)
        self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                       sAmplitude=complex(E_mag, 0),
                                       pAmplitude=complex(0, 0))
    else:
        raise ValueError('Invalid polarization specification')
