import numpy as np
import os
#  import matplotlib.pyplot as plt
import scipy.constants as c


def main():
    print('adfbadb')
    base = '/home/kyle_robertson/schoolwork/gradschool/nanowire/code/NK'
    nkfiles = ['004_SiO2_nk_Hz.txt','006_GaAs_nk_Walker_modified_Hz.txt',
               '007_Cyclotrene_nk_Hz.txt','008_ITO_nk_Hz.txt','009_AlInP_nk_Hz.txt']
    for nkfile in nkfiles:
        dpath = os.path.join(base,nkfile)
        data = np.loadtxt(dpath,skiprows=1)
        # Get the wavelength in vacuum in micrometers
        inc_wvlgth = (c.c/data[:,0])*1e6
        # Wavelength in a given material is the vacuum wavelength divided by
        # the index of refraction. This is accurate for the first layer in the
        # device, but as light progresses through each layer the wavelength is
        # decreased at each interface. So, the minimum wavelength in the
        # substrate might be lower than whats calculated below because the
        # incident wavelength is lower than that of the vacuum wavelength.
        # Thus, we multiply by a factor of .1 when computing the maximum mesh
        # size to be safe 
        mat_wvlgth = inc_wvlgth/data[:,1]
        ind = np.argmin(mat_wvlgth)
        print('Material: %s'%nkfile)
        print('Minimum Wavelength: %f [um]'%mat_wvlgth[ind])
        print('Maxmimum Mesh Size: %f [um]'%(mat_wvlgth[ind]*.1))

if __name__ == '__main__':
    main()
