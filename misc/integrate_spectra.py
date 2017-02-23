import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import scipy.constants as c
import scipy.integrate as intg
import os

def integrate_wvlgth(path,plot,outf):

    wvlgths, int_w = np.loadtxt(path,delimiter=',',unpack=True,usecols=(0,2),skiprows=2)
    # Filter out wavelengths that have less energy than band gap of GaAs
    band_gap = 1.424*c.e  # Joules
    bound = (c.h*c.c*1e9)/band_gap 
    print(bound)
    #bound = 900
    func = lambda x: x < bound
    print(len(wvlgths))
    wvlgths = np.array(list(filter(func,wvlgths)))
    print(wvlgths.shape)
    int_w = int_w[0:len(wvlgths)]
    wvlgths_m = wvlgths*1e-9
    fact = c.e/(c.h*c.c*10)
    integrand = wvlgths*int_w
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(wvlgths,int_w)
    ax1.set_title("Spectrum")
    ax2.plot(wvlgths,integrand)
    ax2.set_title("Integrand")
    plt.show()
    J = fact*intg.trapz(integrand,x=wvlgths_m)
    print("J = %f"%J)

def main():

    parser = ap.ArgumentParser(description="""A script to convert solar spectra in terms of
    wavelength into a spectra in terms of frequency, and vice versa""")
    parser.add_argument('spectra',type=str,help="""Path to the file containing spectral information.
    The first column should be the spectral parameter and the second column should be the
    intensity, separated by commas""")
    parser.add_argument('-u','--unit',choices=('frequency','wavelength'),help="""The unit the
    provided spectral data is in. Will convert from this unit to the other""")
    parser.add_argument('-p','--plot',action='store_true',help="""Plot the original and converted
    spectra side by side""")
    parser.add_argument('-o','--out',type=str,default='converted_spectra.csv',help="""Output path for
    the converted spectrum file""")
    args = parser.parse_args()

    if not os.path.isfile(args.spectra):
        raise ValueError('Spectral file does not exist')

    if args.unit == 'frequency':
        integrate_freq(args.spectra,args.plot,args.out)
    else:
        integrate_wvlgth(args.spectra,args.plot,args.out)

if __name__ == '__main__':
    main()
