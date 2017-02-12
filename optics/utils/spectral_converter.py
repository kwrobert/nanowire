import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import scipy.constants as c
import os

def freq_to_wvlgth(path,plot,outf):
    """Converts a frequency spectrum to a wavelength spectrum"""
    freqs,int_f = np.loadtxt(path,delimiter=',',unpack=True,usecols=(0,2))
    wvlgths = (c.c/freqs)*1e9
    int_w = int_f*c.c/wvlgths**2
    wvlgths = wvlgths[::-1]
    int_w = int_w[::-1]
    if plot:
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(freqs,int_f)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Intensity (W m^-2 Hz^-1)')
        ax1.set_title('Original Spectrum')
        ax2.plot(wvlgths,int_w)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Intensity (W m^-2 nm^-1)')
        ax2.set_title('Converted Spectrum')
        plt.show()
        plt.close(fig)
    out_arr = np.column_stack((wvlgths,int_w))
    np.savetxt(outf,out_arr,header="Wavelength (nm), Power (W m^-2 nm^-1)",delimiter=',')
    return wvlgths, int_w

def wvlgth_to_freq(path,plot,outf):
    """Converts a frequency spectrum to a wavelength spectrum"""
    wvlgths,int_w = np.loadtxt(path,delimiter=',',unpack=True,usecols=(0,2))
    freqs = c.c/(wvlgths*1e-9)
    int_f = int_w*c.c/freqs**2
    freqs = freqs[::-1]
    int_f = int_f[::-1]
    if plot:
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax2.plot(freqs,int_f)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Intensity (W m^-2 Hz^-1)')
        ax2.set_title('Converted Spectrum')
        ax1.plot(wvlgths,int_w)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Intensity (W m^-2 nm^-1)')
        ax1.set_title('Original Spectrum')
        plt.show()
        plt.close(fig)
    out_arr = np.column_stack((freqs,int_f))
    np.savetxt(outf,out_arr,header="Frequency (Hz), Power (W m^-2 Hz^-1)",delimiter=',')
    return freqs, int_f

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
        freq_to_wvlgth(args.spectra,args.plot,args.out)
    else:
        wvlgth_to_freq(args.spectra,args.plot,args.out)

if __name__ == '__main__':
    main()
