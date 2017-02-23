import numpy as np
import scipy.integrate as intg
import scipy.constants as c
import argparse as ap
import os

def main():

    parser = ap.ArgumentParser(description="""Convert the fraction weighted absorptions we
    calculated to photocurrent densities""")
    parser.add_argument('jsc_path',help="Path to jsc.dat file for conversion")
    args = parser.parse_args()

    if not os.path.isfile(args.jsc_path):
        print("Jsc file you specified doesn't exist")

    with open(args.jsc_path,'r') as jsf:
        Jsc_frac = float(jsf.readline().strip())
    
    spectra = np.loadtxt('/home/kyle_robertson/schoolwork/gradschool/nanowire/code/ASTMG173.csv',delimiter=',')
    wvlgths_raw = spectra[:,0]
    inds = np.where((wvlgths_raw>=350) & (wvlgths_raw<=900))
    wvlgths = wvlgths_raw[inds]
    power = spectra[inds,2]
    Jsc_actual = (c.e/(c.h*c.c*10))*Jsc_frac*intg.trapz(wvlgths*power,x=wvlgths*1e-9)
    print(Jsc_actual)

if __name__ == '__main__':
    main()
