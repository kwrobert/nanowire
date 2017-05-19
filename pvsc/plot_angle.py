
import sys
import os
import numpy
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob
import argparse as ap
import numpy as np

def get_value(path):
    with open(path,'r') as f:
        val = float(f.readline())
    return val

def plot_planar_jsc(path):
    jsc_files = glob.glob(os.path.join(path,'**/fractional_absorbtion.dat'),recursive=True)
    jsc_files = sorted(jsc_files)
    jsc_vals = [get_value(path) for path in jsc_files]
    angles = np.arange(0,90,5)
    print(len(jsc_vals))
    print(len(angles))
    plt.figure()
    plt.title('Planar Solar Cell Fractional Absorption')
    plt.xlabel('Angle (degrees)')
    # plt.ylabel('Jsc [mA/cm^2]')
    plt.ylabel('Fractional Absorption')
    plt.plot(angles, jsc_vals,'-o')
    plt.savefig('planar_frac_absorb.pdf')
    # plt.show()
    plt.close()
    return angles, jsc_vals

def plot_nanowire_jsc(path):
    jsc_files = glob.glob(os.path.join(path,'**/fractional_absorbtion.dat'),recursive=True)
    jsc_files = sorted(jsc_files)
    jsc_vals = [get_value(path) for path in jsc_files]
    angles = np.arange(0,90,5)
    print(len(jsc_vals))
    print(len(angles))
    plt.figure()
    plt.title('NW Solar Cell Fractional Absorption')
    plt.xlabel('Angle (degrees)')
    # plt.ylabel('Jsc [mA/cm^2]')
    plt.ylabel('Fractional Absorption')
    plt.plot(angles, jsc_vals,'-o')
    plt.savefig('nw_frac_absorb.pdf')
    # plt.show()
    plt.close()
    return angles, jsc_vals

def plot_overlay(angles, planar_vals, nw_vals):
    
    plt.figure()
    plt.title('Planar Cell and Passivated NW Cell Fractional Absorption')
    plt.xlabel('Shell Thickness [nm]')
    # plt.ylabel('Jsc [mA/cm^2]')
    plt.ylabel('Fractional Absorption')
    plt.plot(angles, planar_vals, '-o', label="Planar Cell")
    plt.plot(angles, nw_vals, '-o', label="NW Cell")
    plt.legend(loc='best')
    plt.savefig('overlay_frac_absorb.pdf')
    # plt.show()
    plt.close()
    return

def main():

    parser = ap.ArgumentParser(description="""Makes the angles study plots for
    PVSC/Photonics north""")
    parser.add_argument('--planar_path',type=str,help="""Path to the planar
    study base directory""")
    parser.add_argument('--nw_path',type=str,help="""Path to the NW
    study base directory""")
    args = parser.parse_args()

    plt.style.use('ggplot')

    if not os.path.isdir(args.planar_path) or not os.path.isdir(args.nw_path):
        raise ValueError('One of the paths you specified does not exist')

    if args.planar_path:
        angles, planar = plot_planar_jsc(args.planar_path)

    if args.nw_path:
        angles, nw = plot_nanowire_jsc(args.nw_path)

    if args.planar_path and args.nw_path:
        plot_overlay(angles, planar, nw)

if __name__ == '__main__':
    main()
