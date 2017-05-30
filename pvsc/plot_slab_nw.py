import sys
import os
import numpy
import matplotlib.pyplot as plt
plt.style.use(['ggplot','presentation','add_marker'])
import glob
import argparse as ap
import numpy as np
import re
import ruamel.yaml as yaml
import scipy.constants as c

def extract_float(string):
    # print('Path: %s'%string)
    simdir = os.path.dirname(string)
    sim_conf = os.path.join(simdir,'sim_conf.yml')
    # config = yaml.load(sim_conf, Loader=yaml.Loader) 
    with open(sim_conf, 'r') as cf:
        config = yaml.safe_load(cf) 
    freq = config['Simulation']['params']['frequency']['value']
    # print('Frequency: %f'%freq)
    return freq 

def get_value(path):
    with open(path,'r') as f:
        ref,trans,absorb = list(map(float,f.readlines()[1].split(',')))
    return ref,trans,absorb 

def plot(path, title, save_name, plot_all=False):
    jsc_files = glob.glob(os.path.join(path,'**/ref_trans_abs.dat'),recursive=True)
    jsc_files = sorted(jsc_files,key=extract_float)
    refs = []
    trans = []
    absorbs = []
    for path in jsc_files:
        ref, tran, absorb = get_value(path)
        refs.append(ref)
        trans.append(tran)
        absorbs.append(absorb)
    wavelengths = [(c.c*1e9)/extract_float(path) for path in jsc_files]
    print(len(absorbs))
    print(len(wavelengths))
    absorbs = list(reversed(absorbs))
    refs = list(reversed(refs))
    trans = list(reversed(trans))
    wavelengths = list(reversed(wavelengths))
    plt.figure()
    # plt.title(title)
    plt.xlabel('Wavelength [nm]')
    if plot_all:
        plt.plot(wavelengths, absorbs, label="Absorption")
        plt.plot(wavelengths, refs, label="Reflection")
        plt.plot(wavelengths, trans, label="Transmission")
        plt.legend(loc='best')
    else:
        plt.plot(wavelengths, absorbs)
        plt.ylabel("Absorption")
    plt.ylim([0,1])
    plt.savefig(save_name)
    plt.close()
    return wavelengths, absorbs, refs, trans

def plot_overlay(angles, y_vals):
    
    plt.figure()
    # plt.title('Planar Cell and NW Cell Absorption')
    plt.xlabel('Shell Thickness [nm]')
    # plt.ylabel('Jsc [mA/cm^2]')
    plt.ylabel('Absorption')
    for vals, label in y_vals:
        plt.plot(angles, vals, '-o', label=label)
    plt.ylim([0,1])
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
    parser.add_argument('--planar_ito',type=str,help="""Path to the planar
    study with ITO base directory""")
    parser.add_argument('--nw_path',type=str,help="""Path to the NW
    study base directory""")
    args = parser.parse_args()

    if args.planar_path and args.nw_path:
        if not os.path.isdir(args.planar_path) or not os.path.isdir(args.nw_path):
            raise ValueError('One of the paths you specified does not exist')

    if args.planar_path:
        angles, planar_abs, planar_trans, planar_refs = plot(args.planar_path,
                              "Planar Solar Cell Absorption",
                              "planar_frac_absorb.pdf", plot_all=True)

    if args.planar_ito:
        angles, ito_abs, ito_refs, ito_trans = plot(args.planar_ito,
                              "Planar Solar Cell w/ ITO Absorption",
                              "planar_ito_frac_absorb.pdf")

    if args.nw_path:
        angles, nw_abs, nw_refs, nw_trans = plot(args.nw_path,
                          "NW Solar Cell w/ ITO Absorption",
                          "nw_frac_absorb.pdf", plot_all=True)

    if args.planar_path and args.nw_path:
        plot_overlay(angles, ((planar_abs, "Planar"),
                              # (planar_ito, "Planar w/ ITO"),
                              (nw_abs, "NW")))



if __name__ == '__main__':
    main()
