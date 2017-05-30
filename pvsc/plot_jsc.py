import sys
import os
import numpy
import matplotlib.pyplot as plt
plt.style.use(['ggplot','presentation','add_marker'])
import glob
import argparse as ap

def get_value(path):
    with open(path,'r') as f:
        val = float(f.readline())
    return val


def plot_optimized_jsc(path):
    jsc_files = glob.glob(os.path.join(path,'**/fractional_absorbtion.dat'),recursive=True)
    jsc_files = sorted(jsc_files)
    print(jsc_files)
    jsc_vals = [get_value(path) for path in jsc_files]
    shell_ts = [10,20,30,40,50,60,70,80]
    print(len(jsc_vals))
    print(len(shell_ts))
    plt.figure()
    # plt.title('Reoptimized Passivated Nanowires')
    plt.xlabel('Shell Thickness [nm]')
    # plt.ylabel('Jsc [mA/cm^2]')
    plt.ylabel(r'$\bar A$')
    plt.plot(shell_ts, jsc_vals)
    plt.savefig('optimized_jsc.pdf')
    # plt.show()
    return shell_ts, jsc_vals

def plot_sweep_jsc(path):
    # jsc_files = glob.glob(os.path.join(path,'**/jsc.dat'),recursive=True)
    jsc_files = glob.glob(os.path.join(path,'**/fractional_absorbtion.dat'),recursive=True)
    jsc_files = sorted(jsc_files)
    print(jsc_files)
    jsc_vals = [get_value(path) for path in jsc_files]
    jsc_vals = list(reversed(jsc_vals))
    shell_ts = [10,20,30,40,50,60,70,80]
    print(len(jsc_vals))
    print(len(shell_ts))
    plt.figure()
    # plt.title('Passivated Nanowires')
    plt.xlabel('Shell Thickness [nm]')
    # plt.ylabel('Jsc [mA/cm^2]')
    plt.ylabel(r'$\bar A$')
    plt.plot(shell_ts, jsc_vals)
    plt.savefig('sweep_jsc.pdf')
    # plt.show()
    return shell_ts, jsc_vals

def plot_overlay(opt_ts, opt_jsc, sweep_ts, sweep_jsc):
    
    plt.figure()
    # plt.title('Optimized and Unoptimized Passivated Nanowires')
    plt.xlabel('Shell Thickness [nm]')
    # plt.ylabel('Jsc [mA/cm^2]')
    plt.ylabel(r'$\bar A$')
    plt.plot(sweep_ts, sweep_jsc, label="Unoptimized")
    plt.plot(opt_ts, opt_jsc, label="Optimized")
    plt.legend(loc='best')
    plt.savefig('overlay_jsc.pdf')
    # plt.show()

def main():

    parser = ap.ArgumentParser(description="""Makes the Jsc plots for
    PVSC/Photonics north""")
    parser.add_argument('--opt_path',type=str,help="""Path to the shell
    reoptimization base directory""")
    parser.add_argument('--sweep_path',type=str,help="""Path to the
    sweep base directory""")
    args = parser.parse_args()

    if not os.path.isdir(args.opt_path) or not os.path.isdir(args.sweep_path):
        raise ValueError('One of the paths you specified does not exist')

    if args.opt_path:
        opt_ts, opt_jsc = plot_optimized_jsc(args.opt_path)

    if args.sweep_path:
        sweep_ts, sweep_jsc = plot_sweep_jsc(args.sweep_path)

    if args.opt_path and args.sweep_path:
        plot_overlay(opt_ts, opt_jsc, sweep_ts, sweep_jsc)

if __name__ == '__main__':
    main()
