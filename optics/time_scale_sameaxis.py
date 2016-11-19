import glob
import re
import os
import datetime
import argparse as ap
import matplotlib
# Enables saving plots over ssh
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
import numpy as np

def main():
    parser = ap.ArgumentParser(description="""Plots the minimum required basis terms to achieve a
    desired accuracy against an independent parameter""")
    parser.add_argument('path',type=str,help="Base directory path for the sorted sweep")
    parser.add_argument('-s','--show',action='store_true',default=False,help="""Show plots
            interactively""")
    args = parser.parse_args()
    
    path = args.path
    if not os.path.isdir(path):
        print("Specified path does not exist")
        quit()
   
    fields = ['default','jones','normal','no_vec']
    #radii = ['nw_radius_0.075','nw_radius_0.06']
    radii = ['nw_radius_0.06']
    #periods = ['array_period_0.25','array_period_0.3']
    periods = ['array_period_0.25']
    systems = {'array_period_0.25':['nw_radius_0.075','nw_radius_0.06'],
               'array_period_0.3':['nw_radius_0.075']}

    for period,radii in systems.items():
        for radius in radii:
            fig, axes = plt.subplots(2,1)
            for field in fields:
                

                tglob = os.path.join(path,field,period+"*",radius,'frequency_3.2962*/time_scaling.dat')
                tlist = glob.glob(tglob)
                if tlist:
                    tpath = tlist[0]
                else:
                    print(tglob)
                    quit()
                print(tpath)
                numbasis, tdata = np.loadtxt(tpath,unpack=True,skiprows=1,delimiter=',')
                axes[0].plot(numbasis,tdata,'-o',label=field)
                axes[0].legend(loc='best')
                axes[0].set_xlabel('Number of Basis Terms')
                axes[0].set_ylabel('Wall Clock Time (seconds)')
                axes[0].set_title('%s, %s'%(period,radius))
                mglob = os.path.join(path,field,period+"*",radius,'minimum_basis_terms_global_t0.05.dat')
                mlist = glob.glob(mglob)
                if mlist:
                    mpath = mlist[0]
                else:
                    print(mglob)
                    quit()
                print(mpath)
                freq, mdata = np.loadtxt(mpath,unpack=True,skiprows=1,delimiter=',')
                axes[1].plot(freq,mdata,'-o',label=field)
                #axes[1].legend(loc='best')
                axes[1].set_ylabel('Number of Basis Terms')
                axes[1].set_xlabel('Frequency (Hz)')
                axes[1].set_title('%s, %s, Threshold = .05'%(period,radius))
            fig.subplots_adjust(hspace=.35)
            plt.savefig(os.path.join(path,'%s_%s_vec_field_timescaling_comparison.pdf'%(period,radius)))
            #plt.show()

    #fig, axes = plt.subplots(3,2)
    #counter = 0
    #for period,radii in systems.items():
    #    for radius in radii:
    #        for field in fields:
    #            
    #            fig, axes = plt.subplots(3,2)
    #            tglob = os.path.join(path,field,period+"*",radius,'frequency_3.2962*/time_scaling.dat')
    #            tlist = glob.glob(tglob)
    #            if tlist:
    #                tpath = tlist[0]
    #            else:
    #                print(tglob)
    #                quit()
    #            print(tpath)
    #            numbasis, tdata = np.loadtxt(tpath,unpack=True,skiprows=1,delimiter=',')
    #            axes[counter][0].plot(numbasis,tdata,'-o',label=field)
    #            axes[counter][0].legend(loc='best')
    #            axes[counter][0].set_xlabel('Number of Basis Terms')
    #            axes[counter][0].set_ylabel('Wall Clock Time (seconds)')
    #            axes[counter][0].set_title('%s, %s'%(period,radius))
    #        for field in fields:
    #            mglob = os.path.join(path,field,period+"*",radius,'minimum_basis_terms_global_t0.05.dat')
    #            mlist = glob.glob(mglob)
    #            if mlist:
    #                mpath = mlist[0]
    #            else:
    #                print(mglob)
    #                quit()
    #            print(mpath)
    #            freq, mdata = np.loadtxt(mpath,unpack=True,skiprows=1,delimiter=',')
    #            axes[counter][1].plot(freq,mdata,'-o',label=field)
    #            axes[counter][1].legend(loc='best')
    #            axes[counter][1].set_ylabel('Number of Basis Terms')
    #            axes[counter][1].set_xlabel('Frequency (Hz)')
    #            axes[counter][1].set_title('%s, %s, Threshold = .05'%(period,radius))
    #        counter += 1
    #fig.subplots_adjust(hspace=.25)
    #plt.savefig(os.path.join(path,'vec_field_timescaling_comparison.pdf'))
    #plt.show()

    #time_glob = os.path.join(path,'**/frequency_3.2962*/**/time_scaling.dat')
    #print(time_glob)
    #time_files = glob.glob(time_glob,recursive=True)
    #print(time_files)
    #quit()
    #for time_file in time_files:
    #    basis_terms, data = np.loadtxt(time_file,unpack=True)
    #    if 'normal' in time_file:
    #        normal = data
    #    elif 'no_vec' in time_file:
    #        no_vec = data
    #    elif 'default' in time_file:
    #        default = data
    #    elif 'jones' in time_file:
    #        jones = data
    #    else:
    #        print('Didnt find valid vec field config in path')
    #quit()
    #plt.figure()
    #excludes = [os.path.join(path,'comp_struct')]
    #print('Beginning global time scaling analysis for %s'%path)
    #for root,dirs,files in os.walk(path,topdown=False):
    #    # If we haven't already analyzed this node in the directory tree
    #    if os.path.split(root)[0] not in excludes:
    #        # Find all error files in current directory
    #        if 'timing.dat' in files:
    #            # If we found an time file, go up one directory and perform the analysis
    #            base = os.path.split(root)[0]
    #            print('Computing time scaling for subdir %s'%base)
    #            analyze(base,args.show)
    #            # Add the level above the basis term sweep to the excludes list so we don't perform
    #            # the analysis for every basis term dir we find an error file in
    #            excludes.append(base)

if __name__ == '__main__':
    main()
