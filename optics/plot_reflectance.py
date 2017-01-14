import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse as ap
import os

def sort_vec(parv,vec):
    tups = zip(parv,vec)
    sort = sorted(tups)
    return zip(*sort)

def main():

    parser = ap.ArgumentParser(description="Plots weighted reflectance for all sims below a node")
    parser.add_argument('node',type=str,help="The node below which data will be collected from")
    parser.add_argument('--ref',action='store_true',help="Plot reflectance")
    parser.add_argument('--trans',action='store_true',help="Plot transmission")
    parser.add_argument('--abs',action='store_true',help="Plot absorbance")
    args = parser.parse_args()

    if not os.path.isdir(args.node):
        print('Node doesnt exist')
        quit()
    
    globstr = os.path.join(args.node,'**/weighted_transmission_data.dat')
    files = glob.glob(globstr,recursive=True)
    
    ref_vec = []
    abs_vec = []
    trans_vec = []
    params = []
    for f in files:
        pstring = os.path.split(os.path.dirname(f))[1] 
        param = float(pstring.split('_')[-1])
        pname = pstring[0:pstring.rfind('_')]
        params.append(param)
        with open(f,'r') as dfile:
            lines = dfile.readlines()
            ref,trans,absorb = lines[1].strip().split(',')
            ref_vec.append(ref)
            abs_vec.append(absorb)
            trans_vec.append(trans)

    params,abs_vec = sort_vec(params,abs_vec)
    params,ref_vec = sort_vec(params,ref_vec)
    params,trans_vec = sort_vec(params,trans_vec)
    params = [p*2000 for p in params]
    plt.figure()
    if args.ref:
        plt.plot(params,ref_vec,label="Reflectance")
        minref = min(ref_vec)
        ind = ref_vec.index(minref)
        print(type(params[ind]))
        print('Minumum reflectance %f at %f'%(float(minref),params[ind]))
    if args.trans:
        plt.plot(params,trans_vec,label="Transmission")
    if args.abs:
        plt.plot(params,abs_vec,label="Absorbance")
    plt.xlabel(pname)
    plt.show()

if __name__ == '__main__':
    main()
