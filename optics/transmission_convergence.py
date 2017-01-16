import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse as ap
import os

def sort_vec(parv,vec):
    tups = zip(parv,vec)
    sort = sorted(tups)
    return zip(*sort)

def get_data(node):
    globstr = os.path.join(node,'**/ref_trans_abs.dat')
    files = glob.glob(globstr,recursive=True)
    print(files) 
    ref_vec = np.zeros(len(files))
    abs_vec = np.zeros(len(files))
    trans_vec = np.zeros(len(files))
    params = np.zeros(len(files))
    for i in range(len(files)):
        f = files[i]
        pstring = os.path.split(os.path.dirname(f))[1] 
        param = float(pstring.split('_')[-1])
        pname = pstring[0:pstring.rfind('_')]
        params[i] = param
        with open(f,'r') as dfile:
            lines = dfile.readlines()
            ref,trans,absorb = lines[1].strip().split(',')
            ref_vec[i] = ref
            abs_vec[i] = absorb
            trans_vec[i] = trans
    print(params)
    print(abs_vec)
    params,abs_vec = sort_vec(params,abs_vec)
    params,ref_vec = sort_vec(params,ref_vec)
    params,trans_vec = sort_vec(params,trans_vec)
    params,abs_vec,trans_vec,ref_vec = np.array(params),np.array(abs_vec),np.array(trans_vec),np.array(ref_vec)
    return params,abs_vec,ref_vec,trans_vec,pname

def main():

    parser = ap.ArgumentParser(description="Plots weighted reflectance for all sims below a node")
    parser.add_argument('node',type=str,help="The xpol node below which data will be collected from")
    parser.add_argument('--ref',action='store_true',help="Plot reflectance")
    parser.add_argument('--trans',action='store_true',help="Plot transmission")
    parser.add_argument('--abs',action='store_true',help="Plot absorbance")
    args = parser.parse_args()

    if not os.path.isdir(args.node):
        print('Node doesnt exist')
        quit()
   
    title = os.path.basename(args.node)
    px,ax,rx,tx,pname = get_data(args.node)
    #print('Params: %s'%str(px))
    #print('Reflectance: %s'%str(rx))
    #print('Transmittance: %s'%str(ax))
    #print('Absorbance: %s'%str(tx))
    plt.figure()
    if args.ref:
        plt.plot(px,rx,label="Reflectance")
        minref = min(rx)
        ind = rx.argmin(minref)
        print('Minumumreflectance %f at %f'%(float(minref),px[ind]))
    if args.trans:
        plt.plot(px,tx,label="Transmittance")
        mintrans = min(tx)
        ind = tx.argmin(mintrans)
        print('Minumum xpol transmittance %f at %f'%(float(mintrans),px[ind]))
    if args.abs:
        plt.plot(px,ax,label="Absorbance")
        minabs = min(ax)
        ind = ax.argmin(minabs)
        print('Minumum absorbance %f at %f'%(float(minabs),px[ind]))
    plt.xlabel(pname)
    plt.legend(loc='best')
    plt.title(title)
    outpath = os.path.join(args.node,'transmission_convergence.pdf')
    plt.savefig(outpath)
    plt.show()

if __name__ == '__main__':
    main()
