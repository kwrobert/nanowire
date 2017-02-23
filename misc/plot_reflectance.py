import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse as ap
import os
import scipy.constants as c

def sort_vec(parv,vec):
    tups = zip(parv,vec)
    sort = sorted(tups)
    return zip(*sort)

def get_data(node,weighted):
    if weighted:
        globstr = os.path.join(node,'**/weighted_transmission_data.dat')
    else:
        globstr = os.path.join(node,'**/ref_trans_abs.dat')
    files = glob.glob(globstr,recursive=True)
    
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

    params,abs_vec = sort_vec(params,abs_vec)
    params,ref_vec = sort_vec(params,ref_vec)
    params,trans_vec = sort_vec(params,trans_vec)
    params,abs_vec,trans_vec,ref_vec = np.array(params),np.array(abs_vec),np.array(trans_vec),np.array(ref_vec)
    params = (c.c*1e9)/params
    params = params[::-1]
    abs_vec = abs_vec[::-1]
    trans_vec = trans_vec[::-1]
    ref_vec = ref_vec[::-1]
    return params,abs_vec,ref_vec,trans_vec,'Wavelength'

def main():

    parser = ap.ArgumentParser(description="Plots weighted reflectance for all sims below a node")
    parser.add_argument('nodex',type=str,help="The xpol node below which data will be collected from")
    parser.add_argument('nodey',type=str,help="The ypol node below which data will be collected from")
    parser.add_argument('--weighted',action='store_true',default=False,help="""Plot weighted or
    unweighted data""")
    parser.add_argument('--ref',action='store_true',help="Plot reflectance")
    parser.add_argument('--trans',action='store_true',help="Plot transmission")
    parser.add_argument('--abs',action='store_true',help="Plot absorbance")
    parser.add_argument('--avg',action='store_true',help="Average data from the two nodes")
    parser.add_argument('--save',help="Path to save plots to")
    args = parser.parse_args()

    if not os.path.isdir(args.nodex) or not os.path.isdir(args.nodey):
        print('Node doesnt exist')
        quit()
   
    title = os.path.basename(args.nodex)
    px,ax,rx,tx,pname = get_data(args.nodex,args.weighted)
    py,ay,ry,ty,pname = get_data(args.nodey,args.weighted)
    #print('Params: %s'%str(px))
    #print('Reflectance: %s'%str(rx))
    #print('Transmittance: %s'%str(ax))
    #print('Absorbance: %s'%str(tx))
    plt.figure()
    if args.ref:
        plt.plot(px,rx,label="X Pol Reflectance")
        plt.plot(py,ry,label="Y Pol Reflectance")
        minref = min(rx)
        ind = rx.argmin(minref)
        print('Minumum xpol reflectance %f at %f'%(float(minref),px[ind]))
        if args.avg:
            avg_ref = .5*(rx+ry)
            plt.plot(py,avg_ref,label="Avg Reflectance")
            minavgref = min(avg_ref)
            ind = avg_ref.argmin(minavgref)
            print('Minumum average reflectance %f at %f'%(float(minavgref),px[ind]))
    if args.trans:
        plt.plot(px,tx,label="X Pol Transmittance")
        plt.plot(py,ty,label="Y Pol Transmittance")
        mintrans = min(tx)
        ind = tx.argmin(mintrans)
        print('Minumum xpol transmittance %f at %f'%(float(mintrans),px[ind]))
        if args.avg:
            avg_trans = .5*(tx+ty)
            plt.plot(py,avg_trans,label="Avg Transmittance")
            minavgtrans = min(avg_trans)
            ind = avg_trans.argmin(minavgref)
            print('Minumum average transmittance %f at %f'%(float(minavgtrans),px[ind]))
    if args.abs:
        plt.plot(px,ax,label="X Pol Absorbance")
        plt.plot(py,ay,label="Y Pol Absorbance")
        minabs = min(ax)
        ind = ax.argmin(minabs)
        print('Minumum xpol absorbance %f at %f'%(float(minabs),px[ind]))
        if args.avg:
            avg_abs = .5*(ax+ay)
            plt.plot(py,avg_abs,label="Avg Absorbance")
            minavgabs = min(avg_abs)
            ind = avg_abs.argmin(minavgabs)
            print('Minumum average absorbance %f at %f'%(float(minavgabs),px[ind]))
    plt.xlabel(pname)
    plt.legend(loc='best')
    plt.title(title)
    if args.save:
        outpath = os.path.join(args.save,'weighted_transmission_plot.pdf')
        plt.savefig(outpath)
    plt.show()

if __name__ == '__main__':
    main()
