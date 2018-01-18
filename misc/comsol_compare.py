import sys
import os
import argparse as ap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use(('presentation',))
from nanowire.optics.postprocess import Simulation
from nanowire.optics.utils.config import Config
from matplotlib.colors import LogNorm

def get_middle_slice(sim):

    if 'normE' not in sim.data:
        sim.normE()

    d = {}
    for k in ('Ex', 'Ey', 'Ez', 'normE'):
        if sim.xsamps % 2 == 0:
            middle = int(round(sim.ysamps / 2))
            d[k] = sim.data[k][:, :, middle]
        else:
            # Gotta average
            left = int(np.floor(sim.ysamps / 2))
            right = int(np.ceil(sim.ysamps / 2))
            left = sim.data[k][:, :, left]
            right = sim.data[k][:, :, right]
            d[k] = (left + right)/2
    return d

def get_comsol_data(sim, path):
    raw_data = np.loadtxt(path, comments='%')
    Ex = raw_data[:, 3] + 1j*raw_data[:, 4]
    Ey = raw_data[:, 5] + 1j*raw_data[:, 6]
    Ez = raw_data[:, 7] + 1j*raw_data[:, 8]
    normE = raw_data[:, 9]
    normE_test = np.sqrt(np.absolute(Ex)**2 + np.absolute(Ey)**2 +
                        np.absolute(Ez)**2)
    # normE_test = np.sqrt(Ex*Ex.conj() + Ey*Ey.conj() +
                        # Ez*Ez.conj())
    diff = np.flipud(np.abs(normE - normE_test).reshape(sim.zsamps,
                                                        sim.ysamps+1))
    normE_test = np.flipud(normE_test.reshape((sim.zsamps, sim.ysamps+1)))
    data_dict = {'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'normE': normE}
    for k, v in data_dict.items():
        data_dict[k] = np.flipud(v.reshape((sim.zsamps, sim.ysamps+1))[:, :-1])
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 12))
    #cax1 = ax1.imshow(normE_test)
    #cbar1 = fig.colorbar(cax1, ax=ax1)
    #ax1.set_title('|E|\nfrom Raw Fields', fontsize=14)
    #cax2 = ax2.imshow(data_dict['normE'])
    #cbar2 = fig.colorbar(cax2, ax=ax2)
    #ax2.set_title('|E|\nfrom COMSOL', fontsize=14)
    #cax3 = ax3.imshow(diff)
    #cbar3 = fig.colorbar(cax3, ax=ax3)
    #ax3.set_title('Absolute\nDifference', fontsize=14)
    #plt.tight_layout()
    #plt.savefig('norm_diff.pdf')
    #plt.show()
    data_dict['normE'] = normE_test[:, :-1]
    return data_dict

def strip_air(sim, rcwa, coms):
    air_t = sim.conf[('Layers', 'Air', 'params', 'thickness', 'value')]
    inds = int(round(air_t/sim.dz))
    for k in rcwa.keys():
        rcwa[k] = rcwa[k][inds:, :]
        coms[k] = coms[k][inds:, :]
    return rcwa, coms

def compare(rcwa, coms, title, sim, cb_label='', pname=False):
    diff = np.abs(rcwa - coms)
    fig, axes = plt.subplots(1, 4, figsize=(13, 11))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.set_title('RCWA')
    ax2.set_title('COMSOL')
    ax3.set_title('Absolute\nDifference')
    ax4.set_title('Relative\nDifference (%)')
    height = sim.height - sim.conf[('Layers', 'Air', 'params', 'thickness', 'value')]
    # height = sim.height
    print("RCWA Max: {}".format(np.amax(rcwa)))
    print("COMSOL Max: {}".format(np.amax(coms)))
    norm_max = max(np.amax(rcwa), np.amax(coms))
    norm_min = min(np.amin(rcwa), np.amin(coms))
    x = np.arange(0, sim.period, sim.dx)
    z = np.arange(0, height + sim.dz, sim.dz)
    # RCWA Heatmap
    cax1 = ax1.imshow(rcwa, vmin=norm_min, vmax=norm_max,
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    # cax1 = ax1.imshow(rcwa, norm=LogNorm(vmin=rcwa.min(), vmax=rcwa.max()))
    cbar1 = fig.colorbar(cax1, ax=ax1)
    cbar1.set_label(cb_label, rotation=270)
    # COMSOL Heatmap
    cax2 = ax2.imshow(coms, vmin=norm_min, vmax=norm_max,
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    # cax2 = ax2.imshow(coms, norm=LogNorm(vmin=coms.min(), vmax=coms.max()))
    cbar2 = fig.colorbar(cax2, ax=ax2)
    cbar2.set_label(cb_label, rotation=270)
    # Absolute Difference Heatmap
    cax3 = ax3.imshow(diff, norm=LogNorm(vmin=diff.min(), vmax=diff.max()),
    # cax3 = ax3.imshow(diff, 
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    cbar3 = fig.colorbar(cax3, ax=ax3)
    rel_diff = diff / np.abs(coms)
    rel_diff = np.clip(100*rel_diff, None, 10)
    print(np.amax(rel_diff))
    # rel_diff = 100*diff / np.abs(coms)
    # rel_diff = 100*np.clip(diff / np.abs(coms), None, 2000)
    # rel_diff = diff / np.abs(coms)
    # cax4 = ax4.imshow(rel_diff, norm=LogNorm(vmin=rel_diff.min(), vmax=10),
    # cax4 = ax4.imshow(rel_diff, vmin=rel_diff.min(), vmax=10, 
    cax4 = ax4.imshow(rel_diff,
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    # Draw boundaries
    middle = int(round(sim.ysamps / 2))
    for ax in [ax1, ax2, ax3, ax4]:
        # sim.draw_geometry_2d('yz', middle, ax, skip_list=['Air'])
        sim.draw_geometry_2d('yz', middle, ax, skip_list=['Air'])
    fig.colorbar(cax4, ax=ax4)
    fig.suptitle(title, fontsize=24, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, .98))
    if pname:
        plt.savefig(pname)
    print('Maximimum Difference = {}'.format(diff.max()))
    print('Maximimum Relative Difference = {}'.format(rel_diff.max()))
    plt.show()

def plot_diff(rcwa, coms, title, sim, pname=False):
    diff = np.abs(rcwa - coms)
    fig, ax = plt.subplots()
    ax.set_title('Absolute Difference')
    height = sim.get_height() - sim.conf[('Layers', 'Air', 'params', 'thickness', 'value')]
    print(height)
    print(sim.height)
    # height = sim.height
    x = np.arange(0, sim.period + sim.dx, sim.dx)
    z = np.arange(0, height + sim.dz, sim.dz)
    cax = ax.imshow(diff, norm=LogNorm(vmin=1, vmax=diff.max()),
                    extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    # cax1 = ax1.imshow(rcwa, norm=LogNorm(vmin=rcwa.min(), vmax=rcwa.max()))
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('|E|^2')
    middle = int(round(sim.ysamps / 2))
    sim.draw_geometry_2d('yz', middle, ax, skip_list=['Air'])
    plt.savefig(pname)
    plt.show()


def main():
    desc = """
    A script for comparing RCWA field data to COMSOL field data at a single
    frequency. 
    
    To avoid dealing with parsing complex number outputs from COMSOL,
    the CSV columns in the COMSOL output file must be in the following order:
    x,y,z,Exreal,Eximag,Eyreal,Eyimag,Ezreal,Eximag,normE. Comments are
    expected to begin with a % sign

    Also note, if you use N sampling points in either the x or y  direction in
    RCWA, you must use N+1 sampling points in COMSOL. This is because S4
    excludes x,y=period in the field outputs, while COMSOL always includes both
    edges. I haven't found a nice way to make COMSOL exclude an endpoint, and
    its impossible to make S4 include the endpoints with the current API. So,
    using N+1 gridpoints in COMSOL ensures that RCWA and COMSOL have data on
    the same gridpoints. COMSOL will just have extra data at the endpoints
    which we can discard. THIS DOES NOT APPLY TO THE Z DIRECTION. USE THE SAME
    NUMBER OF SAMPLING POINTS IN THE Z DIRECTION.
    """ 
    parser = ap.ArgumentParser(description=desc)
    parser.add_argument('rcwa', help='The RCWA YAML config file')
    parser.add_argument('comsol', help='The COMSOL CSV file') 
    parser.add_argument('--save', '-s', action='store_true', help='Save pdf of plots') 
    args = parser.parse_args()

    sim = Simulation(Config(args.rcwa))
    freq = sim.conf[('Simulation', 'params', 'frequency', 'value')]
    lanczos = sim.conf[('Solver', 'LanczosSmoothing')]
    basis = sim.conf[('Simulation', 'params', 'numbasis', 'value')]
    rcwa_data = get_middle_slice(sim)
    comsol_data = get_comsol_data(sim, args.comsol)
    rcwa_data, comsol_data = strip_air(sim, rcwa_data, comsol_data)
    c = 2.99e8
    wvlgth = 1e9*c/freq
    title = 'Freq = {:.2E} Hz'.format(freq)
    title += ', Wavelength = {} nm'.format(wvlgth)
    if lanczos:
        pname = 'f{:.2E}_withlanczos_n{}.pdf'.format(freq, basis)
        pname2 = 'f{:.2E}_withlanczos_n{}_absdiff.pdf'.format(freq, basis)
        title += ', w/ Lanczos, Basis = {}'.format(basis)
    else:
        pname = 'f{:.2E}_nolanczos_n{}.pdf'.format(freq, basis)
        pname2 = 'f{:.2E}_nolanczos_n{}_absdiff.pdf'.format(freq, basis)
        title += ', w/o Lanczos, Basis = {}'.format(basis)
    # compare(rcwa_data['normE']**2, comsol_data['normE']**2, title, sim, pname=pname)
    if args.save:
        pnamex = 'f{:.2E}_n{}_ex.pdf'.format(freq, basis)
        pnamey = 'f{:.2E}_n{}_ey.pdf'.format(freq, basis)
        pnamez = 'f{:.2E}_n{}_ez.pdf'.format(freq, basis)
        pnamenorm = 'f{:.2E}_n{}_enorm.pdf'.format(freq, basis)
    else:
        pnamex = False
        pnamey = False
        pnamez = False
        pnamenorm = False
    compare(rcwa_data['normE']**2, comsol_data['normE']**2, title, sim,
            cb_label='|E|^2', pname=pnamenorm)
    compare(np.absolute(rcwa_data['Ey']), np.absolute(comsol_data['Ex']),
            title, sim, cb_label='|Ex|', pname=pnamex)
    compare(np.absolute(rcwa_data['Ex']), np.absolute(comsol_data['Ey']),
            title, sim, cb_label='|Ey|', pname=pnamey)
    compare(np.absolute(rcwa_data['Ez']), np.absolute(comsol_data['Ez']),
            title, sim, cb_label='|Ez|', pname=pnamez)
    # plot_diff(rcwa_normE, comsol_normE, title, sim, pname=pname2)

if __name__ == '__main__':
    main()
