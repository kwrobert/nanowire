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

    if sim.x_samples % 2 == 0:
        middle = int(round( sim.y_samples / 2 ))
        return sim.data['normE'][:, :, middle]
    else:
        # Gotta average
        left = int(np.floor(sim.y_samples / 2))
        right = int(np.ceil(sim.y_samples / 2))
        normE_left = sim.data['normE'][:, :, left]
        normE_right = sim.data['normE'][:, :, right]
        return (normE_left + normE_right)/2

def get_comsol_normE(sim, path):
    data = np.loadtxt(path, comments='%')
    return np.flipud(data[:, -1].reshape((sim.z_samples, sim.x_samples)))

def strip_air(sim, rcwa, coms):
    air_t = sim.conf[('Layers', 'Air', 'params', 'thickness', 'value')]
    inds = int(round(air_t/sim.dz))
    return rcwa[inds:, :], coms[inds:, :]


def compare(rcwa, coms, title, sim, pname=False):
    diff = np.abs(rcwa - coms)
    fig, axes = plt.subplots(1, 4, figsize=(13, 11))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.set_title('RCWA')
    ax2.set_title('COMSOL')
    ax3.set_title('Absolute\nDifference')
    ax4.set_title('Relative\nDifference (%)')
    height = sim.height - sim.conf[('Layers', 'Air', 'params', 'thickness', 'value')]
    print(height)
    print(sim.height)
    # height = sim.height
    x = np.arange(0, sim.period + sim.dx, sim.dx)
    z = np.arange(0, height + sim.dz, sim.dz)
    cax1 = ax1.imshow(rcwa, vmin=10, vmax=13000,
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    # cax1 = ax1.imshow(rcwa, norm=LogNorm(vmin=rcwa.min(), vmax=rcwa.max()))
    cbar1 = fig.colorbar(cax1, ax=ax1)
    cbar1.set_label('|E|^2', rotation=270)
    cax2 = ax2.imshow(coms, vmin=10, vmax=13000,
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    # cax2 = ax2.imshow(coms, norm=LogNorm(vmin=coms.min(), vmax=coms.max()))
    cbar2 = fig.colorbar(cax2, ax=ax2)
    cbar2.set_label('|E|^2', rotation=270)
    cax3 = ax3.imshow(diff, norm=LogNorm(vmin=1, vmax=diff.max()),
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    cbar3 = fig.colorbar(cax3, ax=ax3)
    rel_diff = 100*diff / np.abs(coms)
    cax4 = ax4.imshow(rel_diff, norm=LogNorm(vmin=rel_diff.min(), vmax=10),
                      extent=[x.min(),x.max(),z.min(),z.max()], aspect='auto')
    # Draw boundaries
    middle = int(round(sim.y_samples / 2))
    for ax in [ax1, ax2, ax3, ax4]:
        # sim.draw_geometry_2d('yz', middle, ax, skip_list=['Air'])
        sim.draw_geometry_2d('yz', middle, ax, skip_list=['Air'])
    fig.colorbar(cax4, ax=ax4)
    fig.suptitle(title, fontsize=24, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, .98))
    if pname:
        plt.savefig(pname)
    plt.show()
    print('Maximimum Difference = {}'.format(diff.max()))

def plot_diff(rcwa, coms, title, sim, pname=False): 
    diff = np.abs(rcwa - coms)
    fig, ax = plt.subplots()
    ax.set_title('Absolute Difference')
    height = sim.height - sim.conf[('Layers', 'Air', 'params', 'thickness', 'value')]
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
    middle = int(round(sim.y_samples / 2))
    sim.draw_geometry_2d('yz', middle, ax, skip_list=['Air'])
    plt.savefig(pname)
    plt.show()


def main():
    parser = ap.ArgumentParser(description="""Compare a COMSOL data file to an
                               RCWA one""")
    parser.add_argument('rcwa', help='The RCWA YAML config file')
    parser.add_argument('comsol', help='The COMSOL text file')
    args = parser.parse_args()

    sim = Simulation(Config(args.rcwa))
    freq = sim.conf[('Simulation', 'params', 'frequency', 'value')]
    lanczos = sim.conf[('Solver', 'LanczosSmoothing')]
    basis = sim.conf[('Simulation', 'params', 'numbasis', 'value')]
    rcwa_normE = get_middle_slice(sim)**2
    comsol_normE = get_comsol_normE(sim, args.comsol)**2
    rcwa_normE, comsol_normE = strip_air(sim, rcwa_normE, comsol_normE)
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
    compare(rcwa_normE, comsol_normE, title, sim, pname=pname)
    plot_diff(rcwa_normE, comsol_normE, title, sim, pname=pname2)

if __name__ == '__main__':
    main()
