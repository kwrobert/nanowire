import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from nanowire.optics.utils.utils import get_nk
import os
import glob

this_file = os.path.realpath(__file__)
nk_dir = os.path.normpath(os.path.join(this_file, '../../nanowire/NK/'))
nk_files = glob.glob(os.path.join(nk_dir, '00*.txt'))
for f in nk_files:
    data = np.loadtxt(f)
    # alpha = 2* omega * k / c
    alpha = 2*data[:,0]*data[:, 2]/const.c
    s4_alpha = alpha*1e-6
    wvs = 1e9*const.c/data[:, 0]
    diff = np.abs(wvs - 367)
    index = np.argmin(diff)
    print(s4_alpha[index])
    plt.figure()
    plt.plot(wvs, s4_alpha)
    title = os.path.basename(f).split('_')[1]
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption Coefficient')
    plt.show()
