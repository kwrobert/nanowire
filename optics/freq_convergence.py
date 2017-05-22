import shutil
import itertools
import os
import copy
import multiprocessing as mp
import numpy as np
import scipy.optimize as optz
import time
import argparse as ap
import ruamel.yaml as yaml
import logging

# get our custom config object and the logger function
import sim_wrapper as sw
import postprocess as pp
import matplotlib.pyplot as plt
from utils.simulator import Simulator
from utils.config import Config
from utils.utils import configure_logger
from collections import OrderedDict


def main():

    parser = ap.ArgumentParser(description="""Runs a convergence analysis on
                               the number of frequency bins to see how many we
                               actually need""")
    parser.add_argument('config_file', type=str, help="""Absolute path to the
    YAML file specifying how you want this wrapper to behave""")
    parser.add_argument('--log_level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        global_conf = Config(path=os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    freq_bins = list(range(60, 150, 10))
    fracs = []
    study_base = global_conf['General']['base_dir']
    for num_bins in freq_bins:
        subdir = 'num_bins_{}'.format(num_bins)
        sweep_dir = os.path.join(study_base, subdir)
        global_conf['General']['base_dir'] = sweep_dir
        global_conf['General']['treebase'] = sweep_dir
        global_conf['Simulation.params.frequency.step'] = num_bins
        print('Running with {} bins under {}'.format(num_bins, sweep_dir))
        sw.run(global_conf, args.log_level)
        print('Computing fractional absorption ...')
        # Compute transmission data for each individual sim
        cruncher = pp.Cruncher(global_conf)
        cruncher.collect_sims()
        for sim in cruncher.sims:
            cruncher.transmissionData(sim)
        # Now get the fraction of photons absorbed
        gcruncher = pp.Global_Cruncher(
            global_conf, cruncher.sims, cruncher.sim_groups, cruncher.failed_sims)
        gcruncher.group_against(
            ['Simulation', 'params', 'frequency', 'value'], global_conf.variable)
        photon_fraction = gcruncher.fractional_absorbtion()[0]
        print('Photon fraction at {} bins: {}'.format(num_bins,
                                                      photon_fraction))
        fracs.append(photon_fraction)

    plt.figure()
    plt.xlabel('Number of Frequency Bins')
    plt.ylabel('Spectral Absorbance')
    plt.title('Frequency Bin Convergence Analysis')
    plt.plot(freq_bins, fracs)
    plt.show()
    plt.savefig(os.path.join(study_base, 'frequency_convergence.pdf'))


if __name__ == '__main__':
    main()
