"""
Script to remove postprocessed data nodes from an HDF5 file for a simulation.
In the event that one of these nodes becomes corrupted, it becomes impossible
to read from or write to that node, and it must be deleted and recreated from
scratch
"""

import sys
import glob
import os.path as osp
import tables as tb
from nanowire.utils.config import Config

SCRIPTS_DIR = osp.dirname(__file__)

conf_path = osp.abspath(sys.argv[1])
print("Loading config {} ...".format(conf_path))
conf = Config.fromFile(conf_path)
ID = conf.ID
print("Config ID: {}".format(ID))
hdf_path = osp.join(osp.dirname(conf_path), 'sim.hdf5')
print("Loading HDF5 file {} ...".format(hdf_path))


to_remove = ['power_absorbed', 'power_absorbed_rescaled', 'normEsquared',
             'normEsquared_rescaled', 'rescaling_factors', 'genRate',
             'genRate_rescaled', 'normE_rescaled',
             'genRate_angularAvg_rescaled', 'transmission_data']
hdf = tb.open_file(hdf_path, 'r+')
for node in to_remove:
    print("Removing node {}".format(node))
    node_path = '/sim_{}/{}'.format(ID, node)
    try:
        node = hdf.remove_node(node_path)
    except tb.NoSuchNodeError:
        print("Node {} not in {}".format(node, hdf_path))
hdf.close()
