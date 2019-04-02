"""
Performs a series of tests to make sure all the data was written to the HDF5
file properly. Accepts a path to a simulation config file as the only CLI
argument
"""

import sys
import os
import argparse
import os.path as osp
import tables as tb
import glob
from nanowire.utils.config import Config

SCRIPTS_DIR = osp.dirname(__file__)

def check_data(conf_path):
    hdf_path = osp.join(osp.dirname(conf_path), 'sim.hdf5')
    sol_path = osp.join(osp.dirname(hdf_path), 'solution.bin')

    if not osp.isfile(sol_path):
        print(conf_path)
        sys.exit(0)

    try:
        hdf = tb.open_file(hdf_path)
    except tb.exceptions.HDF5ExtError:
        print(conf_path)
        sys.exit(0)
    except OSError:
        print(conf_path)
        sys.exit(0)

    nodes = len(hdf.list_nodes('/', classname='Group'))
    if nodes < 1:
        print(conf_path)
        hdf.close()
        sys.exit(0)

    expected_nodes = ['Air_amplitudes',
                      'Air_qvals',
                      'Ex',
                      'Ey',
                      'Ez',
                      'ITO_amplitudes',
                      'ITO_qvals',
                      'NW_AlShell_amplitudes',
                      'NW_AlShell_qvals',
                      'NW_SiO2_amplitudes',
                      'NW_SiO2_qvals',
                      'Substrate_amplitudes',
                      'Substrate_qvals',
                      'fluxes',
                      'xcoords',
                      'ycoords',
                      'zcoords']

    existing_nodes = set(node._v_name for node in hdf.walk_nodes("/"))
    conf = Config.fromFile(conf_path)
    group_loc = "/sim_{}".format(conf.ID)
    for node in expected_nodes:
        if node not in existing_nodes:
            print(conf_path)
            hdf.close()
            sys.exit(0)
        try:
            # Also need to test reading the data
            hdf_node = hdf.get_node(group_loc, name=node)
            hdf_node.read(start=0, stop=1)
        except Exception as e:
            print(conf_path)
            hdf.close()
            sys.exit(0)
    hdf.close()

def main():
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('conf_path', help="""Path to Simulation YAML config file""")
    args = parser.parse_args()
    check_data(args.conf_path)

if __name__ == '__main__':
    main()
