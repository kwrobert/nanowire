import os
import os.path as osp
import tables as tb
import glob

SCRIPTS_DIR = osp.dirname(__file__)
glob_path = osp.join(SCRIPTS_DIR, '../*/sim.hdf5')
data_files = glob.glob(glob_path)

failed = []
for hdf_path in data_files:
    print("Testing file {}".format(hdf_path))
    try:
        hdf = tb.open_file(hdf_path)
        nodes = len(hdf.list_nodes('/', classname='Group'))
        if nodes < 1:
            print('File {} has no data'.format(hdf_path))
            failed.append(hdf_path)
    except tb.exceptions.HDF5ExtError:
        print('File {} is corrupt'.format(hdf_path))
        failed.append(hdf_path)
    else:
        hdf.close()
with open('failed_sims.txt', 'w') as f:
    for path in failed:
        abs_path = osp.abspath(path)
        f.write(abs_path + '\n')
