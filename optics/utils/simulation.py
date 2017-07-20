import pandas
import numpy as np
import os
import time
import glob
import logging
from .utils import open_atomic


class Simulation(object):
    """
    An object that represents a simulation. It contains the data for the
    sim, the data file headers, and its configuration object as attributes.
    This object contains all the information about a simulation, but DOES NOT
    perform any actions on that data. That job is relegated to the various
    processor objects
    """

    def __init__(self, conf):
        self.conf = conf
        self.log = logging.getLogger('postprocess')
        self.data = None
        self.failed = False
        self.avgs = {}
        # Compute and store dx, dy, dz at attributes
        self.z_samples = int(conf['Simulation']['z_samples'])
        self.x_samples = int(conf['Simulation']['x_samples'])
        self.y_samples = int(conf['Simulation']['y_samples'])
        max_depth = conf['Simulation']['max_depth']
        if max_depth:
            self.height = max_depth
            self.dz = max_depth / self.z_samples
        else:
            self.height = self.conf.get_height()
            self.dz = self.height / self.z_samples
        self.period = conf['Simulation']['params']['array_period']['value']
        self.dx = self.period / self.x_samples
        self.dy = self.period / self.y_samples

    def load_npz(self, path):
        """Load all field data for this simulation from an npz file"""
        try:
            # Get the headers and the data
            with np.load(path) as loaded:
                data = {key: arr for key, arr in loaded.iteritems()}
                self.log.debug('Quantities in npz file: %s', str(data.keys()))
        except IOError:
            self.log.error('Following file missing or unloadable: %s', path)
            self.failed = True
            data = None
        return data

    def get_data(self):
        """Loads all the data contained in the npz file into a self.data
        dictionary attribute"""
        self.log.info('Collecting data for sim %s',
                      self.conf['General']['sim_dir'])
        sim_path = self.conf['General']['sim_dir']
        base_name = self.conf['General']['base_name']
        ftype = self.conf['General']['save_as']
        # If data was saved in in npz format
        if ftype == 'npz':
            # Get the paths
            path = os.path.join(sim_path, base_name + '.npz')
            data = self.load_npz(path)
        else:
            raise ValueError('Incorrect file type specified in [General] '
                             'section of config file')
        self.data = data
        self.log.info('Collection complete!')
        return data

    def get_avgs(self):
        """Load all averages"""
        # Get the current path
        base = self.conf['General']['sim_dir']
        ftype = self.conf['General']['save_as']
        if ftype == 'text':
            globstr = os.path.join(base, '*avg.crnch')
            for f in glob.glob(globstr):
                fname = os.path.basename(f)
                key = fname[0:fname.index('.')]
                self.avgs[key] = np.loadtxt(f)
            self.log.info('Available averages: %s', str(self.avgs.keys()))
        elif ftype == 'npz':
            path = os.path.join(base, 'all.avg.npz')
            with np.load(path) as avgs_file:
                for arr in avgs_file.files:
                    self.avgs[arr] = avgs_file[arr]
            self.log.info('Available averages: %s', str(self.avgs.keys()))

    def write_data(self):
        """Writes the data. All writes have been wrapped with an atomic context
        manager than ensures all writes are atomic and do not corrupt data
        files if they are interrupted for any reason"""
        start = time.time()
        # Get the current path
        base = self.conf['General']['sim_dir']
        self.log.info('Writing data for %s', base)
        fname = os.path.join(base, self.conf['General']['base_name'])
        # Save matrices in specified file tipe
        ftype = self.conf['General']['save_as']
        if ftype == 'npz':
            # Save the headers and the data
            with open_atomic(fname, 'w') as out:
                np.savez_compressed(out, **self.data)
            # Save any local averages we have computed
            dpath = os.path.join(base, 'all.avg')
            with open_atomic(dpath, 'w') as out:
                np.savez_compressed(out, **self.avgs)
        else:
            raise ValueError('Specified saving in an unsupported file format')
        end = time.time()
        self.log.info('Write time: %.2f seconds', end - start)

    def get_scalar_quantity(self, quantity):
        """
        Retrieves the entire 3D matrix for some scalar quantity

        :param str quantity: The quantity you would like to retrive (ex: 'Ex'
                             or 'normE')
        :return: A 3D numpy array of the specified quantity
        :raises KeyError: If the specified quantity does not exist in the data
                          dict

        """
        self.log.debug('Retrieving scalar quantity %s', str(quantity))
        try:
            return self.data[quantity]
        except KeyError:
            self.log.error('You attempted to retrieve a quantity that does not'
                           ' exist in the data dict')
            raise

    def clear_data(self):
        """Clears all the data attributes to free up memory"""
        self.data = {}
        self.avgs = {}

    def extend_data(self, quantity, new_data):
        """
        Adds a new column of data to the field arrays, or updates it if it
        already exists
        """
        if quantity in self.data:
            self.log.debug("Quantity %s exists in matrix, updating", quantity)
            self.data[quantity] = new_data
        else:
            self.log.debug('Adding %s to data dict', str(quantity))
            self.data[quantity] = new_data
