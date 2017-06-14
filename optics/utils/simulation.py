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
        self.e_data = None
        self.h_data = None
        self.failed = False
        self.e_lookup = {}
        self.h_lookup = {}
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

    def load_txt(self, path):
        try:
            with open(path, 'r') as dfile:
                hline = dfile.readline()
                if hline[0] == '#':
                    # We have headers in the file pandas read_csv is WAY faster
                    # than loadtxt. We've already read the header line so we
                    # pass header=None
                    d = pandas.read_csv(dfile, delim_whitespace=True,
                                        header=None, skip_blank_lines=True)
                    data = d.as_matrix()
                    headers = hline.strip('#\n').split(',')
                    lookup = {headers[ind]: ind for ind in range(len(headers))}
                    self.log.debug(
                        'Here is the E field header lookup: %s', str(lookup))
                else:
                    # Make sure we seek to beginning of file so we don't lose
                    # the first row
                    dfile.seek(0)
                    d = pandas.read_csv(dfile, delim_whitespace=True,
                                        header=None, skip_blank_lines=True)
                    data = d.as_matrix()
                    self.log.debug('File is missing headers')
                    lookup = None
        except FileNotFoundError:
            self.log.error('Following file missing: %s', path)
            self.failed = True
            data, lookup = None, None
        return data, lookup

    def load_npz(self, path):
        try:
            # Get the headers and the data
            with np.load(path) as loaded:
                data = loaded['data']
                # lookup = {key:i for i, key in enumerate(loaded['headers'])}
                lookup = loaded['headers'][0]
                self.log.debug(str(type(lookup)))
                self.log.debug(str(lookup))
                self.log.debug(
                    'Here is the E field header lookup: %s', str(lookup))
        except IOError:
            self.log.error('Following file missing or unloadable: %s', path)
            self.failed = True
            data, lookup = None, None
        return data, lookup

    def get_data(self):
        """Returns the already crunched E and H data for this particular sim"""
        self.log.info('Collecting data for sim %s',
                      self.conf['General']['sim_dir'])
        sim_path = self.conf['General']['sim_dir']
        base_name = self.conf['General']['base_name']
        ftype = self.conf['General']['save_as']
        ignore = self.conf['General']['ignore_h']
        # If data was saved into text files
        if ftype == 'text':
            # Load E field data
            e_path = os.path.join(sim_path, base_name + '.E')
            e_data, e_lookup = self.load_txt(e_path)
            # Load H field data
            if not ignore:
                h_path = os.path.join(sim_path, base_name + '.H')
                h_data, h_lookup = self.load_txt(h_path)
            else:
                h_data, h_lookup = None, {}
        # If data was saved in in npz format
        elif ftype == 'npz':
            # Get the paths
            e_path = os.path.join(sim_path, base_name + '.E.npz')
            e_data, e_lookup = self.load_npz(e_path)
            self.log.debug('E shape after getting: %s', str(e_data.shape))
            if not ignore:
                h_path = os.path.join(sim_path, base_name + '.H.npz')
                h_data, h_lookup = self.load_npz(h_path)
            else:
                h_data, h_lookup = None, {}
        else:
            raise ValueError('Incorrect file type specified in [General] '
                             'section of config file')
        self.e_data, self.e_lookup, self.h_data, self.h_lookup = e_data, e_lookup, h_data, h_lookup
        self.log.info('Collection complete!')
        return e_data, e_lookup, h_data, h_lookup

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
            self.log.info('Available averages: %s' % str(self.avgs.keys()))
        elif ftype == 'npz':
            path = os.path.join(base, 'all.avg.npz')
            with np.load(path) as avgs_file:
                for arr in avgs_file.files:
                    self.avgs[arr] = avgs_file[arr]
            self.log.info('Available averages: %s' % str(self.avgs.keys()))

    def write_data(self):
        """Writes the data. All writes have been wrapped with an atomic context
        manager than ensures all writes are atomic and do not corrupt data
        files if they are interrupted for any reason"""
        start = time.time()
        # Get the current path
        base = self.conf['General']['sim_dir']
        ignore = self.conf['General']['ignore_h']
        self.log.info('Writing data for %s' % base)
        fname = self.conf['General']['base_name']
        epath = os.path.join(base, fname + '.E')
        hpath = os.path.join(base, fname + '.H')
        # Save matrices in specified file tipe
        self.log.debug('Here are the E matrix headers: %s', str(self.e_lookup))
        self.log.debug('Here are the H matrix headers: %s', str(self.h_lookup))
        ftype = self.conf['General']['save_as']
        if ftype == 'text':
            epath = epath
            # These make sure all writes are atomic and thus we won't get any
            # partially written files if processing is interrupted for any
            # reason (like a keyboard interrupt)
            with open_atomic(epath, 'w', npz=False) as out:
                np.savetxt(out, self.e_data, header=','.join(
                    self.e_lookup.keys()))
            if not ignore:
                hpath = hpath + '.crnch'
                with open_atomic(hpath, 'w', npz=False) as out:
                    np.savetxt(out, self.h_data, header=','.join(
                        self.h_lookup.keys()))
            # Save any local averages we have computed
            for avg, mat in self.avgs.items():
                dpath = os.path.join(base, avg + '.avg.crnch')
                with open_atomic(dpath, 'w', npz=False) as out:
                    np.savetxt(out, mat)
        elif ftype == 'npz':
            # Save the headers and the data
            with open_atomic(epath, 'w') as out:
                np.savez_compressed(out, headers=np.array(
                    [self.e_lookup]), data=self.e_data)
            if not ignore:
                with open_atomic(hpath, 'w') as out:
                    np.savez_compressed(out, headers=np.array(
                        [self.h_lookup]), data=self.h_data)
            # Save any local averages we have computed
            dpath = os.path.join(base, 'all.avg')
            with open_atomic(dpath, 'w') as out:
                np.savez_compressed(out, **self.avgs)
        else:
            raise ValueError('Specified saving in an unsupported file format')
        end = time.time()
        self.log.info('Write time: {:.2} seconds'.format(end - start))

    def get_scalar_quantity(self, quantity):
        self.log.debug('Retrieving scalar quantity %s', str(quantity))
        self.log.debug(type(self.e_lookup))
        self.log.debug('E Header: %s', str(self.e_lookup))
        self.log.debug('H Header: %s', str(self.h_lookup))
        try:
            col = self.e_lookup[quantity]
            self.log.debug(col)
            self.log.debug('Column of E field quantity %s is %s',
                           str(quantity), str(col))
            return self.e_data[:, col]
        except KeyError:
            col = self.h_lookup[quantity]
            self.log.debug(col)
            self.log.debug('Column of H field quantitty %s is %s',
                           str(quantity), str(col))
            return self.h_data[:, col]
        except KeyError:
            self.log.error('You attempted to retrieve a quantity that does not exist in the e and h \
                    matrices')
            raise

    def clear_data(self):
        """Clears all the data attributes to free up memory"""
        self.e_data = None
        self.h_data = None
        self.avgs = {}

    def extend_data(self, quantity, new_data, field):
        """Adds a new column of data to the field arrays, or updates it if it
        already exists"""
        # TODO: This is NOT foolproof. Need a better way of distinguishing
        # between E field and H field quantities
        if field == "H":
            self.log.info("Inserting an H field quantity")
            # If this quantity is already in the matrix, just update it.
            # Otherwise append it as a new column
            if quantity in self.h_lookup:
                self.log.debug("Quantity {} exists in matrix, "
                               "updating".format(quantity))
                col = self.h_lookup[quantity]
                self.h_data[:, col] = new_data
            else:
                self.log.debug('Matrix shape before extending: %s', str(self.h_data.shape))
                # This approach is 4 times faster than np.column_stack()
                dat = np.zeros((self.h_data.shape[0], self.h_data.shape[1] + 1))
                dat[:, :-1] = self.h_data
                dat[:, -1] = new_data
                self.h_data = dat
                # Now append this quantity and its column the the header dict
                self.h_lookup[quantity] = dat.shape[1] - 1
                self.log.debug('Matrix shape after extending: %s', str(self.h_data.shape))
        else:
            self.log.info("Inserting an E field quantity")
            if quantity in self.e_lookup:
                self.log.debug("Quantity {} exists in matrix, "
                               "updating".format(quantity))
                col = self.e_lookup[quantity]
                self.e_data[:, col] = new_data
            else:
                self.log.debug('Matrix shape before extending: %s' % str(self.e_data.shape))
                # This approach is 4 times faster than np.column_stack()
                dat = np.zeros((self.e_data.shape[0], self.e_data.shape[1] + 1))
                dat[:, :-1] = self.e_data
                dat[:, -1] = new_data
                self.e_data = dat
                # Now append this quantity and its column the the header dict
                self.e_lookup[quantity] = dat.shape[1] - 1
                self.log.debug('Matrix shape after extending: %s' % str(self.e_data.shape))
