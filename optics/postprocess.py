import numpy as np
from scipy import interpolate
import scipy.constants as c
import scipy.integrate as intg
import argparse as ap
import os
import time
import copy
import re
import glob
import logging
import itertools
from collections import OrderedDict
import matplotlib
# Enables saving plots over ssh
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use(['ggplot', 'paper'])
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Literally just for the initial data load
import multiprocessing as mp
import multiprocessing.dummy as mpd

from utils.config import Config
from utils.simulation import Simulation
from utils.utils import configure_logger, cmp_dicts, open_atomic

# Configure module level logger if not running as main process
if not __name__ == '__main__':
    logger = configure_logger(level='INFO', name='postprocess',
                              console=True, logfile='logs/postprocess.log')


def counted(fn):
    def wrapper(self):
        wrapper.called += 1
        return fn(self)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper


class Processor(object):
    """Base data processor class that has some methods every other processor needs"""

    def __init__(self, global_conf, sims=[], sim_groups=[], failed_sims=[]):
        self.log = logging.getLogger('postprocess')
        self.log.debug("Processor base init")
        self.gconf = global_conf
        self.sims = sims
        self.sim_groups = sim_groups
        # A place to store any failed sims (i.e sims that are missing their
        # data file)
        self.failed_sims = failed_sims

    def collect_sims(self):
        """Collect all the simulations beneath the base of the directory tree"""
        sims = []
        failed_sims = []
        ftype = self.gconf['General']['save_as']
        if ftype == 'npz':
            datfile = self.gconf['General']['base_name'] + '.npz'
        else:
            raise ValueError('Invalid file type specified in config')
        for root, dirs, files in os.walk(self.gconf['General']['base_dir']):
            conf_path = os.path.join(root, 'sim_conf.yml')
            if 'sim_conf.yml' in files and datfile in files:
                self.log.info('Gather sim at %s', root)
                sim_obj = Simulation(Config(conf_path))
                sim_obj.conf.expand_vars()
                sims.append(sim_obj)
            elif 'sim_conf.yml' in files:
                sim_obj = Simulation(Config(conf_path))
                self.log.error('The following sim is missing its data file: %s',
                               sim_obj.conf['General']['sim_dir'])
                failed_sims.append(sim_obj)
        self.sims = sims
        self.failed_sims = failed_sims
        return sims, failed_sims

    def sort_sims(self):
        """Sorts simulations by their parameters the way a human would. Called human sorting or
        natural sorting. Thanks stackoverflow"""

        self.sims.sort(key=self.sim_key)
        for group in self.sim_groups:
            paths = [sim.conf['General']['sim_dir'] for sim in group]
            self.log.debug('Group paths before sorting: %s', str(paths))
            group.sort(key=self.sim_key)
            paths = [sim.conf['General']['sim_dir'] for sim in group]
            self.log.debug('Group paths after sorting: %s', str(paths))

    def get_param_vals(self, parseq):
        """Return all possible values of the provided parameter for this sweep"""
        vals = []
        for sim in self.sims:
            val = sim.conf[parseq]
            if val not in vals:
                vals.append(val)
        return vals

    def filter_by_param(self, pars):
        """Accepts a dict where the keys are parameter names and the values are a list of possible
        values for that parameter. Any simulation whose parameter does not match any of the provided
        values is removed from the sims and sim_groups attribute"""

        assert(type(pars) == dict)
        for par, vals in pars.items():
            self.sims = [sim for sim in self.sims if sim.conf[par] in vals]
            groups = []
            for group in self.sim_groups:
                filt_group = [sim for sim in group if sim.conf[par] in vals]
                groups.append(filt_group)
            self.sim_groups = groups
        assert(len(self.sims) >= 1)
        return self.sims, self.sim_groups

    def group_against(self, key, variable_params, sort_key=None):
        """Groups simulations by against particular parameter. Within each
        group, the parameter specified will vary, and all other
        parameters will remain fixed. Populates the sim_groups attribute and
        also returns a list of lists. The simulations with each group will be
        sorted in increasing order of the specified parameter. An optional key
        may be passed in, the groups will be sorted in increasing order of the
        specified key"""

        self.log.info('Grouping sims against: %s' % str(key))
        # We need only need a shallow copy of the list containing all the sim objects
        # We don't want to modify the orig list but we wish to share the sim
        # objects the two lists contain
        sims = copy.copy(self.sims)
        sim_groups = [[sims.pop()]]
        # While there are still sims that havent been assigned to a group
        while sims:
            # Get the comparison dict for this sim
            sim = sims.pop()
            val1 = sim.conf[key]
            # We want the specified key to vary, so we remove it from the
            # comparison dict
            del sim.conf[key]
            cmp1 = {'Simulation': sim.conf[
                'Simulation'], 'Layers': sim.conf['Layers']}
            match = False
            # Loop through each group, checking if this sim belongs in the
            # group
            for group in sim_groups:
                sim2 = group[0]
                val2 = sim2.conf[key]
                del sim2.conf[key]
                cmp2 = {'Simulation': group[0].conf[
                    'Simulation'], 'Layers': group[0].conf['Layers']}
                params_same = cmp_dicts(cmp1, cmp2)
                if params_same:
                    match = True
                    # We need to restore the param we removed from the
                    # configuration earlier
                    sim.conf[key] = val1
                    group.append(sim)
                group[0].conf[key] = val2
            # If we didnt find a matching group, we need to create a new group
            # for this simulation
            if not match:
                sim.conf[key] = val1
                sim_groups.append([sim])
        # Get the params that will define the path in the results dir for each
        # group that will be stored
        ag_key = tuple(key[0:-1])
        result_pars = [var for var in variable_params if var != ag_key]
        for group in sim_groups:
            # Sort the individual sims within a group in increasing order of
            # the parameter we are grouping against a
            group.sort(key=lambda sim: sim.conf[key])
            path = '{}/grouped_against_{}'.format(group[0].conf['General']['treebase'],
                                                  ag_key[-1])
            # If the only variable param is the one we grouped against, make
            # the top dir
            if not result_pars:
                try:
                    os.makedirs(path)
                except OSError:
                    pass
            # Otherwise make the top dir and all the subdirs
            else:
                for par in result_pars:
                    full_key = par + ('value',)
                    # All sims in the group will have the same values for
                    # result_pars so we can just use the first sim in the group
                    path = os.path.join(path, '{}_{:.4E}/'.format(par[-1],
                                                                  group[0].conf[full_key]))
                    self.log.info('RESULTS DIR: {}'.format(path))
                    try:
                        os.makedirs(path)
                    except OSError:
                        pass
            for sim in group:
                sim.conf['General']['results_dir'] = path
        # Sort the groups in increasing order of the provided sort key
        if sort_key:
            sim_groups.sort(key=lambda group: group[0].conf[key])
        self.sim_groups = sim_groups
        return sim_groups

    def group_by(self, key, sort_key=None):
        """Groups simulations by a particular parameter. Within each group, the
        parameter specified will remain fixed, and all other parameters will
        vary. Populates the sim_groups attribute and also returns a list of
        lists. The groups will be sorted in increasing order of the specified
        parameter. An optional key may be passed in, the individual sims within
        each group will be sorted in increasing order of the specified key"""

        self.log.info('Grouping sims by: %s', str(key))
        # This works by storing the different values of the specifed parameter
        # as keys, and a list of sims whose value matches the key as the value
        pdict = {}
        for sim in self.sims:
            if sim.conf[key] in pdict:
                pdict[sim.conf[key]].append(sim)
            else:
                pdict[sim.conf[key]] = [sim]
        # Now all the sims with matching values for the provided key are just
        # the lists located at each key. We sort the groups in increasing order
        # of the provided key
        groups = sorted(pdict.values(), key=lambda group: group[0].conf[key])
        # If specified, sort the sims within each group in increasing order of
        # the provided sorting key
        if sort_key:
            for group in groups:
                group.sort(key=lambda sim: sim.conf[sort_key])
        self.sim_groups = groups
        return groups

    def get_plane(self, scalar, plane, pval):
        """Given a 3D array containing values for a 3D scalar field, returns a
        2D array containing the data on a given plane, for a specified index
        value (pval) of that plane. So, specifying plane=xy and pval=30 would
        return data on the 30th x,y plane (a plane at the given z index). The
        number of samples (i.e data points) in each coordinate direction need
        not be equal"""

        if plane == 'yz' or plane == 'zy':
            # z along rows, y along columns
            return scalar[:, pval, :]
        elif plane == 'xz' or plane == 'zx':
            # x along columns, z along rows
            return scalar[:, :, pval]
        elif plane == 'xy' or plane == 'yx':
            # x along rows, y along columns
            return scalar[pval, :, :]

    def get_line(self, arr, xsamp, ysamp, zsamp, line_dir, c1, c2):
        """Given a 1D array containing values for a 3D scalar field, reshapes
        the array into 3D and returns a new 1D array containing the data on a
        line in a given direction, for a specified index value for the other
        two spatial coordinates. So, specifying line_dir=z and c1=5,c2=5 would
        return all the data along the z-direction at the 5th x,y index. Note
        coordinates c1,c2 must always be specified in (x,y,z) order"""

        scalar = arr.reshape(zsamp + 1, xsamp, ysamp)
        if line_dir == 'x':
            # z along rows, y along columns
            return scalar[c2, :, c1]
        elif line_dir == 'y':
            # x along columns, z along rows
            return scalar[c2, c1, :]
        elif line_dir == 'z':
            # x along rows, y along columns
            return scalar[:, c1, c2]

    def process(self, sim):
        """Retrieves data for a particular simulation, then processes that
        data"""
        raise NotImplementedError

    def process_all(self):
        """Processes all the sims collected and stored in self.sims and
        self.sim_groups"""
        raise NotImplementedError


class Cruncher(Processor):
    """Crunches all the raw data. Calculates quantities specified in the global
    config file and either appends them to the existing data files or creates
    new ones as needed"""

    def __init__(self, global_conf, sims=[], sim_groups=[], failed_sims=[]):
        super(Cruncher, self).__init__(
            global_conf, sims, sim_groups, failed_sims)
        self.log.debug("This is THE CRUNCHER!!!!!")

    def calculate(self, quantity, sim, args):
        try:
            result = getattr(self, quantity)(sim, *args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity, exc_info=True, stack_info=True)
            raise

    def process(self, sim):
        sim_path = os.path.basename(sim.conf['General']['sim_dir'])
        self.log.info('Crunching data for sim %s', sim_path)
        sim.get_data()
        if sim.failed:
            self.log.error('Following simulation missing data: %s', sim_path)
            self.failed_sims.append(sim)
        else:
            # For each quantity
            for quant, data in self.gconf['Postprocessing']['Cruncher'].items():
                if data['compute']:
                    argsets = data['args']
                    self.log.info('Computing %s with args %s',
                                  str(quant), str(argsets))
                    if argsets and type(argsets[0]) == list:
                        for argset in argsets:
                            self.log.info('Computing individual argset'
                                          ' %s', str(argset))
                            if argset:
                                self.calculate(quant, sim, argset)
                            else:
                                self.calculate(quant, sim, [])
                    else:
                        if argsets:
                            self.calculate(quant, sim, argsets)
                        else:
                            self.calculate(quant, sim, [])
            sim.write_data()
            sim.clear_data()

    def process_all(self):
        self.log.info('Beginning data crunch ...')
        if not self.gconf['General']['post_parallel']:
            for sim in self.sims:
                self.process(sim)
        else:
            num_procs = mp.cpu_count() - \
                self.gconf['General']['reserved_cores']
            self.log.info(
                'Crunching sims in parallel using %s cores ...', str(num_procs))
            pool = mpd.Pool(processes=num_procs)
            pool.map(self.process, self.sims)

    def normE(self, sim):
        """
        Calculates the norm of E. Adds it to the data dict for the simulation
        and also returns a 3D array 
        :param :class: `utils.Simulation.Simulation` sim: 
        The simulation object for which you wish to calculate normE
        :return: A 3D numpy array containing normE
        """

        # Get the magnitude of E and add it to our data
        E_mag = np.zeros_like(sim.data['Ex'])
        for comp in ('Ex', 'Ey', 'Ez'):
            E_mag += np.absolute(sim.data[comp])**2
        E_mag = np.sqrt(E_mag)
        sim.extend_data('normE', E_mag.real)
        return E_mag.real

    def normEsquared(self, sim):
        """Calculates and returns normE squared"""

        # Get the magnitude of E and add it to our data
        E_magsq = np.zeros_like(sim.data['Ex'])
        for comp in ('Ex', 'Ey', 'Ez'):
            E_magsq += np.absolute(sim.data[comp])**2
        sim.extend_data('normEsquared', E_magsq.real)
        return E_magsq.real

    def normH(self, sim):
        """Calculate and returns the norm of H"""

        H_mag = np.zeros_like(sim.data['Hx'])
        for comp in ('Hx', 'Hy', 'Hz'):
            H_mag += np.absolute(sim.data[comp])**2
        H_mag = np.sqrt(H_mag)
        sim.extend_data('normH', H_mag.real)
        return H_mag.real

    def normHsquared(self, sim):
        """Calculates and returns the norm of H squared"""

        H_magsq = np.zeros_like(sim.data['Hx'])
        for comp in ('Hx', 'Hy', 'Hz'):
            H_magsq += np.absolute(sim.data[comp])**2
        sim.extend_data('normHsquared', H_magsq.real)
        return H_magsq.real

    def get_nk(self, path, freq):
        """Returns functions to compute index of refraction components n and k at a given
        frequency"""
        # Get data
        freq_vec, n_vec, k_vec = np.loadtxt(path, unpack=True)
        # Get n and k at specified frequency via interpolation
        f_n = interpolate.interp1d(freq_vec, n_vec, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        f_k = interpolate.interp1d(freq_vec, k_vec, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        return f_n(freq), f_k(freq)

    def _get_circle_nk(self, shape, sdata, nk, samps, steps):
        """Returns a 2D matrix containing the N,K values at each point in space
        for a circular in-plane geometry"""
        # Set up the parameters
        cx, cy = sdata['center']['x'], sdata['center']['y']
        rad = sdata['radius']
        rad_sq = rad * rad
        mat = sdata['material']
        dx = steps[0]
        dy = steps[1]
        # Build the matrix
        nk_mat = np.zeros((samps[1], samps[0]))
        for xi in range(samps[0]):
            for yi in range(samps[1]):
                dist = ((xi * dx) - cx)**2 + ((yi * dy) - cy)**2
                if dist <= rad_sq:
                    nk_mat[yi, xi] = nk[mat][0] * nk[mat][1]
        return nk_mat

    def _genrate_nk_geometry(self, sim, lname, nk, samps, steps):
        """Computes the nk profile in a layer with a nontrivial internal
        geometry. Returns a 2D matrix containing the product of n and k at each
        point
        lname: Name of the layer as a string
        ldata: This dict containing all the data for this layer
        nk: A dictionary with the material name as the key an a tuple
        containing (n,k) as the value
        samps: A tuple/list containing the number of sampling points in each
        spatial direction in (x,y,z) order
        steps: Same as samps but instead contains the step sizes in each
        direction"""
        # Initialize the matrix with values for the base material
        base_mat = sim.conf['Layers'][lname]['base_material']
        nk_mat = nk[base_mat][0] * nk[base_mat][1] * \
            np.ones((samps[1], samps[0]))
        # Get the shapes sorted in increasing order
        shapes = sim.conf.sorted_dict(sim.conf['Layers'][lname]['geometry'])
        # Loop through the layers. We want them in increasing order so the
        # smallest shape, which is contained within all the other shapes and
        # should override their nk values, goes last
        for shape, sdata in shapes.items():
            if sdata['type'] == 'circle':
                update = self._get_circle_nk(shape, sdata, nk, samps, steps)
            else:
                raise NotImplementedError('Computing generation rate for layers'
                                          ' with %s shapes is not currently supported' % sdata['type'])
            # Update the matrix with the values from this new shape. The update
            # array will contain nonzero values within the shape, and zero
            # everywhere else. This line updates the nk_mat with only the
            # nonzero from the update matrix, and leaves all other elements
            # untouched
            nk_mat = np.where(update != 0, update, nk_mat)
        return nk_mat

    def genRate(self, sim):
        # We need to compute normEsquared before we can compute the generation
        # rate
        try:
            normEsq = sim.get_scalar_quantity('normEsquared')
        except KeyError:
            normEsq = self.normEsquared(sim)
            sim.extend_data('normEsquared', normEsq)
            # Make sure we don't compute it twice
            try:
                sim.conf['Postprocessing']['Cruncher'][
                    'normEsquared']['compute'] = False
            except KeyError:
                pass
        # Prefactor for generation rate. Note we gotta convert from m^3 to cm^3,
        # hence 1e6 factor
        fact = c.epsilon_0 / (c.hbar * 1e6)
        # Get the indices of refraction at this frequency
        freq = sim.conf['Simulation']['params']['frequency']['value']
        nk = {mat: (self.get_nk(matpath, freq)) for mat, matpath in
              sim.conf['Materials'].items()}
        nk['vacuum'] = (1, 0)
        self.log.debug(nk)
        # Get spatial discretization
        samps = (sim.x_samples, sim.y_samples, sim.z_samples)
        gvec = np.zeros_like(normEsq)
        steps = (sim.dx, sim.dy, sim.dz)
        # Main loop to compute generation in each layer
        boundaries = []
        count = 0
        ordered_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        for layer, ldata in ordered_layers.items():
            # Get boundaries between layers and their starting and ending
            # indices
            layer_t = ldata['params']['thickness']['value']
            self.log.debug('LAYER: %s', layer)
            self.log.debug('LAYER T: %f', layer_t)
            if count == 0:
                start = 0
                end = int(layer_t / sim.dz) + 1
                boundaries.append((layer_t, start, end))
            else:
                prev_tup = boundaries[count - 1]
                dist = prev_tup[0] + layer_t
                start = prev_tup[2]
                end = int(dist / sim.dz) + 1
                boundaries.append((dist, start, end))
            self.log.debug('START: %i', start)
            self.log.debug('END: %i', end)
            if '' in ldata:
                self.log.debug('HAS GEOMETRY')
                # This function returns the N,K profile in that layer as a 2D
                # matrix. Each element contains the product of n and k at that
                # point, using the NK values for the appropriate material
                nk_mat = self._genrate_nk_geometry(
                    sim, layer, nk, samps, steps)
                gvec[start:end, :, :] = fact * \
                    nk_mat * normEsq[start:end, :, :]
            else:
                # Its just a simple slab
                self.log.debug('NO GEOMETRY')
                lmat = ldata['base_material']
                self.log.debug('LAYER MATERIAL: %s', lmat)
                self.log.debug('MATERIAL n: %s', str(nk[lmat][0]))
                self.log.debug('MATERIAL k: %s', str(nk[lmat][1]))
                region = fact * nk[lmat][0] * \
                    nk[lmat][1] * normEsq[start:end, :, :]
                self.log.debug('REGION SHAPE: %s', str(region.shape))
                self.log.debug('REGION: ')
                self.log.debug(str(region))
                gvec[start:end, :, :] = region
            self.log.debug('GEN RATE MATRIX: ')
            self.log.debug(str(gvec))
            count += 1
        sim.extend_data('genRate', gvec)
        return gvec

    def angularAvg(self, sim, quantity):
        """Perform an angular average of some quantity for either the E or H field"""
        try:
            quant = sim.get_scalar_quantity(quantity)
        except KeyError:
            self.calculate(quantity, sim, [])
            quant = sim.get_scalar_quantity(quantity)
            # Make sure we don't compute it twice
            try:
                sim.conf['Postprocessing']['Cruncher'][
                    quantity]['compute'] = False
            except KeyError:
                pass
        # Get spatial discretization
        rsamp = sim.conf['Simulation']['r_samples']
        thsamp = sim.conf['Simulation']['theta_samples']
        period = sim.conf['Simulation']['params']['array_period']['value']
        x = np.linspace(0, period, sim.x_samples)
        y = np.linspace(0, period, sim.y_samples)
        # Maximum r value such that circle and square unit cell have equal area
        rmax = period / np.sqrt(np.pi)
        # Diff between rmax and unit cell boundary at point of maximum
        # difference
        delta = rmax - period / 2.0
        # Extra indices we need to expand layers by
        x_inds = int(np.ceil(delta / sim.dx))
        y_inds = int(np.ceil(delta / sim.dy))
        # Use periodic BCs to extend the data in the x-y plane
        ext_vals = np.zeros((quant.shape[0], quant.shape[1] + \
                             2 * x_inds, quant.shape[2] + 2 * y_inds),
                            dtype=quant.dtype)
        # Left-Right extensions. This indexing madness extracts the slice we
        # want, flips it along the correct dimension then sticks in the correct
        # spot in the extended array
        ext_vals[:, x_inds:-x_inds, 0:y_inds] = quant[:, :, 0:y_inds][:, :, ::-1]
        ext_vals[:, x_inds:-x_inds, -
                 y_inds:] = quant[:, :, -y_inds:][:, :, ::-1]
        # Top-Bottom extensions
        ext_vals[:, 0:x_inds, y_inds:-
                 y_inds] = quant[:, 0:x_inds, :][:, ::-1, :]
        ext_vals[:, -x_inds:, y_inds:-
                 y_inds] = quant[:, -x_inds:, :][:, ::-1, :]
        # Corners, slightly trickier
        # Top left
        ext_vals[:, 0:x_inds, 0:y_inds] = ext_vals[
            :, x_inds:2 * x_inds, 0:y_inds][:, ::-1, :]
        # Bottom left
        ext_vals[:, -x_inds:, 0:y_inds] = ext_vals[:, -
                                                   2 * x_inds:-x_inds, 0:y_inds][:, ::-1, :]
        # Top right
        ext_vals[:, 0:x_inds, -y_inds:] = ext_vals[:,
                                                   0:x_inds, -2 * y_inds:-y_inds][:, :, ::-1]
        # Bottom right
        ext_vals[:, -x_inds:, -y_inds:] = ext_vals[:, -
                                                   x_inds:, -2 * y_inds:-y_inds][:, :, ::-1]
        # Now the center
        ext_vals[:, x_inds:-x_inds, y_inds:-y_inds] = quant[:, :, :]
        # Extend the points arrays to include these new regions
        x = np.concatenate((np.array([sim.dx * i for i in
                                      range(-x_inds, 0)]), x, np.array([x[-1] + sim.dx * i for i in range(1, x_inds + 1)])))
        y = np.concatenate((np.array([sim.dy * i for i in
                                      range(-y_inds, 0)]), y, np.array([y[-1] + sim.dy * i for i in range(1, y_inds + 1)])))
        # The points on which we have data
        points = (x, y)
        # The points corresponding to "rings" in cylindrical coordinates. Note we construct these
        # rings around the origin so we have to shift them to actually correspond to the center of
        # the nanowire
        rvec = np.linspace(0, rmax, rsamp)
        thvec = np.linspace(0, 2 * np.pi, thsamp)
        cyl_coords = np.zeros((len(rvec) * len(thvec), 2))
        start = 0
        for r in rvec:
            xring = r * np.cos(thvec)
            yring = r * np.sin(thvec)
            cyl_coords[start:start + len(thvec), 0] = xring
            cyl_coords[start:start + len(thvec), 1] = yring
            start += len(thvec)
        cyl_coords += period / 2.0
        # For every z layer in the 3D matrix of our quantity
        avgs = np.zeros((ext_vals.shape[0], len(rvec)))
        i = 0
        for layer in ext_vals:
            interp_vals = interpolate.interpn(
                points, layer, cyl_coords, method='linear')
            rings = interp_vals.reshape((len(rvec), len(thvec)))
            avg = np.average(rings, axis=1)
            avgs[i, :] = avg
            i += 1
        avgs = avgs[:, ::-1]
        # Save to avgs dict for this sim
        key = quantity + '_angularAvg'
        sim.avgs[key] = avgs
        return avgs

    def transmissionData(self, sim, port='Substrate'):
        """
        Computes reflection, transmission, and absorbance

        sim: :py:class:`utils.simulation.Simulation`
        port: string
            Default: 'substrate'
            Name of the location at which you would like to place the
            transmission port (i.e where you would like to compute
            transmission). This must correspond to one of the keys placed in
            the fluxes.dat file
        """
        
        self.log.info('Computing transmission data ...')
        base = sim.conf['General']['sim_dir']
        path = os.path.join(base, 'fluxes.dat')
        data = {}
        with open(path, 'r') as f:
            d = f.readlines()
            for line in d[1:]:
                els = line.split(',')
                key = els.pop(0)
                data[key] = list(map(float, els))
        # sorted_layers is an OrderedDict, and thus has the popitem method
        sorted_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        self.log.info('SORTED LAYERS: %s' % str(sorted_layers))
        first_layer = sorted_layers.popitem(last=False)
        # self.log.info('FIRST LAYER: %s'%str(first_layer))
        # An ordered dict is actually just a list of tuples so we can access
        # the key directly like so
        first_name = first_layer[0]
        # p_inc = data[first_name][0]
        # p_ref = np.abs(data[first_name][1])
        # p_trans = data[last_name][0]
        p_inc = np.sqrt(data[first_name][0]**2 + data[first_name][2]**2)
        p_ref = np.sqrt(data[first_name][1]**2 + data[first_name][3]**2)
        p_trans = np.sqrt(data[port][0]**2 + data[port][2]**2)
        reflectance = p_ref / p_inc
        transmission = p_trans / p_inc
        absorbance = 1 - reflectance - transmission
        tot = reflectance + transmission + absorbance
        delta = np.abs(tot - 1)
        self.log.info('Reflectance %f' % reflectance)
        self.log.info('Transmission %f' % transmission)
        self.log.info('Absorbance %f' % absorbance)
        self.log.info('Total = %f' % tot)
        assert(reflectance >= 0)
        assert(transmission >= 0)
        # assert(absorbance >= 0)
        assert(delta < .00001)
        outpath = os.path.join(base, 'ref_trans_abs.dat')
        self.log.info('Writing transmission file')
        if os.path.isfile(outpath):
            with open(outpath, 'a') as out:
                out.write('%s,%f,%f,%f\n' % (port, reflectance, transmission, absorbance))
        else:
            with open(outpath, 'w') as out:
                out.write('# Port, Reflectance,Transmission,Absorbance\n')
                out.write('%s,%f,%f,%f\n' % (port, reflectance, transmission, absorbance))
        return reflectance, transmission, absorbance

    def integrated_absorbtion(self, sim):
        """Computes the absorption of a layer by using the volume integral of the product of the
        imaginary part of the relative permittivity and the norm squared of the E field"""
        raise NotImplementedError('There are some bugs in S4 and other reasons'
                                  ' that this function doesnt work yet')
        base = sim.conf['General']['sim_dir']
        path = os.path.join(base, 'integrated_absorption.dat')
        inpath = os.path.join(base, 'energy_densities.dat')
        freq = sim.conf['Simulation']['params']['frequency']['value']
        # TODO: Assuming incident amplitude and therefore incident power is
        # just 1 for now
        fact = -.5 * freq * c.epsilon_0
        with open(inpath, 'r') as inf:
            lines = inf.readlines()
            # Remove header line
            lines.pop(0)
            # Dict where key is layer name and value is list of length 2 containing real and
            # imaginary parts of energy density integral
            data = {line.strip().split(',')[0]: line.strip().split(',')[
                1:] for line in lines}
        self.log.info('Energy densities: %s' % str(data))
        with open(path, 'w') as outf:
            outf.write('# Layer, Absorption\n')
            for layer, vals in data.items():
                absorb = fact * float(vals[1])
                outf.write('%s,%s\n' % (layer, absorb))


class Global_Cruncher(Cruncher):
    """Computes global quantities for an entire run, instead of local quantities for an individual
    simulation"""

    def __init__(self, global_conf, sims=[], sim_groups=[], failed_sims=[]):
        super(Global_Cruncher, self).__init__(
            global_conf, sims, sim_groups, failed_sims)
        self.log.debug('This is the global cruncher')

    def calculate(self, quantity, args):
        try:
            getattr(self, quantity)(*args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity, exc_info=True, stack_info=True)
            raise

    def process_all(self):
        # For each quantity
        self.log.info('Beginning global cruncher processing ...')
        for quant, data in self.gconf['Postprocessing']['Global_Cruncher'].items():
            if data['compute']:
                argsets = data['args']
                self.log.info('Computing %s with args %s',
                              str(quant), str(argsets))
                if argsets and type(argsets[0]) == list:
                    for argset in argsets:
                        self.log.info('Computing individual argset'
                                      ' %s', str(argset))
                        if argset:
                            self.calculate(quant, argset)
                        else:
                            self.calculate(quant, [])
                else:
                    if argsets:
                        self.calculate(quant, argsets)
                    else:
                        self.calculate(quant, [])

    def diff_sq(self, x, y):
        """Returns the magnitude of the difference vector squared between two vector fields at each
        point in space"""
        # Calculate the magnitude of the difference vector SQUARED at each point in space
        # This is mag(vec(x) - vec(y))^2 at each point in space. This should be a 1D array
        # with # of elements = # sampling points
        mag_diff_vec = sum([np.absolute(v1 - v2)**2 for v1, v2 in zip(x, y)])
        return mag_diff_vec

    def get_slice(self, sim):
        """Returns indices for data that strip out air and substrate regions"""
        # TODO: This function is definitely not general. We need to get a list
        # of layers to exclude from the user. For now, just assume we want to
        # exclude the top and bottom regions
        # sorted_layers is an OrderedDict, and thus has the popitem method
        sorted_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        first_layer = sorted_layers.popitem(last=False)
        print(first_layer)
        last_layer = sorted_layers.popitem()
        # We can get the starting and ending planes from their heights
        start_plane = int(round(first_layer[1]['params'][
                          'thickness']['value'] / sim.dz))
        end_plane = int(round(last_layer[1]['params'][
                        'thickness']['value'] / sim.dz))
        return start_plane, end_plane

    def get_comp_vec(self, sim, field, start, end):
        """Returns the comparison vector"""
        # Compare all other sims to our best estimate, which is sim with highest number of
        # basis terms (last in list cuz sorting)

        # Get the proper file extension depending on the field.
        norm = 'norm'+field
        # Get the comparison vector
        vecs = [sim.data[field+comp][start:end] for comp in ('x', 'y', 'z')]
        normvec = sim.get_scalar_quantity('normE')
        normvec = normvec[start:end]**2
        return vecs, normvec

    def local_error(self, field, exclude=False):
        """Computes the average of the local error between the vector fields of two simulations at
        each point in space"""
        self.log.info(
            'Running the local error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        for group in self.sim_groups:
            if exclude:
                start, end = self.get_slice(group[0])
                excluded = '_excluded'
            else:
                start = 0
                end = None
                excluded = ''
            base = group[0].conf['General']['results_dir']
            errpath = os.path.join(
                base, 'localerror_%s%s.dat' % (field, excluded))
            with open(errpath, 'w') as errfile:
                self.log.info('Computing local error for sweep %s', base)
                # Set the reference sim
                ref_sim = group[-1]
                ref_sim.get_data()
                # Get the comparison vector
                vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                # For all other sims in the groups, compare to best estimate
                # and write to error file
                for i in range(0, len(group) - 1):
                    sim2 = group[i]
                    sim2.get_data()
                    vecs2, normvec2 = self.get_comp_vec(sim2, field, start,
                                                        end)
                    self.log.info("Computing local error between numbasis %i and numbasis %i",
                                  ref_sim.conf['Simulation'][
                                      'params']['numbasis']['value'],
                                  sim2.conf['Simulation']['params']['numbasis']['value'])
                    # Get the array containing the magnitude of the difference vector at each point
                    # in space
                    mag_diff_vec = self.diff_sq(vecs1, vecs2)
                    # Normalize the magnitude squared of the difference vector by the magnitude squared of
                    # the local electric field of the comparison simulation at
                    # each point in space
                    if len(mag_diff_vec) != len(normvec):
                        self.log.error(
                            "The normalization vector has an incorrect number of elements!!!")
                        quit()
                    norm_mag_diff = mag_diff_vec / normvec
                    # Compute the average of the normalized magnitude of all
                    # the difference vectors
                    avg_diffvec_mag = np.sum(
                        norm_mag_diff) / norm_mag_diff.size
                    self.log.info(str(avg_diffvec_mag))
                    errfile.write('%i,%f\n' % (sim2.conf['Simulation']['params'][
                                  'numbasis']['value'], avg_diffvec_mag))
                    sim2.clear_data()
                ref_sim.clear_data()

    def global_error(self, field, exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell"""

        self.log.info(
            'Running the global error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        for group in self.sim_groups:
            if exclude:
                start, end = self.get_slice(group[0])
                excluded = '_excluded'
            else:
                start = 0
                end = None
                excluded = ''
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            errpath = os.path.join(
                base, 'globalerror_%s%s.dat' % (field, excluded))
            with open(errpath, 'w') as errfile:
                self.log.info('Computing global error for sweep %s', base)
                # Set reference sim
                ref_sim = group[-1]
                ref_sim.get_data()
                # Get the comparison vector
                vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                # For all other sims in the groups, compare to best estimate
                # and write to error file
                for i in range(0, len(group) - 1):
                    sim2 = group[i]
                    sim2.get_data()
                    vecs2, normvec2 = self.get_comp_vec(sim2, field, start,
                                                        end)
                    self.log.info("Computing global error between numbasis %i and numbasis %i",
                                  ref_sim.conf['Simulation'][
                                      'params']['numbasis']['value'],
                                  sim2.conf['Simulation']['params']['numbasis']['value'])
                    # Get the array containing the magnitude of the difference vector at each point
                    # in space
                    mag_diff_vec = self.diff_sq(vecs1, vecs2)
                    # Check for equal lengths between norm array and diff mag
                    # array
                    if len(mag_diff_vec) != len(normvec):
                        self.log.error(
                            "The normalization vector has an incorrect number of elements!!!")
                        quit()
                    # Error as a percentage should be the square root of the ratio of sum of mag diff vec
                    # squared to mag efield squared
                    error = np.sqrt(np.sum(mag_diff_vec) / np.sum(normvec))
                    self.log.info(str(error))
                    errfile.write('%i,%f\n' % (sim2.conf['Simulation'][
                                  'params']['numbasis']['value'], error))
                    sim2.clear_data()
                ref_sim.clear_data()

    def adjacent_error(self, field, exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell.
        This computes error between adjacent sims in a sweep through basis terms."""

        self.log.info(
            'Running the global error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        for group in self.sim_groups:
            if exclude:
                start, end = self.get_slice(group[0])
                excluded = '_excluded'
            else:
                start = 0
                end = None
                excluded = ''
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            errpath = os.path.join(
                base, 'adjacenterror_%s%s.dat' % (field, excluded))
            with open(errpath, 'w') as errfile:
                self.log.info('Computing adjacent error for sweep %s', base)
                # For all other sims in the groups, compare to best estimate
                # and write to error file
                for i in range(1, len(group)):
                    # Set reference sim
                    ref_sim = group[i]
                    ref_sim.get_data()
                    # Get the comparison vector
                    vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                    sim2 = group[i - 1]
                    sim2.get_data()
                    vecs2, normvec2 = self.get_comp_vec(sim2, field, start,
                                                        end)
                    self.log.info("Computing adjacent error between numbasis %i and numbasis %i",
                                  ref_sim.conf['Simulation'][
                                      'params']['numbasis']['value'],
                                  sim2.conf['Simulation']['params']['numbasis']['value'])
                    # Get the array containing the magnitude of the difference vector at each point
                    # in space
                    mag_diff_vec = self.diff_sq(vecs1, vecs2)
                    # Check for equal lengths between norm array and diff mag
                    # array
                    if len(mag_diff_vec) != len(normvec):
                        self.log.error(
                            "The normalization vector has an incorrect number of elements!!!")
                        quit()
                    # Error as a percentage should be thkkk square root of the ratio of sum of mag diff vec
                    # squared to mag efield squared
                    error = np.sqrt(np.sum(mag_diff_vec) / np.sum(normvec))
                    self.log.info(str(error))
                    errfile.write('%i,%f\n' % (sim2.conf['Simulation'][
                                  'params']['numbasis']['value'], error))
                    sim2.clear_data()
                    ref_sim.clear_data()

    def scalar_reduce(self, quantity, avg=False):
        """Combine a scalar quantity across all simulations in each group. If
        avg=False then a direct sum is computed, otherwise an average is
        computed"""
        for group in self.sim_groups:
            base = group[0].conf['General']['results_dir']
            self.log.info('Performing scalar reduction for group at %s' % base)
            group[0].get_data()
            group_comb = group[0].get_scalar_quantity(quantity)
            group[0].clear_data()
            # This approach is more memory efficient then building a 2D array
            # of all the data from each group and summing along an axis
            for i in range(1, len(group)):
                group[i].get_data()
                group_comb += group[i].get_scalar_quantity(quantity)
                group[i].clear_data()
            if avg:
                group_comb = group_comb / len(group)
                fname = 'scalar_reduce_avg_%s' % quantity
            else:
                fname = 'scalar_reduce_%s' % quantity

            path = os.path.join(base, fname)
            if group[0].conf['General']['save_as'] == 'npz':
                np.save(path, group_comb)
            else:
                raise ValueError('Invalid file type in config')

    def fractional_absorbtion(self, port='Substrate'):
        """Computes the fraction of the incident spectrum that is absorbed by
        the device. This is a unitless number, and its interpretation somewhat
        depends on the units you express the incident spectrum in. If you
        expressed your incident spectrum in photon number, this can be
        interpreted as the fraction of incident photons that were absorbed. If
        you expressed your incident spectrum in terms of power per unit area,
        then this can be interpreted as the fraction of incident power per unit
        area that gets absorbed. In summary, its the fraction of whatever you
        put in that is being absorbed by the device."""
        valuelist = []
        for group in self.sim_groups:
            base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info('Computing fractional absorbtion for group at %s' % base)
            vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Assuming the sims have been grouped by frequency, sum over all of
            # them
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf['General'][
                                     'sim_dir'], 'ref_trans_abs.dat')
                # If we computed absorbtion for multiple regions, we have to
                # handle them properly here
                d = {}
                with open(dpath, 'r') as f:
                    lines = f.readlines()
                    lines.pop(0)
                for line in lines:
                    row = [s.strip() for s in line.split(',')]
                    row_port = row.pop(0)
                    data = list(map(float, row))
                    d[row_port] = data
                self.log.debug(d)
                # Unpack data for the port we passed in as an argument
                ref, trans, absorb = d[port]
                freq = sim.conf['Simulation']['params']['frequency']['value']
                wvlgth = c.c / freq
                wvlgth_nm = wvlgth * 1e9
                freqs[i] = freq
                wvlgths[i] = wvlgth
                # Get solar power from chosen spectrum
                path = sim.conf['Simulation']['input_power_wv']
                wv_vec, p_vec = np.loadtxt(path, usecols=(0, 2),
                                           unpack=True, delimiter=',')
                # Get p at wvlength by interpolation
                p_wv = interpolate.interp1d(wv_vec, p_vec, kind='linear',
                                            bounds_error=False, fill_value='extrapolate')
                sun_pow = p_wv(wvlgth_nm)
                spectra[i] = sun_pow * wvlgth_nm
                vals[i] = absorb * sun_pow * wvlgth_nm
            # Use Trapezoid rule to perform the integration. Note all the
            # necessary factors of the wavelength have already been included
            # above
            wvlgths = wvlgths[::-1]
            vals = vals[::-1]
            spectra = spectra[::-1]
            #Jsc = intg.simps(Jsc_vals,x=wvlgths,even='avg')
            integrated_absorbtion = intg.trapz(vals, x=wvlgths * 1e9)
            power = intg.trapz(spectra, x=wvlgths * 1e9)
            # factor of 1/10 to convert A*m^-2 to mA*cm^-2
            #wv_fact = c.e/(c.c*c.h*10)
            #wv_fact = .1
            #Jsc = (Jsc*wv_fact)/power
            frac_absorb = integrated_absorbtion / power
            outf = os.path.join(base, 'fractional_absorbtion.dat')
            with open(outf, 'w') as out:
                out.write('%f\n' % frac_absorb)
            self.log.info('Fractional Absorbtion = %f' % frac_absorb)
            valuelist.append(frac_absorb)
        return valuelist

    def Jsc(self, port='Substrate'):
        """Computes photocurrent density. This is just the integrated
        absorption scaled by a unitful factor. Assuming perfect carrier
        collection, meaning every incident photon gets converted to 1 collected
        electron, this factor is q/(hbar*c) which converts to a current per
        unit area"""
        valuelist = []
        for group in self.sim_groups:
            base = group[0].conf['General']['results_dir']
            self.log.info(
                'Computing photocurrent density for group at %s' % base)
            vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Assuming the sims have been grouped by frequency, sum over all of
            # them
            for i, sim in enumerate(group):
                dpath = os.path.join(sim.conf['General'][
                                     'sim_dir'], 'ref_trans_abs.dat')
                d = {}
                with open(dpath, 'r') as f:
                    lines = f.readlines()
                    lines.pop(0)
                for line in lines:
                    row = [s.strip() for s in line.split(',')]
                    row_port = row.pop(0)
                    data = list(map(float, row))
                    d[row_port] = data
                self.log.debug(d)
                # Unpack data for the port we passed in as an argument
                ref, trans, absorb = d[port]
                freq = sim.conf['Simulation']['params']['frequency']['value']
                wvlgth = c.c / freq
                wvlgth_nm = wvlgth * 1e9
                freqs[i] = freq
                wvlgths[i] = wvlgth
                # Get solar power from chosen spectrum
                path = sim.conf['Simulation']['input_power_wv']
                wv_vec, p_vec = np.loadtxt(path, usecols=(0, 2),
                                           unpack=True, delimiter=',')
                # Get p at wvlength by interpolation
                p_wv = interpolate.interp1d(wv_vec, p_vec, kind='linear',
                                            bounds_error=False, fill_value='extrapolate')
                sun_pow = p_wv(wvlgth_nm)
                spectra[i] = sun_pow * wvlgth_nm
                # This is our integrand
                vals[i] = absorb * sun_pow * wvlgth_nm
            # Use Trapezoid rule to perform the integration. Note all the
            # necessary factors of the wavelength have already been included
            # above
            wvlgths = wvlgths[::-1]
            vals = vals[::-1]
            spectra = spectra[::-1]
            integrated_absorbtion = intg.trapz(vals, x=wvlgths)
            # factor of 1/10 to convert A*m^-2 to mA*cm^-2
            wv_fact = c.e / (c.c * c.h * 10)
            Jsc = wv_fact * integrated_absorbtion
            outf = os.path.join(base, 'jsc.dat')
            with open(outf, 'w') as out:
                out.write('%f\n' % Jsc)
            self.log.info('Jsc = %f' % Jsc)
            valuelist.append(Jsc)
        return valuelist

    def weighted_transmissionData(self):
        """Computes spectrally weighted absorption,transmission, and reflection"""
        for group in self.sim_groups:
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info(
                'Computing spectrally weighted transmission data for group at %s' % base)
            abs_vals = np.zeros(len(group))
            ref_vals = np.zeros(len(group))
            trans_vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Get solar power from chosen spectrum
            path = group[0].conf['Simulation']['input_power_wv']
            wv_vec, p_vec = np.loadtxt(path, usecols=(0, 2),
                                       unpack=True, delimiter=',')
            # Get interpolating function for power
            p_wv = interpolate.interp1d(wv_vec, p_vec, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
            # Assuming the leaves contain frequency values, sum over all of
            # them
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf['General'][
                                     'sim_dir'], 'ref_trans_abs.dat')
                with open(dpath, 'r') as f:
                    ref, trans, absorb = list(
                        map(float, f.readlines()[1].split(',')))
                freq = sim.conf['Simulation']['params']['frequency']['value']
                wvlgth = c.c / freq
                wvlgth_nm = wvlgth * 1e9
                freqs[i] = freq
                wvlgths[i] = wvlgth_nm
                sun_pow = p_wv(wvlgth_nm)
                spectra[i] = sun_pow
                abs_vals[i] = sun_pow * absorb
                ref_vals[i] = sun_pow * ref
                trans_vals[i] = sun_pow * trans
            # Now integrate all the weighted spectra and divide by the power of
            # the spectra
            wvlgths = wvlgths[::-1]
            abs_vals = abs_vals[::-1]
            ref_vals = ref_vals[::-1]
            trans_vals = trans_vals[::-1]
            spectra = spectra[::-1]
            power = intg.trapz(spectra, x=wvlgths)
            wght_ref = intg.trapz(ref_vals, x=wvlgths) / power
            wght_abs = intg.trapz(abs_vals, x=wvlgths) / power
            wght_trans = intg.trapz(trans_vals, x=wvlgths) / power
            out = os.path.join(base, 'weighted_transmission_data.dat')
            with open(out, 'w') as outf:
                outf.write('# Reflection, Transmission, Absorbtion\n')
                outf.write('%f,%f,%f' % (wght_ref, wght_trans, wght_abs))
        return wght_ref, wght_trans, wght_abs


class Plotter(Processor):
    """Plots all the things listed in the config file"""

    def __init__(self, global_conf, sims=[], sim_groups=[], failed_sims=[]):
        super(Plotter, self).__init__(
            global_conf, sims, sim_groups, failed_sims)
        self.log.debug("This is the plotter")

    def process(self, sim):
        sim.get_data()
        sim_path = os.path.basename(sim.conf['General']['sim_dir'])
        self.log.info('Plotting data for sim %s', sim_path)
        # For each plot
        for plot, data in self.gconf['Postprocessing']['Plotter'].items():
            if data['compute']:
                argsets = data['args']
                self.log.info('Plotting %s with args %s',
                              str(plot), str(argsets))
                if argsets and type(argsets[0]) == list:
                    for argset in argsets:
                        self.log.info('Plotting individual argset'
                                      ' %s', str(argset))
                        if argset:
                            self.gen_plot(plot, sim, argset)
                        else:
                            self.gen_plot(plot, sim, [])
                else:
                    if argsets:
                        self.gen_plot(plot, sim, argsets)
                    else:
                        self.gen_plot(plot, sim, [])
        sim.clear_data()

    def process_all(self):
        self.log.info("Beginning local plotter method ...")
        for sim in self.sims:
            self.process(sim)

    def gen_plot(self, plot, sim, args):
        try:
            getattr(self, plot)(sim, *args)
        except KeyError:
            #  self.log.error("Unable to plot the following quantity: %s",
            #                 plot,exc_info=True,stack_info=True)
            self.log.error("Unable to plot the following quantity: %s",
                           plot, exc_info=True)
            raise

    def _draw_layer_circle(self, ldata, shape_key, start, end, sim, plane, pval, ax_hand):
        """Draws the circle within a layer"""
        shape_data = ldata['geometry'][shape_key]
        center = shape_data['center']
        radius = shape_data['radius']
        if plane == 'xy':
            circle = mpatches.Circle((center['x'], center['y']), radius=radius,
                                     fill=False)
            ax_hand.add_artist(circle)
        if plane in ["xz", "zx", "yz", "zy"]:
            plane_x = pval*sim.dx
            plane_to_center = np.abs(center['x'] - plane_x)
            self.log.debug('DIST: {}'.format(plane_to_center))
            # Only draw if the observation plane actually intersects with the
            # circle
            if not plane_to_center >= radius:
                # Check if the plane intersects with the center of the circle
                if plane_to_center > 0:
                    intersect_angle = np.arccos(plane_to_center/radius)
                    self.log.debug('ANGLE: {}'.format(intersect_angle))
                    half_width = plane_to_center*np.tan(intersect_angle)
                else:
                    half_width = radius
                self.log.debug('HALF WIDTH: {}'.format(half_width))
                # Vertical lines should span height of the layer
                z = [sim.height - start * sim.dz, sim.height - end * sim.dz]
                # The upper edge
                x = [center['y'] + half_width, center['y'] + half_width]
                line = mlines.Line2D(x, z, linestyle='solid', linewidth=2.0,
                                     color='grey')
                ax_hand.add_line(line)
                # Now the lower edge
                x = [center['y'] - half_width, center['y'] - half_width]
                line = mlines.Line2D(x, z, linestyle='solid', linewidth=2.0,
                                     color='grey')
                ax_hand.add_line(line)
        return ax_hand

    def _draw_layer_geometry(self, ldata, start, end, sim, plane, pval, ax_hand):
        """Given a dictionary with the data containing the geometry for a
        layer, draw the internal geometry of the layer for a given plane type
        and plane value"""
        for shape, data in ldata['geometry'].items():
            if data['type'] == 'circle':
                ax = self._draw_layer_circle(ldata, shape, start, end, sim,
                                             plane, pval, ax_hand)
            else:
                self.log.warning('Drawing of shape {} not '
                                 'supported'.format(data['type']))
        return ax

    def draw_geometry_2d(self, sim, plane, pval, ax_hand):
        """This function draws the layer boundaries and in-plane geometry on 2D
        heatmaps"""
        # Get the layers in order
        ordered_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        period = sim.conf['Simulation']['params']['array_period']['value']
        # Loop through them
        boundaries = []
        count = 0
        for layer, ldata in ordered_layers.items():
            # If x or y, draw bottom edge and text label now. Layer geometry
            # is handled in its own function
            if plane in ["xz", "zx", "yz", "zy"]:
                # Get boundaries between layers and their starting and ending
                # indices
                layer_t = ldata['params']['thickness']['value']
                if count == 0:
                    start = 0
                    end = int(layer_t / sim.dz) + 1
                    boundaries.append((layer_t, start, end, layer))
                else:
                    prev_tup = boundaries[count - 1]
                    dist = prev_tup[0] + layer_t
                    start = prev_tup[2]
                    end = int(dist / sim.dz) + 1
                    boundaries.append((dist, start, end))
                if layer_t > 0:
                    x = [0, period]
                    y = [sim.height - start * sim.dz,
                         sim.height - start * sim.dz]
                    label_y = y[0] - 0.25
                    label_x = x[-1] - .01
                    plt.text(label_x, label_y, layer, ha='right',
                             family='sans-serif', size=16, color='grey')
                    line = mlines.Line2D(x, y, linestyle='solid', linewidth=2.0,
                                         color='grey')
                    ax_hand.add_line(line)
                    count += 1
            else:
                # If look at a fixed z pval, the start and end values are
                # nonsensical but we must pass a value in
                start, end = None, None
            # If we have some internal geometry for this layer, draw it
            if 'geometry' in ldata:
                ax = self._draw_layer_geometry(ldata, start, end, sim, plane, pval, ax_hand)
        return ax

    def heatmap2d(self, sim, x, y, cs, labels, ptype, pval, save_path=None,
                  show=False, draw=False, fixed=None, colorsMap='jet'):
        """A general utility method for plotting a 2D heat map"""
        cm = plt.get_cmap(colorsMap)
        if fixed:
            cNorm = matplotlib.colors.Normalize(
                vmin=np.amin(5.0), vmax=np.amax(100.0))
        else:
            cNorm = matplotlib.colors.Normalize(
                vmin=np.amin(cs), vmax=np.amax(cs))
            # cNorm = matplotlib.colors.LogNorm(vmin=np.amin(cs)+.001, vmax=np.amax(cs))
            # cNorm = matplotlib.colors.LogNorm(vmin=1e13, vmax=np.amax(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        #  ax.pcolormesh(x, y, cs,cmap=cm,norm=cNorm,alpha=.5,linewidth=0)
        #  ax.pcolor(x, y,
        #          cs,cmap=cm,norm=cNorm,alpha=.5,linewidth=0,edgecolors='none')
        ax.imshow(cs,cmap=cm,norm=cNorm,extent=[x.min(),x.max(),y.min(),y.max()],aspect='auto')
        # ax.imshow(cs, cmap=cm, norm=cNorm, extent=[x.min(), x.max(), y.min(), y.max()],
        #           aspect=.1)
        # ax_ins = zoomed_inset_axes(ax, 6, loc=1)
        # ax_ins.imshow(cs[75:100,:], extent=[x.min(), x.max(), .8, 1.4])
        # ax_ins.grid(False)

        # ax.matshow(cs,cmap=cm,norm=cNorm, aspect='auto')
        ax.grid(False)
        scalarMap.set_array(cs)
        # div = make_axes_locatable(ax)
        # zoom_ax = div.append_axes("right",size='100%', pad=.5)
        # zoom_ax.imshow(cs[75:100,:], extent=[x.min(), x.max(), .8, 1.4])
        # zoom_ax.grid(False)
        # cax = div.append_axes("right",size="100%",pad=.05)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[2])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        # start, end = ax.get_xlim()
        # ticks = np.arange(start, end, 0.1)
        # ax.xaxis.set_ticks(ticks)
        # ax.set_xlim((np.amin(x), np.amax(x)))
        # ax.set_ylim((np.amin(y), np.amax(y)))
        # start, end = ax.get_ylim()
        # # print('START: %f'%start)
        # # print('END: %f'%end)
        # ticks = np.arange(end, start - 0.2, -0.2)
        # ticks[-1] = 0
        # # ticks = np.arange(start,end,0.2)
        # # ticks = np.arange(start,end,-0.2)
        # # print('###### TICKS ######')
        # # print(ticks)
        # ax.yaxis.set_ticks(ticks)
        # ax.yaxis.set_ticklabels(list(reversed(ticks)))
        # fig.suptitle(labels[3])
        if draw:
            self.log.info('Beginning geometry drawing routines ...')
            ax = self.draw_geometry_2d(sim, ptype, pval, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def plane_2d(self, sim, quantity, plane, pval, draw=False, fixed=None):
        """Plots a heatmap of a fixed 2D plane"""
        pval = int(pval)
        x = np.arange(0, sim.period, sim.dx)
        y = np.arange(0, sim.period, sim.dy)
        z = np.arange(0, sim.height + sim.dz, sim.dz)
        # Maps planes to an integer for extracting data
        # plane_table = {'x': 0,'y': 1,'z':2}
        # Get the scalar values
        self.log.info('Retrieving scalar %s' % quantity)
        scalar = sim.get_scalar_quantity(quantity)
        self.log.info('DATA SHAPE: %s' % str(scalar.shape))
        # Filter out any undesired data that isn't on the planes
        #mat = np.column_stack((sim.pos_inds[:,0],sim.pos_inds[:,1],sim.pos_inds[:,2],scalar))
        #planes = np.array([row for row in mat if row[plane_table[plane]] == pval])
        #self.log.debug("Planes shape: %s"%str(planes.shape))
        # Get all unique values for x,y,z and convert them to actual values not indices
        #x,y,z = np.unique(planes[:,0])*dx,np.unique(planes[:,1])*dy,np.unique(planes[:,2])
        freq = sim.conf['Simulation']['params']['frequency']['value']
        wvlgth = (c.c / freq) * 1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(
            freq, wvlgth)
        # Get the plane we wish to plot
        self.log.info('Retrieving plane ...')
        cs = self.get_plane(scalar, plane, pval)
        self.log.info('Plotting plane')
        show = sim.conf['General']['show_plots']
        p = False
        if plane == 'yz' or plane == 'zy':
            labels = ('y [um]', 'z [um]', quantity, title)
            if sim.conf['General']['save_plots']:
                p = os.path.join(sim.conf['General']['sim_dir'],
                                 '%s_plane_2d_yz_pval%s.pdf' % (quantity,
                                                               str(pval)))
            self.heatmap2d(sim, y, z, cs, labels, plane, pval,
                           save_path=p, show=show, draw=draw, fixed=fixed)
        elif plane == 'xz' or plane == 'zx':
            labels = ('x [um]', 'z [um]', quantity, title)
            if sim.conf['General']['save_plots']:
                p = os.path.join(sim.conf['General']['sim_dir'],
                                 '%s_plane_2d_xz_pval%s.pdf' % (quantity,
                                                               str(pval)))
            self.heatmap2d(sim, x, z, cs, labels, plane, pval,
                           save_path=p, show=show, draw=draw, fixed=fixed)
        elif plane == 'xy' or plane == 'yx':
            labels = ('y [um]', 'x [um]', quantity, title)
            if sim.conf['General']['save_plots']:
                p = os.path.join(sim.conf['General']['sim_dir'],
                                 '%s_plane_2d_xy_pval%s.pdf' % (quantity,
                                                               str(pval)))
            self.heatmap2d(sim, x, y, cs, labels, plane, pval,
                           save_path=p, show=show, draw=draw, fixed=fixed)

    def scatter3d(self, sim, x, y, z, cs, labels, ptype, colorsMap='jet'):
        """A general utility method for scatter plots in 3D"""
        #fig = plt.figure(figsize=(8,6))
        #ax = fig.add_subplot(111,projection='3d')
        #colors = cm.hsv(E_mag/max(E_mag))
        #colmap = c.ScalarMappable(cmap=cm.hsv)
        # colmap.set_array(E_mag)
        #yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
        #cb = fig.colorbar(colmap)
        # print("X SHAPE: %s"%str(x.shape))
        # print("Y SHAPE: %s"%str(y.shape))
        # print("Z SHAPE: %s"%str(z.shape))
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9, 7))

        #ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), edgecolor='none')
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap)
        cb.set_label(labels[3])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        fig.suptitle(os.path.basename(sim.conf['General']['sim_dir']))
        if sim.conf['General']['save_plots']:
            name = labels[-1] + '_' + ptype + '.pdf'
            path = os.path.join(sim.conf['General']['sim_dir'], name)
            fig.savefig(path)
        if sim.conf['General']['show_plots']:
            plt.show()
        plt.close(fig)

    def full_3d(self, sim, quantity):
        """Generates a full 3D plot of a specified scalar quantity"""
        # The data just tells you what integer grid point you are on. Not what actual x,y coordinate you
        # are at
        x = np.arange(0, sim.period, sim.dx)
        y = np.arange(0, sim.period, sim.dy)
        z = np.arange(0, sim.height + sim.dz, sim.dz)
        points = np.array(list(itertools.product(z, x, y)))
        # Get the scalar
        scalar = sim.get_scalar_quantity(quantity)
        labels = ('X [um]', 'Y [um]', 'Z [um]', quantity)
        # Now plot!
        self.scatter3d(sim, points[:, 1], points[:, 2], points[
                       :, 0], scalar.flatten(), labels, 'full_3d')

    def planes_3d(self, sim, quantity, xplane, yplane):
        """Plots some scalar quantity in 3D but only along specified x-z and y-z planes"""
        xplane = int(xplane)
        yplane = int(yplane)
        # Get the scalar values
        scalar = sim.get_scalar_quantity(quantity)
        x = np.arange(0, sim.period, sim.dx)
        y = np.arange(0, sim.period, sim.dy)
        z = np.arange(0, sim.height + sim.dz, sim.dz)
        # Get the data on the plane with a fixed x value. These means we'll
        # have changing (y, z) points
        xdata = self.get_plane(scalar, 'yz', xplane)
        # z first cuz we want y to be changing before z to correspond with the
        # way numpy flattens arrays. Note this means y points will be in the
        # 2nd column
        xplanepoints = np.array(list(itertools.product(z, y)))
        xdata = xdata.flatten()
        xplanexval = np.array(list(itertools.repeat(x[xplane], len(xdata))))
        xplanedata = np.zeros((xplanepoints.shape[0], 4))
        xplanedata[:, 0] = xplanexval
        xplanedata[:, 1] = xplanepoints[:, 1]
        xplanedata[:, 2] = xplanepoints[:, 0]
        xplanedata[:, 3] = xdata
        # Same procedure for fixed y plane
        ydata = self.get_plane(scalar, 'xz', yplane)
        yplanepoints = np.array(list(itertools.product(z, x)))
        ydata = ydata.flatten()
        yplaneyval = np.array(list(itertools.repeat(y[yplane], len(ydata))))
        yplanedata = np.zeros((yplanepoints.shape[0], 4))
        yplanedata[:, 0] = yplanepoints[:, 1]
        yplanedata[:, 1] = yplaneyval
        yplanedata[:, 2] = yplanepoints[:, 0]
        yplanedata[:, 3] = ydata
        labels = ('X [um]', 'Y [um]', 'Z [um]', quantity)
        # Now stack them vertically and plot!
        all_data = np.vstack((xplanedata, yplanedata))
        self.scatter3d(sim, all_data[:, 0], all_data[:, 1], all_data[:, 2],
                       all_data[:, 3], labels, 'planes_3d')

    def line_plot(self, sim, x, y, ptype, labels):
        """Make a simple line plot"""
        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(labels[2])
        if sim.conf['General']['save_plots']:
            name = labels[1] + '_' + ptype + '.pdf'
            path = os.path.join(sim.conf['General']['sim_dir'], name)
            fig.savefig(path)
        if sim.conf['General']['show_plots']:
            plt.show()
        plt.close(fig)

    def fixed_line(self, sim, quantity, direction, coord1, coord2):
        """Plot a scalar quantity on a line along a the z direction at some pair of
        coordinates in the plane perpendicular to that direction"""
        coord1 = int(coord1)
        coord2 = int(coord2)
        # Get the scalar values
        scalar = sim.get_scalar_quantity(quantity)
        # Filter out any undesired data that isn't on the planes
        data = self.get_line(scalar, sim.x_samples, sim.y_samples,
                             sim.z_samples, direction, coord1, coord2)
        x = np.arange(0, sim.period, sim.dx)
        y = np.arange(0, sim.period, sim.dy)
        z = np.arange(0, sim.height + sim.dz, sim.dz)
        if direction == 'x':
            # z along rows, y along columns
            pos_data = x
        elif direction == 'y':
            # x along columns, z along rows
            pos_data = y
        elif direction == 'z':
            # x along rows, y along columns
            pos_data = z
        freq = sim.conf['Simulation']['params']['frequency']['value']
        wvlgth = (c.c / freq) * 1E9
        title = 'Frequency = {:.4E} Hz, Wavelength = {:.2f} nm'.format(
            freq, wvlgth)
        labels = ('Z [um]', quantity, title)
        ptype = "%s_line_plot_%i_%i" % (direction, coord1, coord2)
        self.line_plot(sim, pos_data, data, ptype, labels)


class Global_Plotter(Plotter):
    """Plots global quantities for an entire run that are not specific to a single simulation"""

    def __init__(self, global_conf, sims=[], sim_groups=[], failed_sims=[]):
        super(Global_Plotter, self).__init__(
            global_conf, sims, sim_groups, failed_sims)
        self.log.debug("Global plotter init")

    def process_all(self):
        self.log.info('Beginning global plotter method ...')
        for plot, data in self.gconf['Postprocessing']['Global_Plotter'].items():
            if data['compute']:
                argsets = data['args']
                self.log.info('Plotting %s with args %s',
                              str(plot), str(argsets))
                if argsets and type(argsets[0]) == list:
                    for argset in argsets:
                        self.log.info('Plotting individual argset'
                                      ' %s', str(argset))
                        if argset:
                            self.gen_plot(plot, argset)
                        else:
                            self.gen_plot(plot, [])
                else:
                    if argsets:
                        self.gen_plot(plot, argsets)
                    else:
                        self.gen_plot(plot, [])

    def gen_plot(self, plot, args):
        try:
            getattr(self, plot)(*args)
        except KeyError:
            self.log.error("Unable to plot the following quantity: %s",
                           plot, exc_info=True, stack_info=True)
            raise

    def convergence(self, quantity, err_type='global', scale='linear'):
        """Plots the convergence of a field across all available simulations"""
        self.log.info('Plotting convergence')
        for group in self.sim_groups:
            base = group[0].conf['General']['base_dir']
            if err_type == 'local':
                fglob = os.path.join(base, 'localerror_%s*.dat' % quantity)
            elif err_type == 'global':
                fglob = os.path.join(base, 'globalerror_%s*.dat' % quantity)
            elif err_type == 'adjacent':
                fglob = os.path.join(base, 'adjacenterror_%s*.dat' % quantity)
            else:
                self.log.error('Attempting to plot an unsupported error type')
                raise ValueError
            paths = glob.glob(fglob)
            for path in paths:
                labels = []
                errors = []
                with open(path, 'r') as datf:
                    for line in datf.readlines():
                        lab, err = line.split(',')
                        labels.append(lab)
                        errors.append(err)
                fig = plt.figure(figsize=(9, 7))
                plt.ylabel('M.S.E of %s' % quantity)
                plt.xlabel('Number of Fourier Terms')
                plt.plot(labels, errors)
                plt.yscale(scale)
                # plt.xticks(x,labels,rotation='vertical')
                plt.tight_layout()
                plt.title(os.path.basename(base))
                if self.gconf['General']['save_plots']:
                    if '_excluded' in path:
                        excluded = '_excluded'
                    else:
                        excluded = ''
                    name = '%s_%sconvergence_%s%s.pdf' % (
                        os.path.basename(base), err_type, quantity, excluded)
                    path = os.path.join(base, name)
                    fig.savefig(path)
                if self.gconf['General']['show_plots']:
                    plt.show()
                plt.close(fig)

    def scalar_reduce(self, quantity, plane, pval, draw=False, fixed=None):
        """Plot the result of a particular scalar reduction for each group"""
        for group in self.sim_groups:
            sim = group[0]
            base = sim.conf['General']['results_dir']
            self.log.info('Plotting scalar reduction of %s for quantity'
                          ' %s' % (base, quantity))
            cm = plt.get_cmap('jet')
            max_depth = sim.conf['Simulation']['max_depth']
            period = sim.conf['Simulation']['params']['array_period']['value']
            x = np.arange(0, period, sim.dx)
            y = np.arange(0, period, sim.dy)
            z = np.arange(0, max_depth + sim.dz, sim.dz)
            if sim.conf['General']['save_as'] == 'npz':
                globstr = os.path.join(
                    base, 'scalar_reduce*_%s.npy' % quantity)
                files = glob.glob(globstr)
            else:
                raise ValueError('Incorrect file type in config')
            title = 'Reduction of %s' % quantity
            for datfile in files:
                p = False
                if sim.conf['General']['save_as'] == 'npz':
                    scalar = np.load(datfile)
                else:
                    raise ValueError('Incorrect file type in config')
                cs = self.get_plane(scalar, plane, pval)
                if plane == 'yz' or plane == 'zy':
                    labels = ('y [um]', 'z [um]', quantity, title)
                    if sim.conf['General']['save_plots']:
                        fname = 'scalar_reduce_%s_plane_2d_yz.pdf' % quantity
                        p = os.path.join(base, fname)
                    show = sim.conf['General']['show_plots']
                    self.heatmap2d(sim, y, z, cs, labels, plane, pval,
                                   save_path=p, show=show, draw=draw, fixed=fixed)
                elif plane == 'xz' or plane == 'zx':
                    labels = ('x [um]', 'z [um]', quantity, title)
                    if sim.conf['General']['save_plots']:
                        fname = 'scalar_reduce_%s_plane_2d_xz.pdf' % quantity
                        p = os.path.join(base, fname)
                    show = sim.conf['General']['show_plots']
                    self.heatmap2d(sim, x, z, cs, labels, plane, pval,
                                   save_path=p, show=show, draw=draw, fixed=fixed)
                elif plane == 'xy' or plane == 'yx':
                    labels = ('y [um]', 'x [um]', quantity, title)
                    if sim.conf['General']['save_plots']:
                        fname = 'scalar_reduce_%s_plane_2d_xy.pdf' % quantity
                        p = os.path.join(base, fname)
                    self.heatmap2d(sim, x, y, cs, labels, plane, pval,
                                   save_path=p, show=show, draw=draw, fixed=fixed)

    def transmission_data(self, absorbance, reflectance, transmission,
                          port='Substrate'):
        """Plot transmissions, absorption, and reflectance assuming leaves are frequency"""
        for group in self.sim_groups:
            # base = group[0].conf['General']['base_dir']
            base = group[0].conf['General']['results_dir']
            self.log.info('Plotting transmission data for group at %s' % base)
            # Assuming the leaves contain frequency values, sum over all of
            # them
            freqs = np.zeros(len(group))
            refl_l = np.zeros(len(group))
            trans_l = np.zeros(len(group))
            absorb_l = np.zeros(len(group))
            for i in range(len(group)):
                sim = group[i]
                dpath = os.path.join(sim.conf['General'][
                                     'sim_dir'], 'ref_trans_abs.dat')
                d = {}
                with open(dpath, 'r') as f:
                    lines = f.readlines()
                    lines.pop(0)
                for line in lines:
                    row = [s.strip() for s in line.split(',')]
                    row_port = row.pop(0)
                    data = list(map(float, row))
                    d[row_port] = data
                # Unpack data for the port we passed in as an argument
                ref, trans, absorb = d[port]
                freq = sim.conf['Simulation']['params']['frequency']['value']
                freqs[i] = freq
                trans_l[i] = trans
                refl_l[i] = ref
                absorb_l[i] = absorb
            freqs = (c.c / freqs[::-1]) * 1e9
            refl_l = refl_l[::-1]
            absorb_l = absorb_l[::-1]
            trans_l = trans_l[::-1]
            plt.figure()
            if absorbance:
                self.log.info('Plotting absorbance')
                plt.plot(freqs, absorb_l, '-o', label='Absorption')
            if reflectance:
                plt.plot(freqs, refl_l, '-o', label='Reflection')
            if transmission:
                plt.plot(freqs, trans_l, '-o', label='Transmission')
            plt.legend(loc='best')
            figp = os.path.join(base, 'transmission_plots_port%s.pdf'%port)
            plt.xlabel('Wavelength (nm)')
            plt.ylim((0, 1.0))
            plt.savefig(figp)
            plt.close()


def main():
    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file', type=str, help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('-nc', '--no_crunch', action="store_true", default=False, help="""Do not perform crunching
            operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-ngc', '--no_gcrunch', action="store_true", default=False, help="""Do not
            perform global crunching operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-np', '--no_plot', action="store_true", default=False, help="""Do not perform plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('-ngp', '--no_gplot', action="store_true", default=False, help="""Do not perform global plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('--log_level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="""Logging level for the run""")
    parser.add_argument('--filter_by', nargs='*', help="""List of parameters you wish to filter by,
            specified like: p1:v1,v2,v3 p2:v1,v2,v3""")
    parser.add_argument('-gb', '--group_by', help="""The parameter you
            would like to group simulations by, specified as a dot separated path
            to the key in the config as: path.to.key.value""")
    parser.add_argument('-ga', '--group_against', help="""The parameter
            you would like to group against, specified as a dot separated path
            to the key in the config as: path.to.key.value""")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = Config(path=os.path.abspath(args.config_file))
        conf.expand_vars()
    else:
        raise ValueError("The file you specified does not exist!")

    if not (args.group_by or args.group_against):
        raise ValueError('Need to group sims somehow. A sensible value would'
                         ' be by/against frequency')
    else:
        if args.group_by:
            group_by = args.group_by.split('.')
        else:
            group_ag = args.group_against.split('.')

    # Configure logger
    lfile = os.path.join(conf['General']['base_dir'], 'logs/postprocess.log')
    logger = configure_logger(level=args.log_level, name='postprocess',
                              console=True, logfile=lfile)
    # Configure plotting style
    try:
        plt.style.use(conf['Postprocessing']['style'])
    except KeyError:
        plt.style.use('ggplot')
    # Collect the sims once up here and reuse them later
    proc = Processor(conf)
    print('Collecting sims')
    sims, failed_sims = proc.collect_sims()
    # First we need to group against if specified. Grouping against corresponds
    # to "leaves" in the tree
    if args.group_against:
        sim_groups = proc.group_against(group_ag, conf.variable)
    # Next we group by. This corresponds to building the parent nodes for each
    # set of leaf groups
    if args.group_by:
        sim_groups = proc.group_by(group_by)
    # print(len(sim_groups))
    # print(group_ag)
    # print(conf.variable)
    # for group in proc.sim_groups:
    #     sim = group[0]
    #     try:
    #         os.makedirs(sim.conf['General']['results_dir'])
    #     except OSError:
    #         pass
    # quit()
    # Filter if specified
    if args.filter_by:
        filt_dict = {}
        for item in args.filter_by:
            par, vals = item.split(':')
            vals = vals.split(',')
            filt_dict[par] = vals
        logger.info('Here is the filter dictionary: %s' % filt_dict)
        sims, sim_groups = proc.filter_by_param(filt_dict)
    # Now do all the work
    if not args.no_crunch:
        crunchr = Cruncher(conf, sims, sim_groups, failed_sims)
        crunchr.process_all()
        # for sim in crunchr.sims:
        #     crunchr.transmissionData(sim)
    if not args.no_gcrunch:
        gcrunchr = Global_Cruncher(conf, sims, sim_groups, failed_sims)
        gcrunchr.process_all()
    if not args.no_plot:
        pltr = Plotter(conf, sims, sim_groups, failed_sims)
        pltr.process_all()
    if not args.no_gplot:
        gpltr = Global_Plotter(conf, sims, sim_groups, failed_sims)
        # gpltr.collect_sims()
        gpltr.process_all()


if __name__ == '__main__':
    main()
