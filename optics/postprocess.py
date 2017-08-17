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
import tables as tb

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
        if self.gconf['General']['save_as'] == 'hdf5':
            datfile = os.path.join(self.gconf['General']['base_dir'],
                                   'data.hdf5')
            self.hdf5 = tb.open_file(datfile, 'r+')
        else:
            self.hdf5 = None
        self.sims = sims
        self.sim_groups = sim_groups
        # A place to store any failed sims (i.e sims that are missing their
        # data file)
        self.failed_sims = failed_sims

    def __del__(self):
        """
        Close HDF5 file
        """
        if self.hdf5 is not None:
            self.hdf5.close()

    def collect_sims(self):
        """Collect all the simulations beneath the base of the directory tree"""
        sims = []
        failed_sims = []
        ftype = self.gconf['General']['save_as']
        if ftype == 'npz':
            datfile = self.gconf['General']['base_name'] + '.npz'
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
        elif ftype == 'hdf5':
            sim_groups = [group for group in self.hdf5.iter_nodes('/', classname="Group")
                          if 'sim_' in group._v_name]

            for group in sim_groups:
                if 'conf' in group._v_attrs:
                    conf_str = group._v_attrs['conf']
                    sim_obj = Simulation(Config(raw_text=conf_str))
                    sim_obj.conf.expand_vars()
                    sims.append(sim_obj)
                else:
                    self.log.critical('The following sim is missing its config file: %s',
                                      group._v_name)
                    raise RuntimeError
        else:
            raise ValueError('Invalid file type specified in config')
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
            result = getattr(sim, quantity)(*args)
        except KeyError:
            self.log.error("Unable to calculate the following quantity: %s",
                           quantity, exc_info=True, stack_info=True)
            raise

    def process(self, sim):
        sim_path = os.path.basename(sim.conf['General']['sim_dir'])
        self.log.info('Crunching data for sim %s', sim_path)
        # sim.get_data()
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
                # ref_sim.get_data()
                # Get the comparison vector
                vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                # For all other sims in the groups, compare to best estimate
                # and write to error file
                for i in range(0, len(group) - 1):
                    sim2 = group[i]
                    # sim2.get_data()
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
                    # self.log.info(str(avg_diffvec_mag))
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
                # ref_sim.get_data()
                # Get the comparison vector
                vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                # For all other sims in the groups, compare to best estimate
                # and write to error file
                for i in range(0, len(group) - 1):
                    sim2 = group[i]
                    # sim2.get_data()
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
                    # self.log.info(str(error))
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
                    # ref_sim.get_data()
                    # Get the comparison vector
                    vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                    sim2 = group[i - 1]
                    # sim2.get_data()
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
                    # self.log.info(str(error))
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
            # group[0].get_data()
            self.log.debug('QUANTITY: %s'%quantity)
            group_comb = group[0].get_scalar_quantity(quantity)
            self.log.debug(group_comb.dtype)
            group[0].clear_data()
            # This approach is more memory efficient then building a 2D array
            # of all the data from each group and summing along an axis
            for sim in group[1:]:
                # sim.get_data()
                self.log.debug(sim.id)
                quant = sim.get_scalar_quantity(quantity)
                self.log.debug(quant.dtype)
                group_comb += quant
                sim.clear_data()
            if avg:
                group_comb = group_comb / len(group)
                fname = 'scalar_reduce_avg_%s' % quantity
            else:
                fname = 'scalar_reduce_%s' % quantity

            path = os.path.join(base, fname)
            ftype = group[0].conf['General']['save_as']
            if ftype == 'npz':
                np.save(path, group_comb)
            elif ftype == 'hdf5':
                self.log.warning('FIX HDF5 SCALAR REDUCE SAVING')
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
            base = group[0].conf['General']['results_dir']
            self.log.info('Computing fractional absorbtion for group at %s' % base)
            vals = np.zeros(len(group))
            freqs = np.zeros(len(group))
            wvlgths = np.zeros(len(group))
            spectra = np.zeros(len(group))
            # Assuming the sims have been grouped by frequency, sum over all of
            # them
            for i, sim in enumerate(group):
                # sim.get_data()
                # Unpack data for the port we passed in as an argument
                ref, trans, absorb = sim.data['transmission_data'][port]
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
                sim.clear_data()
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
                # sim.get_data()
                # Unpack data for the port we passed in as an argument
                ref, trans, absorb = sim.data['transmission_data'][port]
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
                sim.clear_data()
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
            self.log.info('Jsc = %f', Jsc)
            valuelist.append(Jsc)
        return valuelist

    def Jsc_integrated(self):
        """
        Compute te photocurrent density by performing a volume integral of the
        generation rate
        """
        fname = 'scalar_reduce_genRate.npy'
        valueList = np.zeros(len(self.sim_groups))
        for i, group in enumerate(self.sim_groups):
            base = group[0].conf['General']['results_dir']
            self.log.info('Computing integrated Jsc for group at %s', base)
            path = os.path.join(base, fname)
            try:
                genRate = np.load(path)
            except FileNotFoundError:
                self.scalar_reduce('genRate')
                genRate = np.load(path)
            # Gen rate in cm^-3. Gotta convert lengths here from um to cm
            z_vals = np.linspace(0, group[0].height*1e-4, group[0].z_samples)
            x_vals = np.linspace(0, group[0].height*1e-4, group[0].x_samples)
            y_vals = np.linspace(0, group[0].height*1e-4, group[0].y_samples)
            z_integral = intg.trapz(genRate, x=z_vals, axis=0)
            x_integral = intg.trapz(z_integral, x=x_vals, axis=0)
            y_integral = intg.trapz(x_integral, x=y_vals, axis=0)
            # Convert period to cm
            Jsc = (c.e/(group[0].period*1e-4)**2)*y_integral
            outf = os.path.join(base, 'jsc_integrated.dat')
            with open(outf, 'w') as out:
                out.write('%f\n' % Jsc)
            valueList[i] = Jsc
            self.log.info('Jsc_integrated = %f', Jsc)
        return valueList

    def weighted_transmissionData(self, port='Substrate'):
        """Computes spectrally weighted absorption,transmission, and reflection"""
        for group in self.sim_groups:
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
            for i, sim in enumerate(group):
                ref, trans, absorb = sim.data['transmission_data'][port]
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
        # sim.get_data()
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
        ax.imshow(cs,cmap=cm,norm=cNorm,extent=[x.min(),x.max(),y.min(),y.max()],aspect='auto')
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
        # Get the scalar values
        self.log.info('Retrieving scalar %s' % quantity)
        scalar = sim.get_scalar_quantity(quantity)
        self.log.info('DATA SHAPE: %s' % str(scalar.shape))
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
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(figsize=(9, 7))

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
            ftype = sim.conf['General']['save_as']
            if ftype == 'npz':
                globstr = os.path.join(
                    base, 'scalar_reduce*_%s.npy' % quantity)
                files = glob.glob(globstr)
            elif ftype == 'hdf5':
                self.log.warning('FIX LOAD IN GLOBAL SCALAR REDUCE')
                globstr = os.path.join(
                    base, 'scalar_reduce*_%s.npy' % quantity)
                files = glob.glob(globstr)
            else:
                raise ValueError('Incorrect file type in config')
            title = 'Reduction of %s' % quantity
            for datfile in files:
                p = False
                if ftype == 'npz':
                    scalar = np.load(datfile)
                elif ftype == 'hdf5':
                    self.log.warning('FIX LOAD IN GLOBAL SCALAR REDUCE')
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
            for i, sim in enumerate(group):
                # sim.get_data()
                # Unpack data for the port we passed in as an argument
                ref, trans, absorb = sim.data['transmission_data'][port]
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
