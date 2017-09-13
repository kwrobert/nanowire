import argparse as ap
import os
import copy
import logging
import multiprocessing as mp
import multiprocessing.dummy as mpd

from utils.config import Config
from utils.simulation import Simulation, SimulationGroup
from utils.utils import configure_logger, cmp_dicts
from itertools import repeat

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

def _crunch_local(sim, gconf):
    sim = Simulation(Config(sim))
    _process(sim, "Cruncher", gconf)
    sim.write_data()
    sim.clear_data()

def _plot_local(sim, gconf):
    sim = Simulation(Config(sim))
    _process(sim, "Plotter", gconf)
    sim.clear_data()

def _crunch_global(sim_group, gconf):
    sims = [Simulation(Config(path)) for path in sim_group]
    sim_group = SimulationGroup(sims)
    _process(sim_group, "Global_Cruncher", gconf)
    for sim in sim_group.sims:
        sim.clear_data()

def _plot_global(sim_group, gconf):
    sims = [Simulation(Config(path)) for path in sim_group]
    sim_group = SimulationGroup(sims)
    _process(sim_group, "Global_Plotter", gconf)
    for sim in sim_group.sims:
        sim.clear_data()

def _call_func(quantity, obj, args):
    """
    Calls an instance method of an object with args
    """

    try:
        result = getattr(obj, quantity)(*args)
    except KeyError:
        print("Unable to call the following function: %s", quantity)
        raise
    return result

def _process(obj, process, gconf):
    """
    Calls a process on an object. The object could be a Simulation object,
    or a SimulationGroup object. It just loops through the functions
    defined in the process subsection of the Postprocessing section in the
    config file, and uses call_func to call the object's method with the
    names defined in the config.
    """

    to_compute = {quant: data for quant, data in
                  gconf['Postprocessing'][process].items() if
                  data['compute']}
    for quant, data in to_compute.items():
        argsets = data['args']
        if argsets and isinstance(argsets[0], list):
            for argset in argsets:
                if argset:
                    _call_func(quant, obj, argset)
                else:
                    _call_func(quant, obj, [])
        else:
            if argsets:
                _call_func(quant, obj, argsets)
            else:
                _call_func(quant, obj, [])

class Processor(object):
    """
    Generic class for automating the processing of Simulations and
    SimulationGroups
    """

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
        """
        Collect all the simulations beneath the base of the directory tree
        """

        sims = []
        failed_sims = []
        ftype = self.gconf['General']['save_as']
        # Get correct data file name
        if ftype == 'npz':
            datfile = self.gconf['General']['base_name'] + '.npz'
        elif ftype == 'hdf5':
            datfile = 'sim.hdf5'
        else:
            raise ValueError('Invalid file type specified in config')
        # Find the data files and instantiate Simulation objects
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

    def get_param_vals(self, parseq):
        """
        Return all possible values of the provided parameter for this sweep
        """

        vals = []
        for sim in self.sims:
            val = sim.conf[parseq]
            if val not in vals:
                vals.append(val)
        return vals

    def filter_by_param(self, pars):
        """Accepts a dict where the keys are parameter names and the values are
        a list of possible values for that parameter. Any simulation whose
        parameter does not match any of the provided values is removed from the
        sims and sim_groups attribute"""

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

        self.log.info('Grouping sims against: %s', str(key))
        # We need only need a shallow copy of the list containing all the sim
        # objects We don't want to modify the orig list but we wish to share
        # the sim objects the two lists contain
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
                    self.log.info('RESULTS DIR: %s', path)
                    try:
                        os.makedirs(path)
                    except OSError:
                        pass
            for sim in group:
                sim.conf['General']['results_dir'] = path
                outpath = os.path.join(sim.conf['General']['sim_dir'],
                                       'sim_conf.yml')
                sim.conf.write(outpath)
        # Sort the groups in increasing order of the provided sort key
        if sort_key:
            sim_groups.sort(key=lambda group: group[0].conf[key])
        sim_groups = [SimulationGroup(sims) for sims in sim_groups]
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
        groups = [SimulationGroup(sims) for sims in groups]
        self.sim_groups = groups
        return groups

    def replace(self):
        for i, sim in enumerate(self.sims):
            path = os.path.join(sim.conf['General']['sim_dir'], 'sim_conf.yml')
            self.sims[i] = path

        for i, group in enumerate(self.sim_groups):
            new_group = []
            for sim in group.sims:
                new_group.append(os.path.join(sim.conf['General']['sim_dir'],
                                              'sim_conf.yml'))

            self.sim_groups[i] = new_group

    def call_func(self, quantity, obj, args):
        """
        Calls an instance method of an object with args
        """

        try:
            result = getattr(obj, quantity)(*args)
        except KeyError:
            self.log.error("Unable to call the following function: %s",
                           quantity, exc_info=True, stack_info=True)
            raise
        return result

    def _process(self, obj, process):
        """
        Calls a process on an object. The object could be a Simulation object,
        or a SimulationGroup object. It just loops through the functions
        defined in the process subsection of the Postprocessing section in the
        config file, and uses call_func to call the object's method with the
        names defined in the config.
        """

        self.log.info('Running %s process for obj %s', process, str(obj))
        to_compute = {quant: data for quant, data in
                      self.gconf['Postprocessing'][process].items() if
                      data['compute']}
        for quant, data in to_compute.items():
            argsets = data['args']
            self.log.info('Calling %s with args %s', str(quant), str(argsets))
            if argsets and isinstance(argsets[0], list):
                for argset in argsets:
                    self.log.info('Calling with individual argset %s', str(argset))
                    if argset:
                        self.call_func(quant, obj, argset)
                    else:
                        self.call_func(quant, obj, [])
            else:
                if argsets:
                    self.call_func(quant, obj, argsets)
                else:
                    self.call_func(quant, obj, [])

    def crunch_local(self, sim):
        _crunch_local(sim, self.gconf)

    def crunch_local_all(self):
        self.log.info('Beginning data crunch for all sims ...')
        if not self.gconf['General']['post_parallel']:
            for sim in self.sims:
                _crunch_local(sim, self.gconf)
        else:
            num_procs = self.gconf['General']['num_cores']
            self.log.info('Crunching sims in parallel using %s cores ...', str(num_procs))
            args_list = list(zip(self.sims, repeat(self.gconf)))
            pool = mp.Pool(processes=num_procs)
            pool.starmap(_crunch_local, args_list)
            pool.close()
            pool.join()

    def plot_local(self, sim):
        _plot_local(sim, self.gconf)

    def plot_local_all(self):
        self.log.info('Beginning local plotting for all sims ...')
        if not self.gconf['General']['post_parallel']:
            for sim in self.sims:
                _plot_local(sim, self.gconf)
        else:
            num_procs = self.gconf['General']['num_cores']
            self.log.info('Plotting sims in parallel using %s cores ...', str(num_procs))
            pool = mp.Pool(processes=num_procs)
            args_list = list(zip(self.sims, repeat(self.gconf)))
            pool.starmap(_plot_local, args_list)
            pool.close()
            pool.join()

    def crunch_global(self, sim_group):
        _process(sim_group, "Global_Cruncher", self.gconf)

    def crunch_global_all(self):
        self.log.info('Beginning global data crunch for all sim groups ...')
        if not self.gconf['General']['post_parallel']:
            for group in self.sim_groups:
                _crunch_global(group, self.gconf)
        else:
            num_procs = self.gconf['General']['num_cores']
            self.log.info('Crunching sim groups in parallel using %s cores ...', str(num_procs))
            pool = mp.Pool(processes=num_procs)
            args_list = list(zip(self.sim_groups, repeat(self.gconf)))
            pool.starmap(_crunch_global, args_list)
            pool.close()
            pool.join()

    def plot_global(self, sim_group):
        _process(sim_group, "Global_Plotter", self.gconf)

    def plot_global_all(self):
        self.log.info('Beginning global data plot for all sim groups ...')
        if not self.gconf['General']['post_parallel']:
            for group in self.sim_groups:
                _plot_global(group, self.gconf)
        else:
            num_procs = self.gconf['General']['num_cores']
            self.log.info('Plotting sim groups in parallel using %s cores ...', str(num_procs))
            pool = mp.Pool(processes=num_procs)
            args_list = list(zip(self.sim_groups, repeat(self.gconf)))
            pool.starmap(_plot_global, args_list)
            pool.close()
            pool.join()


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
    # Collect the sims once up here and reuse them later
    proc = Processor(conf)
    logger.info('Collecting sims')
    proc.collect_sims()
    # First we need to group against if specified. Grouping against corresponds
    # to "leaves" in the tree
    if args.group_against:
        sim_groups = proc.group_against(group_ag, conf.variable)
    # Next we group by. This corresponds to building the parent nodes for each
    # set of leaf groups
    if args.group_by:
        sim_groups = proc.group_by(group_by)
    # Filter if specified
    if args.filter_by:
        filt_dict = {}
        for item in args.filter_by:
            par, vals = item.split(':')
            vals = vals.split(',')
            filt_dict[par] = vals
        logger.info('Here is the filter dictionary: %s', filt_dict)
        proc.filter_by_param(filt_dict)
    proc.replace()
    # Now do all the work
    if not args.no_crunch:
        proc.crunch_local_all()
    if not args.no_gcrunch:
        proc.crunch_global_all()
    if not args.no_plot:
        proc.plot_local_all()
    if not args.no_gplot:
        proc.plot_global_all()


if __name__ == '__main__':
    main()
