import shutil
import psutil
#import subprocess
import itertools
import os
#import glob
#  import datetime
import copy
#  import hashlib
import multiprocessing as mp
import subprocess
import threading
import Queue
#import pandas
import numpy as np
import scipy.optimize as optz
import postprocess as pp
import tables as tb
# from tables.node import filenode
import time
#import pprint
import argparse as ap
import ruamel.yaml as yaml
import logging
# import gc3libs
# from gc3libs.core import Core, Engine

# get our custom config object and the logger function
from utils.simulator import Simulator
from utils.config import Config
from utils.utils import configure_logger, make_hash, get_combos
# from rcwa_app import RCWA_App
from collections import OrderedDict

def parse_file(path):
    """Super simple utility to parse a yaml file given a path"""
    with open(path, 'r') as cfile:
        text = cfile.read()
    conf = yaml.load(text, Loader=yaml.Loader)
    return conf

def run_sim(conf, q=None):
    """
    Actually runs simulation in a given directory. Expects the Config
    for the simulation as an argument.
    """
    log = logging.getLogger()
    start = time.time()
    sim = Simulator(copy.deepcopy(conf), q=q)
    if not sim.conf.variable_thickness:
        sim.conf.interpolate()
        sim.conf.evaluate()
        sim.update_id()
        sim.make_logger()
        try:
            os.makedirs(sim.dir)
        except OSError:
            pass
        log.info('Executing sim %s'%sim.id[0:10])
        sim.save_all()
        # path = os.path.join(os.path.basename(sim.dir), 'sim.hdf5')
        # sim.q.put(path, block=True)
        # sim.mode_solve()
    else:
        log.info('Computing a thickness sweep at %s' % sim.id[0:10])
        orig_id = sim.id[0:10]
        # Get all combinations of layer thicknesses
        keys, combos, bin_size = get_combos(sim.conf, sim.conf.variable_thickness)
        # Update base directory to new sub directory
        sim.conf['General']['base_dir'] = sim.dir
        # Set things up for the first combo
        combo = combos.pop()
        # First update all the thicknesses in the config. We make a copy of the
        # list because it gets continually updated in the config object
        var_thickness = sim.conf.variable_thickness
        for i, combo in enumerate(combo):
            keyseq = var_thickness[i]
            sim.conf[keyseq] = {'type': 'fixed', 'value': float(combo)}
        # With all the params updated we can now run substutions and
        # evaluations in the config that make have referred to some thickness
        # data, then make the subdir from the sim id and get the data
        sim.conf.interpolate()
        sim.conf.evaluate()
        sim.update_id()
        sim.make_logger()
        try:
            os.makedirs(sim.dir)
        except OSError:
            pass
        subpath = os.path.join(orig_id, sim.id[0:10])
        log.info('Computing initial thickness at %s', subpath)
        sim.save_all()
        # path = os.path.join(sim.dir, 'data.hdf5')
        # sim.q.put(path, block=True)
        # Now we can repeat the same exact process, but instead of rebuilding
        # the device we just update the thicknesses
        for combo in combos:
            for i, combo in enumerate(combo):
                keyseq = var_thickness[i]
                sim.conf[keyseq] = {'type': 'fixed', 'value': float(combo)}
            sim.update_id()
            subpath = os.path.join(orig_id, sim.id[0:10])
            log.info('Computing additional thickness at %s', subpath)
            os.makedirs(sim.dir)
            sim.save_all(update=True)
            # path = os.path.join(sim.dir, 'data.hdf5')
            # sim.q.put(path, block=True)
    end = time.time()
    runtime = end - start
    log.info('Simulation %s completed in %.2f seconds!', sim.id[0:10], runtime)
    sim.clean_sim()
    return None


class LayerFlux(tb.IsDescription):
    layer = tb.StringCol(60, pos=0)
    forward = tb.ComplexCol(pos=1, itemsize=8)
    backward = tb.ComplexCol(pos=2, itemsize=8)


class FileMerger(threading.Thread):

    def __init__(self, q, write_dir='', group=None, target=None, name=None):
        super(FileMerger, self).__init__(group=group, target=target, name=name)
        self.q = q
        outpath = os.path.join(write_dir, 'data.hdf5')
        print('Main file is %s' % outpath)
        self.hdf5 = tb.open_file(outpath, 'w')
         
    def run(self):
        while True:
            # print('QSIZE: %i'%self.q.qsize())
            try:
                path = self.q.get(False)
            except Queue.Empty:
                time.sleep(.1)
                continue
            else:
                if path is None:
                    self.hdf5.close()
                    break
                subfile = tb.open_file(path, 'r')
                # assert subfile != self.hdf5
                # # Path is the string to the file we want to merge
                # # self.hdf5.copy_children(subfile.root, self.hdf5.root,
                # #                         recursive=True, overwrite=True)
                # # subfile.copy_children(subfile.root, self.hdf5.root,
                # #                       recursive=True)
                for group in subfile.iter_nodes('/', classname='Group'):
                    # abssubdir, subfname = os.path.split(path)
                    # subdir = os.path.basename(abssubdir)
                    # where = '{}:{}'.format(os.path.join(subdir, subfname),
                    #                        group._v_name)
                    # print('Saving here', where)
                    self.hdf5.create_external_link('/', group._v_name, group)
                subfile.close()
                #     print('Copying group ', group)
                #     # self.hdf5.copy_node(group, newparent=self.hdf5.root,
                #     #                   recursive=True)
                #     group._f_copy(newparent=self.hdf5.root, recursive=True)
        return


class FileWriter(threading.Thread):

    def __init__(self, q, write_dir='', group=None, target=None, name=None):
        super(FileWriter, self).__init__(group=group, target=target, name=name)
        self.q = q
        outpath = os.path.join(write_dir, 'data.hdf5')
        self.hdf5 = tb.open_file(outpath, 'a')

    def run(self):
        while True:
            # print('QSIZE: %i'%self.q.qsize())
            try:
                data = self.q.get(False)
            except Queue.Empty:
                time.sleep(.1)
                continue
            else:
                if data is None:
                    self.hdf5.close()
                    break
                # Data tuple contains the following:
                # (string of method name to call, args list, kwargs dict)
                getattr(self, data[0])(*data[1], **data[2])
        return

    def create_array(self, *args, **kwargs):
        """
        This method is a completely tranparent wrapper around the create_array
        method of a PyTables HDF5 file object. It passes through any arguments
        and keyword arguments through untouched
        """
        if 'compression' in kwargs and kwargs['compression']:
            del kwargs['compression']
            filter_obj = tb.Filters(complevel=4, complib='blosc')
            try:
                self.hdf5.create_carray(*args, filters=filter_obj, **kwargs)
            except tb.NodeError:
                self.hdf5.remove_node(args[0], name=args[1])
                self.hdf5.create_carray(*args, filters=filter_obj, **kwargs)
        else:
            try:
                self.hdf5.create_array(*args, **kwargs)
            except tb.NodeError:
                self.hdf5.remove_node(args[0], name=args[1])
                self.hdf5.create_array(*args, **kwargs)

    def create_flux_table(self, flux_dict, *args, **kwargs):
        """
        Creates the table of layer fluxes for a simulation. Expects a
        dictionary whose keys are layer names and whose values are tuples
        containing the (forward, backward) complex fluxes as arguments. All
        following args and kwargs are passed through to the create_table method
        of the PyTables file object
        """

        try:
            table = self.hdf5.create_table(*args, description=LayerFlux,
                                           **kwargs)
        except tb.NodeError:
            table = self.hdf5.get_node(args[0], name=args[1],
                                       classname='Table')
            table.remove_rows(0)
        row = table.row
        for layer, (forward, backward) in flux_dict.iteritems():
            row['layer'] = layer
            row['forward'] = forward
            row['backward'] = backward
            row.append()
        table.flush()

    def save_attr(self, attr, path, name):
        """
        Save an attribute under the given name to a node in the config file
        """
        node = self.hdf5.get_node(path)
        node._v_attrs[name] = attr
        # fnode = filenode.new_node(self.hdf5, where=path, name='sim_conf.yml')
        # fnode.write(conf_str)
        # fnode.close()

    def clean_file(self, *args, **kwargs):
        """
        Deletes everything beneath the root group in the file
        """
        for node in self.hdf5.iter_nodes('/'):
            self.hdf5.remove_node(node._v_pathname, recursive=True)

class SimulationManager:

    """
    A class to manage running many simulations either in series or in parallel,
    collect and emit logs, write out data to files, etc
    """

    def __init__(self, gconf, log_level='INFO'):
        self.gconf = gconf
        lfile = os.path.join(gconf['General']['base_dir'], 'logs/sim_wrapper.log')
        try:
            log_level = self.gconf['General']['log_level']
        except KeyError:
            pass
        self.log = configure_logger(level=log_level, console=True, logfile=lfile)
        self.sim_confs = []
        self.write_queue = None
        self.reader = None

    def make_queue(self):
        """
        Makes the queue for transferring data from simulation subprocesses to
        the FileWriter thread. Sets a maximum size on the queue based on the
        number of data points in the arrays and the total ram on the system.
        """
        total_mem = psutil.virtual_memory().total
        # If we have hardcoded in a fixed number of samples, we can compute the
        # number of data points here.
        samps = [self.gconf['Simulation'][s] for s in ('x_samples',
                                                       'y_samples',
                                                       'z_samples')]
        # We can multiply by the ones that are hardcoded. For those
        # that are not, we have no way of evaluating the string expressions yet
        # so we'll just assume that they are 150 points
        # TODO: Maybe fix this random guessing
        max_points = 1
        for samp in samps:
            if type(samp) == int or type(samp) == float:
                max_points *= round(samp)
            else:
                max_points *= 150
        # Numpy complex128 consists of two 64 bit numbers, plus some overhead.
        # So 16 bytes + 8 bytes of overhead to be safe
        arr_mem = max_points*24
        # Subtract a gigabyte from total system memory to leave safety room
        maxsize = round((total_mem-(1024**3))/arr_mem)
        self.log.info('Maximum Queue Size: %i', maxsize)
        manager = mp.Manager()
        # We can go ahead and use maxsize directly because we left safety space
        # and there will also be items on the queue that are not massive arrays
        # and thus take up less space
        self.write_queue = manager.Queue(maxsize=maxsize)

    def make_listener(self):
        """
        Sets up the thread that listens to a queue for requests to write data
        to an HDF5 file. This prevents multiple subprocesses from attempting to
        write data to the HDF5 file at the same time
        """

        self.log.debug('Making listener')
        if self.write_queue is None:
            self.make_queue()
        basedir = self.gconf['General']['base_dir']
        self.reader = FileMerger(self.write_queue, write_dir=basedir)
        # self.reader = FileWriter(self.write_queue, write_dir=basedir)
        self.reader.start()

    def make_confs(self):
        """Make all the configuration dicts for each parameter combination"""
        self.log.info('Constructing simulator objects ...')
        locs, combos, bin_size = get_combos(self.gconf, self.gconf.variable)
        for combo in combos:
            # Make a copy of the global config for this parameter combos. This copy
            # represents an individual simulation
            sim_conf = self.gconf.copy()
            del sim_conf['Postprocessing']
            # Now we just overwrite all the variable parameters with their new
            # fixed values. Note that itertools.product is so wonderful and
            # nice that it preserves the order of the values in every combo
            # such that the combo values always line up with the proper
            # parameter name
            for i, combo in enumerate(combo):
                if 'frequency' in locs[i]:
                    sim_conf[self.gconf.variable[i]] = {'type': 'fixed',
                                                        'value': float(combo),
                                                        'bin_size': bin_size}
                else:
                    sim_conf[self.gconf.variable[i]] = {'type': 'fixed',
                                                        'value': float(combo)}
                    sim_conf[('Simulation','params','frequency')].update({'bin_size':
                                                                    0})
            self.sim_confs.append(sim_conf)
        if not combos[0]:
            sim_conf[('Simulation', 'params', 'frequency')].update({'bin_size': 0})

    def execute_jobs(self):
        """Given a list of configuration dictionaries, run them either serially or in
        parallel by applying run_sim to each dict. We do this instead of applying
        to an actual Simulator object because the Simulator objects are not
        pickeable and thus cannot be parallelized by the multiprocessing lib"""

        if self.gconf['General']['execution'] == 'serial':
            self.log.info('Executing sims serially')
            # Make the write queue, then instanstiate and run the thread that
            # pulls data from the queue and writes to the HDF5 file
            # if self.gconf['General']['save_as'] == 'hdf5':
            #     self.make_listener()
            # else:
            #     self.make_queue()
            for conf in self.sim_confs:
                run_sim(conf, q=self.write_queue)
            # self.write_queue.put(None, block=True)
            # if self.reader is not None:
            #     self.log.info('Joining FileWriter thread')
            #     self.reader.join()
        elif self.gconf['General']['execution'] == 'parallel':
            # if self.gconf['General']['save_as'] == 'hdf5':
            #     self.make_listener()
            # else:
            #     self.make_queue()
            # All this crap is necessary for killing the parent and all child
            # processes with CTRL-C
            num_procs = self.gconf['General']['num_cores']
            self.log.info('Executing sims in parallel using %s cores ...', str(num_procs))
            pool = mp.Pool(processes=num_procs)
            total_sims = len(self.sim_confs)
            remaining_sims = len(self.sim_confs)
            def callback(ind):
                callback.remaining_sims -= 1
                callback.log.info('%i out of %i simulations remaining'%(callback.remaining_sims,
                                                                callback.total_sims))
            callback.remaining_sims = remaining_sims
            callback.total_sims = total_sims
            callback.log = self.log
            results = {}
            # results = []
            self.log.debug('Entering try, except pool clause')
            inds = []
            try:
                # res = pool.map_async(run_sim, self.sim_confs, callback=callback)
                # res.get(999999999)
                # pool.close()
                for ind, conf in enumerate(self.sim_confs):
                    res = pool.apply_async(run_sim, (conf, ),
                                           {'q':self.write_queue}, callback=callback)
                    # results.append(res)
                    results[ind] = res
                    inds.append(ind)
                    # res = pool.apply_async(run_sim, (conf,), callback=callback)
                    # results[ind] = res
                    # print(del_list)
                    # for completed_ind in del_list:
                    #     results[completed_ind].wait(9999999)
                    #     del results[completed_ind]
                    # del_list = []
                self.log.debug("Waiting on results")
                # for sid, res in results.items():
                self.log.debug('Results before wait loop: %s',
                               str(list(results.keys())))
                for ind in inds:
                    # We need to add this really long timeout so that subprocesses
                    # receive keyboard interrupts. If our simulations take longer
                    # than this timeout, an exception would be raised but that
                    # should never happen
                    res = results[ind]
                    self.log.debug('Sim #: %s', str(ind))
                    res.wait(99999999)
                    self.log.debug('Done waiting on Sim ID %s', str(ind))
                    del results[ind]
                    self.log.debug('Cleaned results: %s',
                                   str(list(results.keys())))
                    # self.log.debug('Number of items in queue: %i',
                    #                self.write_queue.qsize())
                self.log.debug('Finished waiting')
                pool.close()
            except KeyboardInterrupt:
                pool.terminate()
            self.log.debug('Joining pool')
            pool.join()
            # self.write_queue.put(None, block=True)
            # if self.reader is not None:
            #     self.log.info('Joining FileWriter thread')
            #     self.reader.join()
            # for res in results:
            #     print(res)
            #     print(res.get())
        elif self.gconf['General']['execution'] == 'gc3':
            self.log.info('Executing jobs using gc3 submission tools')
            self.gc3_submit(self.gconf, self.sim_confs)

    def spectral_wrapper(self, opt_pars):
        """A wrapper function to handle spectral sweeps and postprocessing for the scipy minimizer. It
        accepts the initial guess as a vector, the base config file, and the keys that map to the
        initial guesses. It runs a spectral sweep (in parallel if specified), postprocesses the results,
        and returns the spectrally weighted reflection"""
        # TODO: Pass in the quantity we want to optimize as a parameter, then compute and return that
        # instead of just assuming reflection

        # Optimizing shell thickness could result is negative thickness so we need to take absolute
        # value here
        #opt_pars[0] = abs(opt_pars[0])
        self.log.info('Param keys: %s', str(self.gconf.optimized))
        self.log.info('Current values %s', str(opt_pars))
        # Clean up old data unless the user asked us not to. We do this first so on the last
        # iteration all our data is left intact
        basedir = self.gconf['General']['base_dir']
        ftype = self.gconf['General']['save_as']
        hdf_file = os.path.join(basedir, 'data.hdf5')
        if not self.gconf['General']['opt_keep_intermediates']:
            for item in os.listdir(basedir):
                if os.path.isdir(item) and item != 'logs':
                    shutil.rmtree(item)
                if 'data.hdf5' in item:
                    os.remove('data.hdf5')

        # Set the value key of all the params we are optimizing over to the current
        # guess
        for i in range(len(self.gconf.optimized)):
            keyseq = self.gconf.optimized[i]
            self.log.info(keyseq)
            valseq = list(keyseq) + ['value']
            self.log.info(valseq)
            self.gconf[valseq] = float(opt_pars[i])
        # Make all the sim objects
        self.sim_confs = []
        sims = self.make_confs()
        # Let's reuse the convergence information from the previous iteration if it exists
        # NOTE: This kind of assumes your initial guess was somewhat decent with regards to the in plane
        # geometric variables and the optimizer is staying relatively close to that initial guess. If
        # the optimizer is moving far away from its previous guess at each step, then the fact that a
        # specific frequency may have been converged previously does not mean it will still be converged
        # with this new set of variables.
        info_file = os.path.join(basedir, 'conv_info.txt')
        if os.path.isfile(info_file):
            conv_dict = {}
            with open(info_file, 'r') as info:
                for line in info:
                    freq, numbasis, conv_status = line.strip().split(',')
                    if conv_status == 'converged':
                        conv_dict[freq] = (True, numbasis)
                    elif conv_status == 'unconverged':
                        conv_dict[freq] = (False, numbasis)
            for sim in sims:
                freq = str(sim.conf['Simulation']['params']['frequency']['value'])
                conv, numbasis = conv_dict[freq]
                # Turn off adaptive convergence and update the number of basis
                # terms
                if conv:
                    self.log.info('Frequency %s converged at %s basis terms', freq, numbasis)
                    sim.conf['General']['adaptive_convergence'] = False
                    sim.conf['Simulation']['params'][
                        'numbasis']['value'] = int(numbasis)
                # For sims that haven't converged, set the number of basis terms to the last
                # tested value so we're closer to our goal of convergence
                else:
                    self.log.info('Frequency %s converged at %s basis terms', freq, numbasis)
                    sim.conf['Simulation']['params'][
                        'numbasis']['value'] = int(numbasis)
        # With the leaf directories made and the number of basis terms adjusted,
        # we can now kick off our frequency sweep
        self.execute_jobs()
        # We need to wait for the writer thread to empty the queue for us
        # before we can postprocess the data
        if ftype == 'hdf5' and self.write_queue is not None:
            while not self.write_queue.empty():
                self.log.info('Waiting for queue to empty')
                self.log.info(self.write_queue.qsize())
                time.sleep(.1)
        #####
        # TODO: This needs to be generalized. The user could pass in the name of
        # a postprocessing function in the config file. The function will be called
        # and used as the quantity for optimization
        #####

        # With our frequency sweep done, we now need to postprocess the results.
        # Configure logger
        # log = configure_logger('error','postprocess',
        #                          os.path.join(self.gconf['General']['base_dir'],'logs'),
        #                          'postprocess.log')
        # Compute transmission data for each individual sim
        cruncher = pp.Cruncher(self.gconf)
        cruncher.collect_sims()
        for sim in cruncher.sims:
            # cruncher.transmissionData(sim)
            sim.get_data()
            sim.transmissionData()
            sim.write_data()
        # Now get the fraction of photons absorbed
        gcruncher = pp.Global_Cruncher(
            self.gconf, cruncher.sims, cruncher.sim_groups, cruncher.failed_sims)
        gcruncher.group_against(
            ['Simulation', 'params', 'frequency', 'value'], self.gconf.variable)
        photon_fraction = gcruncher.fractional_absorbtion()[0]
        # Lets store information we discovered from our adaptive convergence procedure so we can resue
        # it in the next iteration.
        if self.gconf['General']['adaptive_convergence']:
            self.log.info('Storing adaptive convergence results ...')
            with open(info_file, 'w') as info:
                for sim in sims:
                    freq = sim.conf['Simulation']['params']['frequency']['value']
                    conv_path = os.path.join(sim.dir, 'converged_at.txt')
                    nconv_path = os.path.join(sim.dir, 'not_converged_at.txt')
                    if os.path.isfile(conv_path):
                        conv_f = open(conv_path, 'r')
                        numbasis = conv_f.readline().strip()
                        conv_f.close()
                        conv = 'converged'
                    elif os.path.isfile(nconv_path):
                        conv_f = open(nconv_path, 'r')
                        numbasis = conv_f.readline().strip()
                        conv_f.close()
                        conv = 'unconverged'
                    else:
                        # If we were converged on a previous iteration, adaptive
                        # convergence was switched off and there will be no file to
                        # read from
                        conv = 'converged'
                        numbasis = sim.conf['Simulation'][
                            'params']['numbasis']['value']
                    info.write('%s,%s,%s\n' % (str(freq), numbasis, conv))
            self.log.info('Finished storing convergence results!')
        #print('Total time = %f'%delta)
        #print('Num calls after = %i'%gcruncher.weighted_transmissionData.called)
        # This is a minimizer, we want to maximize the fraction of photons absorbed and thus minimize
        # 1 minus that fraction
        return 1 - photon_fraction


    def run_optimization(self):
        """Runs an optimization on a given set of parameters"""
        self.log.info("Running optimization")
        self.log.info(self.gconf.optimized)
        # Make sure the only variable parameter we have is a sweep through
        # frequency
        for keyseq in self.gconf.variable:
            if keyseq[-1] != 'frequency':
                self.log.error('You should only be sweep through frequency during an '
                          'optimization')
                quit()
        # Collect all the guesses
        guess = np.zeros(len(self.gconf.optimized))
        for i in range(len(self.gconf.optimized)):
            keyseq = self.gconf.optimized[i]
            par_data = self.gconf.get(keyseq)
            guess[i] = par_data['guess']
        # Max iterations and tolerance
        tol = self.gconf['General']['opt_tol']
        ftol = self.gconf['General']['func_opt_tol']
        max_iter = self.gconf['General']['opt_max_iter']
        os.makedirs(os.path.join(self.gconf['General']['base_dir'], 'opt_dir'))
        # Run the simplex optimizer
        opt_val = optz.minimize(self.spectral_wrapper,
                                guess,
                                method='Nelder-Mead',
                                options={'maxiter': max_iter, 'xatol': tol,
                                         'fatol': ftol, 'disp': True})
        self.log.info(opt_val.message)
        self.log.info('Optimal values')
        self.log.info(self.gconf.optimized)
        self.log.info(opt_val.x)
        # Write out the results to a file
        out_file = os.path.join(
            self.gconf['General']['base_dir'], 'optimization_results.txt')
        with open(out_file, 'w') as out:
            out.write('# Param name, value\n')
            for key, value in zip(self.gconf.optimized, opt_val.x):
                out.write('%s: %f\n' % (str(key), value))
        self.write_queue.put(None, block=True)
        return opt_val.x

    def run(self):
        """
        The main run methods that decides what kind of simulation to run based on the
        provided config objects
        """

        if not self.gconf.optimized:
            # Get all the sims
            self.make_confs()
            self.log.info("Executing job campaign")
            self.execute_jobs()
        elif self.gconf.optimized:
            self.run_optimization()
        else:
            self.log.error('Unsupported configuration for a simulation run. Not a '
                           'single sim, sweep, or optimization. Make sure your sweeps are '
                           'configured correctly, and if you are running an optimization '
                           'make sure you do not have any sorting parameters specified')


    def gc3_submit(self):
        """
        This function runs jobs on a bunch of remote hosts via SSH
        and well as on the local machine using a library called gc3pie. Requires
        gc3pie to be installed and configured. Currently super slow and not really
        worth using.
        """
        self.log.info('GC3 FUNCTION')
        jobs = []
        for conf in self.sim_confs:
            # Set up the config object and make local simulation directory
            conf.interpolate()
            conf.evaluate()
            sim_id = make_hash(conf.data)
            sim_dir = os.path.join(conf['General']['base_dir'], sim_id[0:10])
            conf['General']['sim_dir'] = sim_dir
            try:
                os.makedirs(sim_dir)
            except OSError:
                pass
            conf.write(os.path.join(sim_dir, 'sim_conf.yml'))
            # Create the GC3 Application object and append to list of apps to run
            app = RCWA_App(conf)
            jobs.append(app)
        print(jobs)
        # Get the config for all resources, auth, etc.
        cfg = gc3libs.config.Configuration(*gc3libs.Default.CONFIG_FILE_LOCATIONS,
                                           auto_enable_auth=True)
        gcore = Core(cfg)
        eng = Engine(gcore, tasks=jobs, retrieve_overwrites=True)
        # eng = Engine(gcore, tasks=jobs)
        try:
            eng.progress()
            stats = eng.stats()
            while stats['TERMINATED'] < stats['total']:
                time.sleep(10)
                print('Checking jobs')
                eng.progress()
                stats = eng.stats()
                print(stats)
                states = eng.update_job_state()
                print(states)
        except KeyboardInterrupt:
            print('KILLING REMOTE JOBS BEFORE QUITTING')
            print('PLEASE BE PATIENT')
            # Kill all remote jobs
            for task in jobs:
                eng.kill(task)
            # update job states
            eng.progress()
            # free remote resources
            for task in jobs:
                eng.free(task)
            # now raise exception
            raise


def main():

    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file', type=str, help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('--log_level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="""Logging level for the run""")
    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        global_conf = Config(path=os.path.abspath(args.config_file))
        # Expand all the environment variables in the global config but
        # otherwise don't modify anything
        global_conf.expand_vars()
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    manager = SimulationManager(global_conf, log_level=args.log_level)
    manager.run()

if __name__ == '__main__':
    main()
