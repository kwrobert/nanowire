import shutil
import psutil
import os
import sys
import copy
from multiprocessing.pool import Pool
import multiprocessing as mp
import threading
import Queue
import numpy as np
import scipy.optimize as optz
import postprocess as pp
import tables as tb
# from tables.node import filenode
import time
import ruamel.yaml as yaml
import logging
import traceback
import logging_tree
import S4
import scipy.interpolate as spi
import scipy.integrate as intg
import scipy.constants as constants
# import gc3libs
# from gc3libs.core import Core, Engine

# get our custom config object and the logger function
from utils.utils import make_hash, get_combos
from utils.config import Config

# from rcwa_app import RCWA_App
class RejectSimFilter(logging.Filter):

    def filter(self, record):
        if hasattr(record, 'ID'):
            return 0
        else:
            return 1


# Configure logging for this module
# Get numeric level safely
logfile = 'logs/simulate.log'
debug = getattr(logging, 'debug'.upper(), None)
info = getattr(logging, 'info'.upper(), None)
# Set formatting
formatter = logging.Formatter('%(asctime)s [%(module)s:%(name)s:%(levelname)s]'
                              ' - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(debug)
log_dir, logfile = os.path.split(os.path.expandvars(logfile))
# Set up file handler
try:
    os.makedirs(log_dir)
except OSError:
    # Log dir already exists
    pass
output_file = os.path.join(log_dir, logfile)
fhandler = logging.FileHandler(output_file)
fhandler.setFormatter(formatter)
fhandler.setLevel(debug)
fhandler.addFilter(RejectSimFilter())
logger.addHandler(fhandler)
# Logging to console
ch = logging.StreamHandler()
ch.setLevel(info)
ch.setFormatter(formatter)
ch.addFilter(RejectSimFilter())
logger.addHandler(ch)


# This will log any uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value,
                                                    exc_traceback))


sys.excepthook = handle_exception


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            # Here we add some debugging help.
            log = logging.getLogger(__name__)
            log.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can clean up.
            # This kills the parent process if it calls .get or .wait on the
            # AsyncResult object returned by apply_async
            # raise
        # It was fine, give a normal answer
        return result


class LoggingPool(Pool):
    def apply_async(self, func, args=(), kwds={}, callback=None):
        return Pool.apply_async(self, LogExceptions(func), args, kwds,
                                callback)


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
    log = logging.getLogger(__name__)
    start = time.time()
    sim = Simulator(copy.deepcopy(conf), q=q)
    try:
        if not sim.conf.variable_thickness:
            sim.evaluate_config()
            sim.update_id()
            sim.make_logger()
            try:
                os.makedirs(sim.dir)
            except OSError:
                pass
            log.info('Executing sim %s', sim.id[0:10])
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
            first_combo = combos.pop()
            # First update all the thicknesses in the config. We make a copy of the
            # list because it gets continually updated in the config object
            var_thickness = sim.conf.variable_thickness
            for i, param_val in enumerate(first_combo):
                keyseq = var_thickness[i]
                sim.conf[keyseq] = {'type': 'fixed', 'value': float(param_val)}
            # With all the params updated we can now run substutions and
            # evaluations in the config that make have referred to some thickness
            # data, then make the subdir from the sim id and get the data
            sim.evaluate_config()
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
                for i, param_val in enumerate(combo):
                    keyseq = var_thickness[i]
                    sim.conf[keyseq] = {'type': 'fixed', 'value': float(param_val)}
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
    except:
        trace = traceback.format_exc()
        msg = 'Sim {} raised the following exception:\n{}'.format(sim.id,
                                                                  trace)
        log.error(msg)
        sim.log.error(trace)
        raise
    return None


class LayerFlux(tb.IsDescription):
    layer = tb.StringCol(60, pos=0)
    forward = tb.ComplexCol(pos=1, itemsize=8)
    backward = tb.ComplexCol(pos=2, itemsize=8)


class InstanceFilter(logging.Filter):

    def __init__(self, ID, name=""):
        super(InstanceFilter, self).__init__(name=name)
        self.ID = ID

    def filter(self, record):
        if not hasattr(record, 'ID'):
            return 0
        if record.ID == self.ID:
            return 1
        else:
            return 0

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
        if os.path.isfile(gconf):
            self.gconf = Config(path=os.path.abspath(gconf))
        else:
            self.gconf = gconf
        self.gconf.expand_vars()
        lfile = os.path.join(self.gconf['General']['base_dir'],
                             'logs/sim_manager.log')
        try:
            log_level = self.gconf['General']['log_level']
        except KeyError:
            pass
        # self.log = configure_logger(level=log_level, console=True,
        #                             logfile=lfile, name=__name__)
        self.log = logging.getLogger(__name__)
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
            # pool = LoggingPool(processes=num_procs)
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
                for ind, conf in enumerate(self.sim_confs):
                    res = pool.apply_async(run_sim, (conf, ),
                                           {'q':self.write_queue},
                                           callback=callback)
                    results[ind] = res
                    inds.append(ind)
                self.log.debug("Waiting on results")
                self.log.debug('Results before wait loop: %s',
                               str(list(results.keys())))
                for ind in inds:
                    # We need to add this really long timeout so that
                    # subprocesses receive keyboard interrupts. If our
                    # simulations take longer than this timeout, an exception
                    # would be raised but that should never happen
                    res = results[ind]
                    self.log.debug('Sim #: %s', str(ind))
                    res.wait(99999999)
                    # res.get(99999999)
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
        self.log.info('Finished executing jobs!')

    def spectral_wrapper(self, opt_pars):
        """A wrapper function to handle spectral sweeps and postprocessing for the scipy minimizer. It
        accepts the initial guess as a vector, the base config file, and the keys that map to the
        initial guesses. It runs a spectral sweep (in parallel if specified), postprocesses the results,
        and returns the spectrally weighted reflection"""
        # TODO: Pass in the quantity we want to optimize as a parameter, then compute and return that
        # instead of just assuming reflection

        # Optimizing shell thickness could result is negative thickness so we need to take absolute
        # value here
        # opt_pars[0] = abs(opt_pars[0])
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
                    os.remove(hdf_file)

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

class Simulator():

    def __init__(self, conf, q=None):
        self.conf = conf
        self.q = q
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        period = self.conf['Simulation']['params']['array_period']['value']
        self.id = make_hash(conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'], self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir
        self.s4 = S4.New(Lattice=((period, 0), (0, period)),
                         NumBasis=int(round(numbasis)))
        self.data = {}
        self.flux_dict = {}
        self.hdf5 = None
        self.runtime = 0
        self.period = period

    def __del__(self):
        """
        Need to make sure we close the file descriptor for the fileHandler that
        this instance attached to the module level logger. If we don't, we'll
        eventually use up all the available file descriptors on the system
        """
        # Sometimes we hit an error before the log object gets created and
        # assigned as an attribute. Without the try, except we would get an
        # attribute error which makes error messages confusing and useless
        self._clean_logger()

    def _clean_logger(self):
        """
        Cleans up all the logging stuff associated with this instance
        """
        try:
            self.fhandler.close()
            module_logger = logging.getLogger(__name__)
            module_logger.removeHandler(module_logger)
        except AttributeError:
            pass

    def open_hdf5(self):
        fpath = os.path.join(self.dir, 'sim.hdf5')
        self.hdf5 = tb.open_file(fpath, 'w')

    def clean_sim(self):
        try:
            self._clean_logger()
            del self.log
            del self.s4
        except AttributeError:
            pass

    def evaluate_config(self):
        """
        Expands all environment variables in the config and resolves all
        references
        """
        self.conf.interpolate()
        self.conf.evaluate()

    def update_id(self):
        """Update sim id. Used after changes are made to the config"""
        self.id = make_hash(self.conf.data)
        sim_dir = os.path.join(self.conf['General']['base_dir'], self.id[0:10])
        self.conf['General']['sim_dir'] = sim_dir
        self.dir = sim_dir
        try:
            os.makedirs(sim_dir)
        except OSError:
            pass

    def make_logger(self, log_level='info'):
        """Makes the logger for this simulation"""
        self._clean_logger()
        # Add the file handler for this instance's log file and attach it to
        # the module level logger
        self.fhandler = logging.FileHandler(os.path.join(self.dir, 'sim.log'))
        self.fhandler.addFilter(InstanceFilter(self.id))
        formatter = logging.Formatter('%(asctime)s [%(module)s:%(name)s:%(levelname)s] - %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
        self.fhandler.setFormatter(formatter)
        log = logging.getLogger(__name__)
        log.addHandler(self.fhandler)
        # Store the logger adapter to the module level logger as an attribute.
        # We use this to log in any methods, and the sim_id of this instance
        # will get stored in the log record
        self.log = logging.LoggerAdapter(log, {'ID': self.id})
        self.log.debug('Logger initialized')

    def configure(self):
        """Configure options for the RCWA solver"""
        self.s4.SetOptions(**self.conf['Solver'])

    def _get_epsilon(self, path):
        """Returns complex dielectric constant for a material by pulling in nk
        text file, interpolating, computing nk values at freq, and
        converting"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        # Get data
        freq_vec, n_vec, k_vec = np.loadtxt(path, unpack=True)
        # Get n and k at specified frequency via interpolation
        f_n = spi.interp1d(freq_vec, n_vec, kind='nearest',
                           bounds_error=False, fill_value='extrapolate')
        f_k = spi.interp1d(freq_vec, k_vec, kind='nearest',
                           bounds_error=False, fill_value='extrapolate')
        n, k = f_n(freq), f_k(freq)
        # Convert to dielectric constant
        # NOTE!!: This assumes the relative magnetic permability (mew) is 1
        epsilon_real = n**2 - k**2
        epsilon_imag = 2 * n * k
        epsilon = complex(epsilon_real, epsilon_imag)
        return epsilon

    def _get_incident_amplitude_anna(self):
        freq = self.conf['Simulation']['params']['frequency']['value']
        path = '/home/krobert/software/nanowire/nanowire/Input_sun_power.txt'
        freq_vec, p_vec = np.loadtxt(path, unpack=True)
        p_of_f = spi.interp1d(freq_vec, p_vec)
        intensity = p_of_f(freq)
        self.log.debug('Incident Intensity: %s', str(intensity))
        area = self.period*self.period
        power = intensity*area
        self.log.debug('Incident Power: %s', str(power))
        # We need to reduce amplitude of the incident wave depending on
        #  incident polar angle
        # E = np.sqrt(constants.c*constants.mu_0*f_p(freq))*np.cos(polar_angle)
        E = np.sqrt(constants.c * constants.mu_0 * intensity)
        self.log.debug('Incident Amplitude: %s', str(E))
        return E

    def _get_incident_amplitude(self):
        """Returns the incident amplitude of a wave depending on frequency"""
        freq = self.conf['Simulation']['params']['frequency']['value']
        polar_angle = self.conf['Simulation']['params']['polar_angle']['value']
        path = self.conf['Simulation']['input_power']
        bin_size = self.conf['Simulation']['params']['frequency']['bin_size']
        # Get NREL AM1.5 data
        freq_vec, p_vec = np.loadtxt(path, unpack=True, delimiter=',')
        # Get all available intensity values within this bin
        left = freq - bin_size / 2.0
        right = freq + bin_size / 2.0
        inds = np.where((left < freq_vec) & (freq_vec < right))[0]
        # Check for edge cases
        if len(inds) == 0:
            # It is unphysical to claim that an input wave of a single
            # frequency can contain any power. If we're simulating at a single
            # frequency, just assume the wave has the intensity contained within
            # the NREL bin surrounding that frequency
            self.log.warning('Your bins are smaller than NRELs! Using NREL'
                             ' bin size')
            closest_ind = np.argmin(np.abs(freq_vec - freq))
            # Is the closest one to the left or the right?
            if freq_vec[closest_ind] > freq:
                other_ind = closest_ind - 1
                left = freq_vec[other_ind]
                left_intensity = p_vec[other_ind]
                right = freq_vec[closest_ind]
                right_intensity = p_vec[closest_ind]
            else:
                other_ind = closest_ind + 1
                right = freq_vec[other_ind]
                right_intensity = p_vec[other_ind]
                left = freq_vec[closest_ind]
                left_intensity = p_vec[closest_ind]
        elif inds[0] == 0:
            raise ValueError('Your leftmost bin edge lies outside the'
                             ' range provided by NREL')
        elif inds[-1] == len(freq_vec):
            raise ValueError('Your rightmost bin edge lies outside the'
                             ' range provided by NREL')
        else:
            # A simple linear interpolation given two pairs of data points, and the
            # desired x point
            def lin_interp(x1, x2, y1, y2, x):
                return ((y2 - y1) / (x2 - x1)) * (x - x2) + y2
            # If the left or right edge lies between NREL data points, we do a
            # linear interpolation to get the irradiance values at the bin edges.
            # If the left of right edge happens to be directly on an NREL bin edge
            # (unlikely) the linear interpolation will just return the value at the
            # NREL bin. Also the selection of inds above excluded the case of left
            # or right being equal to an NREL bin,
            left_intensity = lin_interp(freq_vec[inds[0] - 1], freq_vec[inds[0]],
                                    p_vec[inds[0] - 1], p_vec[inds[0]], left)
            right_intensity = lin_interp(freq_vec[inds[-1]], freq_vec[inds[-1] + 1],
                                     p_vec[inds[-1]], p_vec[inds[-1] + 1], right)
        # All the frequency values within the bin and including the bin edges
        freqs = [left]+list(freq_vec[inds])+[right]
        # All the intensity values
        intensity_values = [left_intensity]+list(p_vec[inds])+[right_intensity]
        self.log.debug(freqs)
        self.log.debug(intensity_values)
        # Just use a trapezoidal method to integrate the spectrum
        intensity = intg.trapz(intensity_values, x=freqs)
        self.log.debug('Incident Intensity: %s', str(intensity))
        area = self.period*self.period
        power = intensity*area
        self.log.debug('Incident Power: %s', str(power))
        # We need to reduce amplitude of the incident wave depending on
        #  incident polar angle
        # E = np.sqrt(constants.c*constants.mu_0*f_p(freq))*np.cos(polar_angle)
        E = np.sqrt(constants.c * constants.mu_0 * intensity)
        self.log.debug('Incident Amplitude: %s', str(E))
        return E

    def set_excitation(self):
        """Sets the exciting plane wave for the simulation"""
        f_phys = self.conf['Simulation']['params']['frequency']['value']
        self.log.debug('Physical Frequency = %E' % f_phys)
        c_conv = constants.c / self.conf['Simulation']['base_unit']
        f_conv = f_phys / c_conv
        self.s4.SetFrequency(f_conv)
        E_mag = self._get_incident_amplitude()
        # E_mag = self._get_incident_amplitude_anna()
        polar = self.conf['Simulation']['params']['polar_angle']['value']
        azimuth = self.conf['Simulation']['params']['azimuthal_angle']['value']
        # To define circularly polarized light, basically just stick a j
        # (imaginary number) in front of one of your components. The component
        # you choose to stick the j in front of is a matter of convention. In
        # S4, if the incident azimuthal angle is 0, p-polarization is along
        # x-axis. Here, we choose to make the y-component imaginary. The
        # handedness is determined both by the component you stick the j in
        # front of and the sign of the imaginary component. In our convention,
        # minus sign means rhcp, plus sign means lhcp. To be circularly
        # polarized, the magnitudes of the two components must be the same.
        # This means E-field vector rotates clockwise when observed from POV of
        # source. Left handed = counterclockwise.
        # TODO: This might not be properly generalized to handle
        # polarized light if the azimuth angle IS NOT 0. Might need some extra
        # factors of cos/sin of azimuth to gen proper projections onto x/y axes
        polarization = self.conf['Simulation']['polarization']
        if polarization == 'rhcp':
            # Right hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(0, -E_mag),
                                           pAmplitude=complex(E_mag, 0))
        elif polarization == 'lhcp':
            # Left hand circularly polarized
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(0, E_mag),
                                           pAmplitude=complex(E_mag, 0))
        elif polarization == 'lpx':
            # Linearly polarized along x axis (TM polarixation)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(0, 0),
                                           pAmplitude=complex(E_mag, 0))
        elif polarization == 'lpy':
            # Linearly polarized along y axis (TE polarization)
            self.s4.SetExcitationPlanewave(IncidenceAngles=(polar, azimuth),
                                           sAmplitude=complex(E_mag, 0),
                                           pAmplitude=complex(0, 0))
        else:
            raise ValueError('Invalid polarization specification')

    def build_device(self):
        """Build the device geometry"""

        # First define all the materials
        for mat, mat_path in self.conf['Materials'].items():
            eps = self._get_epsilon(mat_path)
            self.s4.SetMaterial(Name=mat, Epsilon=eps)
        self.s4.SetMaterial(Name='vacuum', Epsilon=complex(1, 0))
        # We need to properly sort our layers because order DOES matter. Light
        # will be incident upon the first layer specified
        for layer, ldata in sorted(self.conf['Layers'].items(),
                                   key=lambda tup: tup[1]['order']):
            self.log.debug('Building layer: %s' % layer)
            self.log.debug('Layer Order %i' % ldata['order'])
            base_mat = ldata['base_material']
            layer_t = ldata['params']['thickness']['value']
            self.s4.AddLayer(Name=layer, Thickness=layer_t,
                             S4_Material=base_mat)
            if 'geometry' in ldata:
                self.log.debug('Building geometry in layer: {}'.format(layer))
                for shape, sdata in sorted(ldata['geometry'].items(), key=lambda tup: tup[1]['order']):
                    self.log.debug('Building object {} of type {} at order'
                                  ' {}'.format(shape, sdata['type'], sdata['order']))
                    shape_mat = sdata['material']
                    if sdata['type'] == 'circle':
                        rad = sdata['radius']
                        cent = sdata['center']
                        coord = (cent['x'], cent['y'])
                        self.s4.SetRegionCircle(S4_Layer=layer, S4_Material=shape_mat, Center=coord,
                                                Radius=rad)
                    else:
                        raise NotImplementedError(
                            'Shape %s is not yet implemented' % sdata['type'])

    def get_height(self):
        """Get the total height of the device"""

        height = 0
        for layer, ldata in self.conf['Layers'].items():
            layer_t = ldata['params']['thickness']['value']
            height += layer_t
        return height

    def set_lattice(self, period):
        """Updates the S4 simulation object with a new array period"""
        numbasis = self.conf['Simulation']['params']['numbasis']['value']
        self.s4 = S4.New(Lattice=((period, 0), (0, period)), NumBasis=numbasis)

    def set_basis(self, numbasis):
        """Updates the S4 simulation object with a new set of basis terms"""
        period = self.conf['Simulation']['params']['array_period']['value']
        self.s4 = S4.New(Lattice=((period, 0), (0, period)), NumBasis=numbasis)

    def update_thicknesses(self):
        """Updates all layer thicknesses without rebuilding the device. This
        allows reuse of any layer eigenmodes already computed and utilizes a
        fundamental efficiency of the RCWA solver"""
        for layer, ldata in self.conf['Layers'].items():
            thickness = ldata['params']['thickness']['value']
            self.s4.SetLayerThickness(S4_Layer=layer, Thickness=thickness)

    #  @ph.timecall
    def _compute_fields(self):
        """Constructs and returns a 2D numpy array of the vector electric
        field. The field components are complex numbers"""
        self.log.debug('Computing fields ...')
        x_samp = self.conf['Simulation']['x_samples']
        y_samp = self.conf['Simulation']['y_samples']
        z_samp = self.conf['Simulation']['z_samples']
        max_depth = self.conf['Simulation']['max_depth']
        if max_depth:
            self.log.debug('Computing up to depth of {} '
                          'microns'.format(max_depth))
            zvec = np.linspace(0, max_depth, z_samp)
        else:
            self.log.debug('Computing for entire device')
            height = self.get_height()
            zvec = np.linspace(0, height, z_samp)
        Ex = 0j*np.zeros((z_samp, x_samp, y_samp))
        Ey = 0j*np.zeros((z_samp, x_samp, y_samp))
        Ez = 0j*np.zeros((z_samp, x_samp, y_samp))
        for zcount, z in enumerate(zvec):
            E, H = self.s4.GetFieldsOnGrid(z=z, NumSamples=(x_samp, y_samp),
                                           Format='Array')
            for xcount, xval in enumerate(E):
                for ycount, yval in enumerate(xval):
                    Ex[zcount, xcount, ycount] = yval[0]
                    Ey[zcount, xcount, ycount] = yval[1]
                    Ez[zcount, xcount, ycount] = yval[2]
        self.log.debug('Finished computing fields!')
        return Ex, Ey, Ez

    def get_field(self):
        if self.conf['General']['adaptive_convergence']:
            Ex, Ey, Ez, numbasis, conv = self.adaptive_convergence()
            self.data.update({'Ex':Ex,'Ey':Ey,'Ez':Ez})
            self.converged = (conv, numbasis)
        else:
            Ex, Ey, Ez = self._compute_fields()
            self.data.update({'Ex':Ex,'Ey':Ey,'Ez':Ez})

    def get_fluxes(self):
        """
        Get the fluxes at the top and bottom of each layer. This is a surface
        integral of the component of the Poynting flux perpendicular to this
        x-y plane of the interface, and have forward and backward componenets.
        Returns a dict where the keys are the layer name and the values are a
        length 2 tuple with the forward component first and the backward
        component second. The components are complex numbers
        """
        self.log.debug('Computing fluxes ...')
        for layer, ldata in self.conf['Layers'].items():
            self.log.debug('Computing fluxes through layer: %s' % layer)
            # This gets flux at top of layer
            forw, back = self.s4.GetPowerFlux(S4_Layer=layer)
            self.flux_dict[layer] = (forw, back)
            # This gets flux at the bottom
            offset = ldata['params']['thickness']['value']
            forw, back = self.s4.GetPowerFlux(S4_Layer=layer, zOffset=offset)
            key = layer + '_bottom'
            self.flux_dict[key] = (forw, back)
        if self.conf['General']['save_as'] == 'npz':
            self.data['fluxes'] = self.flux_dict
        self.log.debug('Finished computing fluxes!')
        return self.flux_dict

    def get_dielectric_profile(self):
        """
        Gets the dielectric profile throughout the device. This is useful for
        determining the resolution of the dielectric profile used by S4. It uses
        the same number of sampling points as specified in the config file for
        retrieiving field data.
        """
        self.log.debug('Computing dielectric profile ...')
        period =  self.conf['Simulation']['params']['array_period']['value']
        x_samp = self.conf['Simulation']['x_samples']
        y_samp = self.conf['Simulation']['y_samples']
        z_samp = self.conf['Simulation']['z_samples']
        height = self.get_height()
        x_vec = np.linspace(0, period, x_samp)
        y_vec = np.linspace(0, period, y_samp)
        z_vec = np.linspace(0, height, z_samp)
        xv, yv, zv = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')
        eps_mat = np.zeros((z_samp, x_samp, y_samp), dtype=np.complex128)
        # eps_mat = np.zeros((z_samp, x_samp, y_samp))
        for ix in range(x_samp):
            for iy in range(y_samp):
                for iz in range(z_samp):
                    eps_val =  self.s4.GetEpsilon(xv[ix, iy, iz],
                                                  yv[ix, iy, iz],
                                                  zv[ix, iy, iz])
                    eps_mat[iz, ix, iy] = eps_val
        self.data.update({'dielectric_profile': eps_mat})
        self.log.debug('Finished computing dielectric profile!')

    # def get_integrals(self):
    #     self.log.debug('Computing volume integrals')
    #     integrals = {}
    #     for layer, ldata in self.conf['Layers'].items():
    #         self.log.debug('Computing integral through layer: %s' % layer)
    #         result = self.s4.GetLayerVolumeIntegral(S4_Layer=layer, Quantity='E')
    #         self.log.debug('Integral = %s', str(result))
    #         integrals[layer] = result
    #     print(integrals)
    #     return integrals

    def save_data(self):
        """Saves the self.data dictionary to an npz file. This dictionary
        contains all the fields and the fluxes dictionary"""

        if self.conf['General']['save_as'] == 'npz':
            self.log.debug('Saving fields to NPZ')
            if self.conf['General']['adaptive_convergence']:
                if self.converged[0]:
                    out = os.path.join(self.dir, 'converged_at.txt')
                else:
                    out = os.path.join(self.dir, 'not_converged_at.txt')
                self.log.debug('Writing convergence file ...')
                with open(out, 'w') as outf:
                    outf.write('{}\n'.format(self.converged[1]))
            out = os.path.join(self.dir, self.conf["General"]["base_name"])
            # Compression adds a small amount of time. The time cost is
            # nonlinear in the file size, meaning the penalty gets larger as the
            # field grid gets finer. However, the storage gains are enormous!
            # Compression brought the file from 1.1G to 3.9M in a test case.
            # I think the compression ratio is so high because npz is a binary
            # format, and all compression algs benefit from large sections of
            # repeated characters
            np.savez_compressed(out, **self.data)
        elif self.conf['General']['save_as'] == 'hdf5':
            compression = self.conf['General']['compression']
            if compression:
                filter_obj = tb.Filters(complevel=8, complib='blosc')
            gpath = '/sim_'+self.id[0:10]
            for name, arr in self.data.iteritems():
                self.log.debug("Saving array %s", name)
                if compression:
                    self.hdf5.create_carray(gpath, name, createparents=True,
                                       atom=tb.Atom.from_dtype(arr.dtype),
                                       obj=arr, filters=filter_obj)
                else:
                    self.hdf5.create_array(gpath, name, createparents=True,
                                      atom=tb.Atom.from_dtype(arr.dtype),
                                      obj=arr)
            table = self.hdf5.create_table(gpath, 'fluxes', description=LayerFlux,
                                      expectedrows=len(list(self.conf['Layers'].keys())),
                                      createparents=True)
            row = table.row
            for layer, (forward, backward) in self.flux_dict.iteritems():
                row['layer'] = layer
                row['forward'] = forward
                row['backward'] = backward
                row.append()
            table.flush()
                    # # Save the field arrays
                    # self.log.debug('Saving fields to HDF5')
                    # path = '/sim_'+self.id[0:10]
                    # for name, arr in self.data.iteritems():
            #     self.log.debug("Saving array %s", name)
            #     tup = ('create_array', (path, name),
            #            {'compression': self.conf['General']['compression'],
            #             'createparents': True, 'obj': arr,
            #             'atom': tb.Atom.from_dtype(arr.dtype)})
            #     self.q.put(tup, block=True)
            # # Save the flux dict to a table
            # self.log.debug('Saving fluxes to HDF5')
            # self.log.debug(self.flux_dict)
            # tup = ('create_flux_table', (self.flux_dict, path, 'fluxes'),
            #        {'createparents': True,
            #         'expectedrows': len(list(self.conf['Layers'].keys()))})
            # self.q.put(tup, block=True)
        else:
            raise ValueError('Invalid file type specified in config')

    def save_conf(self):
        """
        Saves the simulation config object to a file
        """
        if self.conf['General']['save_as'] == 'npz':
            self.log.debug('Saving conf to YAML file')
            self.conf.write(os.path.join(self.dir, 'sim_conf.yml'))
        elif self.conf['General']['save_as'] == 'hdf5':
            self.log.debug('Saving conf to HDF5 file')
            self.conf.write(os.path.join(self.dir, 'sim_conf.yml'))
            path = '/sim_{}'.format(self.id[0:10])
            try:
                node = self.hdf5.get_node(path)
            except tb.NoSuchNodeError:
                self.log.warning('You need to create the group for this '
                'simulation before you can set attributes on it. Creating now')
                node = self.hdf5.create_group(path)
            node._v_attrs['conf'] = self.conf.dump()
            # attr_name = 'conf'
            # tup = ('save_attr', (self.conf.dump(), path, attr_name), {})
            # self.q.put(tup, block=True)
        else:
            raise ValueError('Invalid file type specified in config')

    def save_time(self):
        """
        Saves the run time for the simulation
        """
        if self.conf['General']['save_as'] == 'npz':
            self.log.debug('Saving runtime to text file')
            time_file = os.path.join(self.dir, 'time.dat')
            with open(time_file, 'w') as out:
                out.write('{}\n'.format(self.runtime))
        elif self.conf['General']['save_as'] == 'hdf5':
            self.log.debug('Saving runtime to HDF5 file')
            path = '/sim_{}'.format(self.id[0:10])
            try:
                node = self.hdf5.get_node(path)
            except tb.NoSuchNodeError:
                self.log.warning('You need to create the group for this '
                'simulation before you can set attributes on it. Creating now')
                node = self.hdf5.create_group(path)
            node._v_attrs['runtime'] = self.runtime
            # attr_name = 'runtime'
            # tup = ('save_attr', (self.runtime, path, attr_name), {})
            # self.q.put(tup, block=True)
        else:
            raise ValueError('Invalid file type specified in config')

    def calc_diff(self, fields1, fields2, exclude=False):
        """Calculate the percent difference between two vector fields"""
        # This list contains three 3D arrays corresponding to the x,y,z
        # componenets of the e field. Within each 3D array is the complex
        # magnitude of the difference between the two field arrays at each
        # spatial point squared
        diffs_sq = [np.absolute(arr1 - arr2)**2 for arr1, arr2 in zip(fields1, fields2)]
        # Sum the squared differences of each component
        mag_diffs = sum(diffs_sq)
        # Now compute the norm(E)^2 of the comparison sim at each sampling
        # point
        normsq = sum([np.absolute(field)**2 for field in fields1])
        # We define the percent difference as the ratio of the sums of the
        # difference vector magnitudes to the comparison field magnitudes,
        # squared rooted.
        # TODO: This seems like a somewhat shitty metric that washes out any
        # large localized deviations. Should test other metrics
        diff = np.sqrt(np.sum(mag_diffs) / np.sum(normsq))
        self.log.debug('Percent difference = {}'.format(diff))
        return diff

    def adaptive_convergence(self):
        """Performs adaptive convergence by checking the error between vector
        fields for simulations with two different numbers of basis terms.
        Returns the field array, last number of basis terms simulated, and a
        boolean representing whether or not the simulation is converged"""
        self.log.debug('Beginning adaptive convergence procedure')
        start_basis = self.conf['Simulation']['params']['numbasis']['value']
        basis_step = self.conf['General']['basis_step']
        ex, ey, ez = self._compute_fields()
        max_diff = self.conf['General']['max_diff']
        max_iter = self.conf['General']['max_iter']
        percent_diff = 100
        iter_count = 0
        while percent_diff > max_diff and iter_count < max_iter:
            new_basis = start_basis + basis_step
            self.log.debug('Checking error between {} and {} basis'
                          ' terms'.format(start_basis, new_basis))
            self.set_basis(new_basis)
            self.build_device()
            self.set_excitation()
            ex2, ey2, ez2 = self._compute_fields()
            percent_diff = self.calc_diff([ex, ey, ex], [ex2, ey2, ez2])
            start_basis = new_basis
            ex, ey, ez = ex2, ey2, ez2
            iter_count += 1
        if percent_diff > max_diff:
            self.log.warning('Exceeded maximum number of iterations')
            return ex2, ey2, ez2, new_basis, False
        else:
            self.log.debug('Converged at {} basis terms'.format(new_basis))
            return ex2, ey2, ez2, new_basis, True

    def mode_solve(self, update=False):
        """Find modes of the system. Supposedly you can get the resonant modes
        of the system from the poles of the S matrix determinant, but I
        currently can't really make any sense of this output"""
        if not update:
            self.configure()
            self.build_device()
        else:
            self.update_thicknesses()
        mant, base, expo = self.s4.GetSMatrixDeterminant()
        self.log.debug('Matissa: %s'%str(mant))
        self.log.debug('Base: %s'%str(base))
        self.log.debug('Exponent: %s'%str(expo))
        res = mant*base**expo
        self.log.debug('Result: %s'%str(res))

    def save_all(self, update=False):
        """Gets all the data for this similation by calling the relevant class
        methods. Basically just a convenient wrapper to execute all the
        functions defined above"""
        # TODO: Split this into a get_all and save_all function. Will give more
        # granular sense of timing and also all getting data without having to
        # save
        start = time.time()
        if not update:
            self.configure()
            self.build_device()
            self.set_excitation()
        else:
            self.update_thicknesses()
        self.get_field()
        self.get_fluxes()
        if self.conf['General']['dielectric_profile']:
            self.get_dielectric_profile()
        self.open_hdf5()
        self.save_data()
        self.save_conf()
        end = time.time()
        self.runtime = end - start
        self.save_time()
        self.hdf5.flush()
        self.hdf5.close()
        self.log.info('Simulation {} completed in {:.2}'
                      ' seconds!'.format(self.id[0:10], self.runtime))
        return
