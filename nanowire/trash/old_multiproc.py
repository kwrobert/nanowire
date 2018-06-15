"""
This is all the old multiprocessing code.

In this approach all the simulation subprocesses pushed their results onto a
Queue, and there was a single Writer thread that would pull results off the
Queue and write them into a single HDF5 file.

This worked, but consumed a ton of memory and wasn't very performant. It turns
out it's not a great idea to push really large numpy arrays through a queue.
The Queue always got backed up and the writer thread couldn't keep up with the
data input rate

I spent a lot of time getting this working, so I can't bring myself to just
delete the code even though I stopped using it a long time ago.
"""

class FileMerger(threading.Thread):

    def __init__(self, q, write_dir='', group=None, target=None, name=None):
        super(FileMerger, self).__init__(group=group, target=target, name=name)
        self.q = q
        outpath = osp.join(write_dir, 'data.hdf5')
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
                    # abssubdir, subfname = osp.split(path)
                    # subdir = osp.basename(abssubdir)
                    # where = '{}:{}'.format(osp.join(subdir, subfname),
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
        outpath = osp.join(write_dir, 'data.hdf5')
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
        for layer, (forward, backward) in flux_dict.items():
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

    def __init__(self):
        # ... probably other stuff
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
        samps = [self.gconf['General'][s] for s in ('x_samples',
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

    def execute_jobs(self, *args, func=run_sim, **kwargs):
        """
        Given a list of configuration dictionaries, run them either serially or
        in parallel by applying the provided func (default run_sim) to each
        dict. We do this instead of applying to an actual Simulator object
        because the Simulator objects are not pickeable and thus cannot be
        parallelized by the multiprocessing lib
        """

        if self.gconf['General']['execution'] == 'serial':
            self.log.info('Executing sims serially')
            # Make the write queue, then instanstiate and run the thread that
            # pulls data from the queue and writes to the HDF5 file
            # if self.gconf['General']['save_as'] == 'hdf5':
            #     self.make_listener()
            # else:
            #     self.make_queue()
            for conf in self.sim_confs:
                func(conf, q=self.write_queue)
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
            self.log.info('Total sims to execute: %i', len(self.sim_confs))
            num_procs = self.gconf['General']['num_cores']
            if num_procs > len(self.sim_confs):
                num_procs = len(self.sim_confs)
            self.log.info('Executing sims in parallel using %s cores ...', str(num_procs))
            # pool = LoggingPool(processes=num_procs)
            pool = mp.Pool(processes=num_procs, maxtasksperchild=2)
            # pool = mp.Pool(processes=num_procs)
            total_sims = len(self.sim_confs)
            remaining_sims = len(self.sim_confs)
            def callback(ind):
                callback.remaining_sims -= 1
                callback.log.info('%i out of %i simulations remaining'%(callback.remaining_sims,
                                                                callback.total_sims))
                return None
            callback.remaining_sims = remaining_sims
            callback.total_sims = total_sims
            callback.log = self.log
            results = {}
            self.log.debug('Entering try, except pool clause')
            try:
                for ind, conf in enumerate(self.sim_confs):
                    res = pool.apply_async(func, (conf, *args),
                                           {'q':self.write_queue, **kwargs},
                                           callback=callback)
                    results[ind] = res
                # self.log.debug('Closing pool')
                # pool.close()
                self.log.debug("Waiting on results")
                self.log.debug('Results before wait loop: %s',
                               str(list(results.keys())))
                # while len(results) > num_procs:
                while results:
                    inds = list(results.keys())
                    for ind in inds:
                        # We need to add this really long timeout so that
                        # subprocesses receive keyboard interrupts. If our
                        # simulations take longer than this timeout, an exception
                        # would be raised but that should never happen
                        res = results[ind]
                        self.log.debug('Sim #%i', ind)
                        if res.ready():
                            success = res.successful()
                            if success:
                                self.log.debug('Sim #%i completed successfully!', ind)
                                res.get(10)
                                self.log.debug('Done getting Sim #%i', ind)
                            else:
                                self.log.warning('Sim #%i raised exception!',
                                                 ind)
                                res.wait(10)
                                # try:
                                #     res.get(100)
                                # except Exception as e:
                                #     self.log.warning('Sim #%i raised following'
                                #                      ' exception: %s', ind,
                                #                      traceback.format_exc())
                            del results[ind]
                        else:
                            self.log.debug('Sim #%i not ready', ind)
                    self.log.debug('Cleaned results: %s',
                                   str(list(results.keys())))
                    time.sleep(1)

                    # res.wait(99999999)
                    # res.get(99999999)
                    # self.log.debug('Number of items in queue: %i',
                    #                self.write_queue.qsize())
                self.log.debug('Closing pool')
                pool.close()
                self.log.debug('Finished waiting')
                self.log.debug('Joining pool')
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
            # self.write_queue.put(None, block=True)
            # if self.reader is not None:
            #     self.log.info('Joining FileWriter thread')
            #     self.reader.join()
            # for res in results:
            #     print(res)
            #     print(res.get())
        elif self.gconf['General']['execution'] == 'dispy':
            self.log.info('Executing jobs using dispy cluster')
            self.dispy_submit()
        self.log.info('Finished executing jobs!')
