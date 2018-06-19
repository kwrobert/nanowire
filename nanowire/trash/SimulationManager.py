"""
This is the old SimulationManager class code. I used this optimize the
absorption in the arrays. Probably some useful stuff in here so I'm keeping it
around, but the whole thing needs a serious redesign
"""

class SimulationManager:

    """
    A class to manage running many simulations either in series or in parallel,
    collect and emit logs, write out data to files, etc
    """

    def __init__(self, gconf, base_dir='', log_level='INFO'):
        self.gconf = gconf
        lfile = osp.join(self.gconf['General']['base_dir'],
                             'logs/sim_manager.log')
        try:
            log_level = self.gconf['General']['log_level']
        except KeyError:
            pass
        # self.log = configure_logger(level=log_level, console=True,
        #                             logfile=lfile, name=__name__)
        self.log = logging.getLogger(__name__)
        self.sim_confs = []

    def load_confs(self, base_dir=''):
        """
        Collect all the simulations beneath the base of the directory tree
        """

        sims = []
        failed_sims = []
        # Find the data files and instantiate Config objects
        if not base_dir:
            base_dir = osp.expandvars(self.gconf['General']['base_dir'])
        self.log.info(base_dir)
        for root, dirs, files in os.walk(base_dir):
            conf_path = osp.join(root, 'sim_conf.yml')
            if 'sim_conf.yml' in files and 'sim.hdf5' in files:
                self.log.info('Gather sim at %s', root)
                conf_obj = Config(conf_path)
                sims.append(conf_obj)
            elif 'sim_conf.yml' in files:
                conf_obj = Config(conf_path)
                self.log.error('Sim missing its data file: %s',
                               conf_obj.conf['General']['sim_dir'])
                failed_sims.append(conf_obj)
        self.sim_confs = sims
        self.failed_sims = failed_sims
        if not sims:
            self.log.error('Unable to find any successful simulations')
            raise RuntimeError('Unable to find any successful simulations')
        return sims, failed_sims

    def run_parallel(self, *args, func=run_sim, **kwargs):
        """
        Run the loaded configurations in parallel by applying the provided func
        (default run_sim) to each dict using the Python multiprocessing library
        and the apply_async function. We do this instead of applying to an
        actual Simulator object because the Simulator objects are not pickeable
        and thus cannot be parallelized by the multiprocessing lib
        """

        if self.gconf['General']['execution'] == 'serial':
            self.log.info('Executing sims serially')
            for conf in self.sim_confs:
                func(conf)
        elif self.gconf['General']['execution'] == 'parallel':
            self.log.info('Total sims to execute: %i', len(self.sim_confs))
            num_procs = self.gconf['General']['num_cores']
            if num_procs > len(self.sim_confs):
                num_procs = len(self.sim_confs)
            self.log.info('Executing sims in parallel using %s cores ...', str(num_procs))
            # pool = LoggingPool(processes=num_procs)
            # pool = mp.Pool(processes=num_procs, maxtasksperchild=2)
            pool = mp.Pool(processes=num_procs)
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
        elif self.gconf['General']['execution'] == 'dispy':
            self.log.info('Executing jobs using dispy cluster')
            self.dispy_submit()
        self.log.info('Finished executing jobs!')

    def dispy_submit(self):
        import time
        log = logging.getLogger(__name__)
        log.info("Beginning dispy submit procedure")
        try:
            nodes = self.gconf['General']['nodes']
            ip = self.gconf['General']['ip_addr']
        except KeyError:
            log.error("Need to specify 'nodes' and 'ip_addr' entries in "
                      "General section")
        if ip == 'auto':
            pub_iface = get_public_iface()
            ip = get_public_ip(pub_iface)
        node_allocs = []
        for node in nodes:
            # If its not a string, hopefully it's a length 2 list or tuple
            # with (str, int) = (ip_addr, port_number)
            if not isinstance(node, str):
                node_allocs.append(dispy.NodeAllocate(node[0], port=node[1]))
            # Use default port locally
            elif node == 'local':
                node_allocs.append(dispy.NodeAllocate(ip))
            else:
                node_allocs.append(dispy.NodeAllocate(node))
        cluster = dispy.JobCluster(run_sim_dispy,
                                   dest_path=time.strftime('%Y-%m-%d'),
                                   cleanup=True,
                                   # cleanup=False, loglevel=dispy.logger.DEBUG,
                                   nodes=node_allocs, ip_addr=ip)
        # Wait until we connect to at least one node
        while len(cluster.status().nodes) == 0:
            time.sleep(1)
        cluster.print_status()
        jobs = {}
        for i, conf in enumerate(self.sim_confs):
            job = cluster.submit(conf)
            job.id = i
            jobs[i] = job
            stat = DISPY_LOOKUP[job.status]
            log.info('Job ID: {}, Status: {}'.format(i, stat))
        while jobs:
            toremove = []
            for job_id, job in jobs.items():
                status = DISPY_LOOKUP[job.status]
                if (job.status == dispy.DispyJob.Finished
                        or job.status in (dispy.DispyJob.Terminated, dispy.DispyJob.Cancelled,
                                          dispy.DispyJob.Abandoned)):
                    toremove.append(job_id)
            for job_id in toremove:
                job = jobs[job_id]
                stat = DISPY_LOOKUP[job.status]
                time = job.result
                if job.status == dispy.DispyJob.Terminated:
                    log.info('Job ID: %i, Status: %s, Exception:\n%s',
                             job_id, stat, job.exception)
                else:
                    log.info('Job ID: %i, Status: %s, Runtime: %f s'
                             ', Host: %s', job_id, stat, time, job.ip_addr)
                del jobs[job_id]
        return None

    def run(self, *args, filter_dict={}, load=False, func=run_sim, **kwargs):
        """
        The main run methods that decides what kind of simulation to run based
        on the provided config objects
        """

        if not self.gconf.optimized:
            # Get all the sims
            if load:
                self.load_confs()
            else:
                self.make_confs()

            if filter_dict:
                for k, vals in filter_dict.items():
                    par = [ks for ks in k.split('.')]
                    vals = list(map(type(self.sim_confs[0][par]), vals))
                    self.sim_confs = [c for c in self.sim_confs if c[par] in vals]
            self.log.info("Executing job campaign")
            self.execute_jobs(func=func, *args, **kwargs)
        elif self.gconf.optimized:
            self.run_optimization()
        else:
            self.log.error('Unsupported configuration for a simulation run. Not a '
                           'single sim, sweep, or optimization. Make sure your sweeps are '
                           'configured correctly, and if you are running an optimization '
                           'make sure you do not have any sorting parameters specified')
