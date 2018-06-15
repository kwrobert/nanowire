"""
This is the old run_sim function that had this horrible hackiness for dealing
with thickness sweeps. I should probably delete this soon before anyone else
sees this and loses respect for me
"""

def run_sim(conf, output_dir):
    """
    Runs a single simulation given a Config object representing the
    configuration for the simulation and a string pointing to a directory the
    simulation should write its output data to.

    .. note:: It is important that the Config object has the correct structure
    for the simulation to run properly

    :param conf: A Config object representing the configuration for this
    particular simulation
    :type conf: nanowire.preprocess.config.Config
    :param output_dir: Directory the simulation will write its data to
    :type output_dir: str
    """

    log = logging.getLogger(__name__)
    start = time.time()
    sim = Simulator(copy.deepcopy(conf))
    try:
        if not sim.conf.variable_thickness:
            sim.setup()
            log.info('Executing sim %s', sim.ID[0:10])
            sim.save_all()
            # sim.mode_solve()
        else:
            log.info('Computing a thickness sweep at %s' % sim.ID[0:10])
            orig_id = sim.ID[0:10]
            # Get all combinations of layer thicknesses
            keys, combos = get_combos(sim.conf, sim.conf.variable_thickness)
            # Update base directory to new sub directory
            sim.conf['General']['base_dir'] = sim.dir
            # Set things up for the first combo
            first_combo = combos.pop()
            # First update all the thicknesses in the config. We make a copy of the
            # list because it gets continually updated in the config object
            var_thickness = sim.conf.variable_thickness
            for i, param_val in enumerate(first_combo):
                keyseq = var_thickness[i]
                sim.conf[keyseq] = param_val
            # With all the params updated we can now run substutions and
            # evaluations in the config that make have referred to some thickness
            # data, then make the subdir from the sim.ID and get the data
            sim.evaluate_config()
            sim.update_id()
            try:
                os.makedirs(sim.dir)
            except OSError:
                pass
            sim.make_logger()
            subpath = osp.join(orig_id, sim.ID[0:10])
            log.info('Computing initial thickness at %s', subpath)
            sim.save_all()
            # Now we can repeat the same exact process, but instead of rebuilding
            # the device we just update the thicknesses
            for combo in combos:
                for i, param_val in enumerate(combo):
                    keyseq = var_thickness[i]
                    sim.conf[keyseq] = param_val
                sim.update_id()
                subpath = osp.join(orig_id, sim.ID[0:10])
                log.info('Computing additional thickness at %s', subpath)
                os.makedirs(sim.dir)
                sim.save_all(update=True)
        end = time.time()
        runtime = end - start
        log.info('Simulation %s completed in %.2f seconds!', sim.ID[0:10], runtime)
        sim.clean_sim()
    except:
        trace = traceback.format_exc()
        msg = 'Sim {} raised the following exception:\n{}'.format(sim.ID,
                                                                  trace)
        log.error(msg)
        # We might encounter an exception before the logger instance for this
        # sim gets created
        try:
            sim.log.error(trace)
        except AttributeError:
            pass
    return None
