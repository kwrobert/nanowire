import os
import click
# from nanowire.optics.simulate import SimulationManager, run_sim, update_sim
# from nanowire.optics.utils.config import Config

exist_read_path = click.Path(exists=True, resolve_path=True, readable=True,
                             file_okay=True, dir_okay=False)
exist_read_dir = click.Path(exists=True, resolve_path=True, readable=True,
                            writable=True, dir_okay=True, file_okay=False)
cli_path = click.Path()


@click.group()
def optics():
    """
    The command line interface to the nanowire optics library.

    Type optics [SUBCOMMAND] --help for help with the various subcommands
    """


h1 = "Path to the parameters file. Used for providing additional parameters " \
     "to the preprocessor and performing parameter sweeps"
h2 = "Path to the parent group of the table inside the HDF5 file used for " \
     "storing configurations"""
h3 = "Name of the table at the end of table_path for storing configurations"
h4 = "A list of keys within the configs to skip when generating the config ID"


@optics.command()
@click.argument('template', type=exist_read_path)
@click.argument('db', type=cli_path)
@click.option('-p', '--params', default=None, type=exist_read_path, help=h1)
@click.option('-t', '--table-path', default='/', help=h2, show_default=True)
@click.option('-n', '--table-name', default='simulations', help=h3,
              show_default=True)
@click.option('-s', '--skip-keys', default=['General', 'Materials'], help=h4,
              show_default=True)
@click.option('--update/--no-update', default=False,
              help="Update IDs if they already exist in the DB")
def preprocess(template, db, params, table_path, table_name, skip_keys, update):
    """
    Preprocess the template file located at path TEMPLATE.

    The preprocessor consumes a YAML template and an optional params file. For
    each unique combination of parameters in the params file (Cartesian product
    of the parameters), a Config object is generated from the template and
    stored in the HDF5 file DB, as well as on disk in a subdirectory named
    using the first 10 characters of the Config ID
    """

    import nanowire.preprocess as prep

    pp = prep.Preprocessor(template)
    click.echo('Generating configs ...')
    pp.generate_configs(params=params)
    click.echo('Writing configs to disk ...')
    pp.write_configs()
    click.echo('Writing configs to database ...')
    pp.add_to_database(db, tb_path=table_path, tb_name=table_name,
                       skip_keys=skip_keys, update=update)
    click.secho('Preprocessing complete!', fg='green')


h1 = "The directory to store all the simulation outputs in. Defaults to " \
     "the same directory as the provided config file"


@optics.command()
@click.argument('config', type=exist_read_path)
@click.option('-o', '--output-dir', type=exist_read_dir, default=None, help=h1)
def run(config, output_dir):
    """
    Run a single simulation.

    CONFIG must be a path to a config file for a single simulation in valid
    YAML format. No preprocessing is performed on this file. If this file was
    generated by the preprocessor, you are all set. However, you could choose
    to write a config file by hand and use it to run simulations if you like
    """

    import nanowire.optics.simulate as simul
    from nanowire.utils.config import Config

    conf = Config.fromFile(config)
    if output_dir is None:
        output_dir = os.path.dirname(config)
    simul.run_sim(conf, output_dir)
    return None

h1 = "The directory to dump all configs into. Defaults to " \
     "the same directory as the provided DB file"
@optics.command()
@click.argument('db', type=exist_read_path)
@click.option('-o', '--output-dir', type=exist_read_dir, default=None, help=h1)
@click.option('-i', '--id', type=click.STRING , multiple=True,
              help="Only dump these IDs. Can be specified multiple times")
def dump_configs(db, output_dir, id):
    """
    Dump all configs stored in DB to files

    DB must be a path to an HDF5 file containing the database of simulation
    configs. All configs in the database are dumped to YAML files on disk, with
    the files stored in subdirectories named using the first 10 characters of
    the config ID
    """

    from nanowire.utils.config import dump_configs

    if output_dir is None:
        output_dir = os.path.dirname(db)
    def get_fname(row):
        ID = row['ID'].decode()
        outdir = os.path.join(ID[0:10], 'sim_conf.yml')
        return outdir
    click.secho('Dumping configs ...')
    paths, db = dump_configs(db, outdir=output_dir, IDs=id,
                             fname=get_fname)
    db.close()
    return None

h0 = "Base directory that all simulations will dump their output data to. " \
     "If not specified, defaults to the directory of the input database"
h1 = "Only run simulations whose parameters match the provided query. The " \
     "query may contained double-underscore separated paths to entries in " \
     "the simulation config. For example\n" \
     "-f '(Simulation__numbasis == 200) & (Simulation__frequency*2 < 5e14)'"
h3 = "Path to the parent group of the table inside the HDF5 file used for " \
     "storing configurations"""
h4 = "Name of the table at the end of table_path for storing configurations"


@optics.command()
@click.argument('db', type=exist_read_path)
@click.argument('exec_mode',
                type=click.Choice(['serial', 'parallel', 'dispy']))
@click.option('-b', '--base-dir',
              callback=lambda ctx, p, v: os.path.dirname(ctx.params['db']) if not v else v,
              type=exist_read_dir, help=h0)
@click.option('-p', '--params', default=None, type=exist_read_path,
              help="Optional params for the config file parser")
@click.option('-q', '--query', type=click.STRING, help=h1)
@click.option('-u', '--update', default=False, is_flag=True, help=h2)
@click.option('-t', '--table-path', default='/', help=h3, show_default=True)
@click.option('-m', '--table-name', default='simulations', help=h4,
              show_default=True)
@click.option('-n', '--nodes', type=click.STRING, multiple=True,
              help="Nodes to run on. Specify multiple times for multiple nodes")
@click.option('-i', '--ip-addr', type=click.STRING,
              help="IP of local host for use with dispy")
@click.option('-j', '--num-cores', type=click.INT,
              help="Number of cores to use if running in parallel")
@click.option('-v', '--log-level',
              type=click.Choice(['info', 'debug', 'warning', 'critical', 'error']),
              default='info',
              help="Set verbosity of logging")

def run_all(db, exec_mode, base_dir, params, query, update, table_path,
            table_name, nodes, ip_addr, num_cores, log_level):
    """
    Run all simulations matching QUERY located in the HDF5 DB

    Collects all simulations inside the HDF5 database DB matching query string
    QUERY and runs them using the specified MODE.
    """

    import nanowire.optics.simulate as simul

    manager = simul.SimulationManager(db, nodes=nodes, ip=ip_addr,
                                      num_cores=num_cores,
                                      log_level=log_level.upper())
    manager.load_confs(base_dir=base_dir, query=query, table_path=table_path,
                       table_name=table_name)
    if update:
        manager.run(exec_mode, func=simul.update_sim, load=True)
    else:
        manager.run(exec_mode)

@optics.command()
@click.argument('db', type=exist_read_path)
@click.argument('template', type=exist_read_path)
@click.option('-b', '--base-dir',
              callback=lambda ctx, p, v: os.path.dirname(ctx.params['db']) if not v else v,
              type=exist_read_dir, help=h0)
@click.option('-p', '--params', default=None, type=exist_read_path,
              help="Optional params for the config file parser")
@click.option('-q', '--query', type=str, help=h1)
@click.option('-t', '--table-path', default='/', help=h3, show_default=True)
@click.option('-n', '--table-name', default='simulations', help=h4,
              show_default=True)
@click.option('--crunch/--no-crunch', default=True,
              help="Peform/do not perform crunching operations. Useful when "
                   "data has already been crunched but new plots need to be "
                   "generated")
@click.option('--gcrunch/--no-gcrunch', default=True,
              help="Peform/do not perform group crunching operations. Useful "
                   "when data has already been crunched but new plots need to "
                   "be generated")
@click.option('--plot/--no-plot', default=True,
              help="Peform/do not perform plotting operations. Useful when "
                   "you only want to crunch your data without plotting")
@click.option('--gplot/--no-gplot', default=True,
              help="Peform/do not perform global plotting operations. Useful "
                   "when you only want to crunch your data without plotting")
@click.option('-gb', '--group-by', type=click.STRING,
              help="The parameter you would like to group simulations by, "
                   "specified as a forward slash separated path to the key "
                   "in the config as: path/to/key/value")
@click.option('-ga', '--group-against', type=click.STRING,
              help="The parameter you would like to group simulations "
                   "against, specified as a forward slash separated path to "
                   "the key in the config as: path/to/key/value")
@click.option('-j', '--num-cores', type=click.INT, default=0,
              help="Number of cores to use if running in parallel")
@click.option('--print-ids', is_flag=True, default=False,
              help="Print IDs of simulations to be processed, without running "
                   "anything")
@click.option('-v', '--log-level', help="Set verbosity of logging",
              type=click.Choice(['info', 'debug', 'warning', 'critical', 'error']),
              default='info')
def postprocess(db, template, base_dir, params, query, table_path, table_name,
                crunch, gcrunch, plot, gplot, group_by, group_against,
                print_ids, num_cores, log_level):
    """
    Postprocess all simulations matching QUERY located in the HDF5 DB

    Collects all simulations inside the HDF5 database DB matching query string
    QUERY and postprocesses them according to the plan specified in CONFIG.
    """

    import nanowire.optics.postprocess as post

    # click.echo('Parsing config file ...')
    # processor = prep.Preprocessor(config)
    # parsed_dicts = processor.generate_configs(skip_keys=[], params=params)
    # if len(parsed_dicts) != 1:
    #     raise ValueError('Must have only 1 set of unique parameters for the '
    #                      'manager configuration')
    # conf = parsed_dicts[0]
    proc = post.Processor(db, template, base_dir=base_dir, num_cores=num_cores)
    proc.load_confs(base_dir=base_dir, query=query,
                    table_path=table_path, table_name=table_name)
    if group_against:
        proc.group_against(group_against)
    elif group_against:
        proc.group_by(group_by)
    else:
        click.secho('WARNING: No grouping specified', fg='yellow')
    if print_ids:
        for conf in proc.sim_confs:
            print(conf.ID)
        return
    else:
        proc.process(crunch=crunch, gcrunch=gcrunch, plot=plot, gplot=gplot,
                     grouped_against=group_against, grouped_by=group_by)
