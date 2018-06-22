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
@click.option('-t', '--table_path', default='/', help=h2, show_default=True)
@click.option('-n', '--table_name', default='simulations', help=h3,
              show_default=True)
@click.option('-s', '--skip_keys', default=['General', 'Materials'], help=h4,
              show_default=True)
def preprocess(template, db, params, table_path, table_name, skip_keys):
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
    pp.add_to_database(db_path=db, tb_path=table_path, tb_name=table_name,
                       skip_keys=skip_keys)
    click.secho('Preprocessing complete!', fg='green')


h1 = "The directory to store all the simulation outputs in. Defaults to " \
     "the same directory as the provided config file"


@optics.command()
@click.argument('config', type=exist_read_path)
@click.option('-o', '--output_dir', type=exist_read_dir, default=None, help=h1)
def run(config, output_dir):
    """
    Run a single simulation.

    CONFIG must be a path to a config file for a single simulation in valid
    YAML format. No preprocessing is performed on this file. If this file was
    generated by the preprocessor, you are all set. However, you could choose
    to write a config file by hand and use it to run simulations if you like
    """

    import nanowire.optics.simulate as simul
    import nanowire.preprocess as prep

    conf = prep.Config.fromFile(config)
    if output_dir is None:
        output_dir = os.path.dirname(config)
    simul.run_sim(conf, output_dir)
    return None


h0 = "Base directory containing all configs. If not specified, defaults to " \
     "the directory of the input config"
h1 = "Only run simulations whose parameters match the provided query. The " \
     "query may contained double-underscore separated paths to entries in " \
     "the simulation config. For example\n" \
     "-f '(Simulation__numbasis == 200) & (Simulation__frequency*2 < 5e14)'"
h2 = "Update the electric field arrays with new z samples without " \
     "overwriting the old data. Can only upate the sampling in z " \
     "because there is no way to update in x-y without destroying " \
     "the regularity of the grid"
h3 = "Path to the parent group of the table inside the HDF5 file used for " \
     "storing configurations"""
h4 = "Name of the table at the end of table_path for storing configurations"


@optics.command()
@click.argument('config', type=exist_read_path)
@click.argument('db', type=exist_read_path)
@click.option('-b', '--base_dir',
              callback=lambda ctx, p, v: os.path.dirname(ctx.params['db']) if not v else v,
              type=exist_read_dir, help=h0)
@click.option('-p', '--params', default=None, type=exist_read_path,
              help="Optional params for the config file parser")
@click.option('-q', '--query', type=str, help=h1)
@click.option('-u', '--update', default=False, is_flag=True, help=h2)
@click.option('-t', '--table_path', default='/', help=h3, show_default=True)
@click.option('-n', '--table_name', default='simulations', help=h4,
              show_default=True)
@click.option('-v', '--log_level', help="Set verbosity of logging",
              type=click.Choice(['info', 'debug', 'warning', 'critical', 'error']),
              default='info')
def run_all(config, db, base_dir, params, query, update, table_path,
            table_name, log_level):
    """
    Run all the simulations located beneath BASE_DIR.

    The directory tree beneath BASE_DIR is traversed recursively from the top
    down and all the config files beneath it are collected. A simulation is run
    for each config file found, and the output of each simulation is stored in
    the same directory as the corresponding config file.

    Can optionally configure how the manager runs via a config file, command
    line parameters, or a combination of the two. The config file will be
    treated as a template and can thus contain any special templating syntax
    """

    import nanowire.optics.simulate as simul
    import nanowire.preprocess as prep

    processor = prep.Preprocessor(config)
    parsed_dicts = processor.generate_configs(params=params)
    if len(parsed_dicts) != 1:
        raise ValueError('Must have only 1 set of unique parameters for the '
                         'manager configuration')
    conf = parsed_dicts[0]
    manager = simul.SimulationManager(conf, log_level=log_level.upper())
    manager.load_confs(db, base_dir=base_dir, query=query,
                       table_path=table_path, table_name=table_name)
    if update:
        manager.run(func=simul.update_sim, load=True)
    else:
        manager.run()