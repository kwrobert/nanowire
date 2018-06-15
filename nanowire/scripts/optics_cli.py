import os
import click
# from nanowire.optics.simulate import SimulationManager, run_sim, update_sim
# from nanowire.optics.utils.config import Config

exist_read_path = click.Path(exists=True, resolve_path=True, readable=True,
                             file_okay=True)
cli_path = click.Path()


@click.group()
def optics():
    pass

@optics.command()
h = "Path to the template to parse"
@click.argument('template', type=exist_read_path, help=h)
h = "Path to the HDF5 file to add the generated configurations to"
@click.argument('db', type=cli_path, help=h)
h = "Path to the parameters file used for performing parameter sweeps"
@click.option('-p', '--params', default=None, type=exist_read_path)
h = """Path to the parent group of the table inside the HDF5 file used for
    storing configurations"""
@click.option('-t', '--table_path', default='/', help=h)
h = "Name of the table at the end of table-path for storing configurations"
@click.option('-n', '--table_name', default='simulations', help=h)
h = "A list of keys within the configs to skip when generating the config ID"
@click.option('-s', '--skip_keys', default=['General', 'Materials'], help=h)
def preprocess(template, db, params, table_path, table_name, skip_keys):
    import nanowire.preprocess.preprocessor as pp
    pp = pp.Preprocessor(template, params=params)
    click.echo('Generating configs ...')
    pp.generate_configs()
    click.echo('Writing configs to disk ...')
    pp.write_configs()
    click.echo('Writing configs to database ...')
    pp.add_to_database(db_path=db, tb_path=table_path, tb_name=table_name,
                       skip_keys=skip_keys)
    click.secho('Preprocessing complete!', fg='green')
