import os
import pprint
import conff
import posixpath
import itertools
import tables as tb
import logging
from collections import MutableMapping
from ..utils.utils import get_pytables_desc, add_row
from .config import Config, find_lists

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def fn_pint_quantity(*args):
    """
    Parse a pint Quantity in a config file.

    If len(*args) = 1, we assume its a string representing a pint Quantity.
    If len(*args) = 2, we assume its a sequence whose first item is the
    floating point value of the quantity, and whose second item is a string
    representing the units
    """
    raise NotImplementedError('Still need to add pint Quantity functionality')


class Preprocessor:
    """
    This class consumes a configuration template and an optional params file
    and generates all the simulation configuration files, placing them in
    subdirectories named by the Simulation ID
    """

    def __init__(self, template):
        self.log = logging.getLogger(__name__)
        if not os.path.isfile(template):
            raise ValueError('Template must be a path to a regular file')
        self.template = template
        self.variable = []
        self.optimized = []
        self.confs = []

    def generate_configs(self, skip_keys=['General'], params=None):
        """
        Generate all the unique Config objects containing a single set of
        parameters and add them to the self.confs attribute.

        :param skip_keys: A list of strings representing branches in the config
        to skip when generating the ID of each config. The strings must be
        slash separated paths to a location in the Config `like/this/one`. This
        list is passed on directly to the underlying Config objects without
        modification. Use this to prevent parameters that do not affect
        simulation outputs at all from affecting the Config ID

        .. note:: If you need one of your parameters to be an actual list that
        doesn't get used as a sequence of values for generating parameter
        combinations, put the list you wish to keep as a list directly in the
        template passed to the constructor of Preprocessor. This function
        """
        parser = conff.Parser(fns={'Q': fn_pint_quantity})
        if params is None:
            in_pars = {}
        elif type(params) == str:
            if not os.path.isfile(params):
                raise ValueError('params must be a path to a regular file or '
                                 'a dict')
            in_pars = parser.load(params)
        elif isinstance(params, MutableMapping):
            in_pars = parser.parse(params)
        else:
            raise ValueError('params must be a path to a regular file or '
                             'a dict')
        locs, lists = find_lists(in_pars)
        if locs:
            paths = [posixpath.join(*l) for l in locs]
            combos = list(itertools.product(*lists))
            names = Config({'P': in_pars})
            for combo in combos:
                for i, val in enumerate(combo):
                    path = posixpath.join('P', paths[i])
                    names[path] = val
                parser.update_names(names)
                conf = Config(parser.load(self.template),
                              skip_keys=skip_keys)
                self.confs.append(conf)
        else:
            conf = Config(parser.load(self.template),
                          skip_keys=skip_keys)
            self.confs.append(conf)
        return self.confs

    def write_configs(self, out=None):
        """
        Write all configs to disk in directory `out`. If `out` is not provided,
        configs are written in the same directory as the provided template
        """

        if out is None:
            out = os.path.dirname(self.template)
        if not os.path.isdir(out):
            raise ValueError('{} is not a directory'.format(out))

        for i, conf in enumerate(self.confs):
            outdir = os.path.join(out, conf.ID[0:10])
            outfile = os.path.join(outdir, 'sim_conf.yml')
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            conf.write(outfile)

    def add_to_database(self, db_path, tb_path='/', tb_name='simulations',
                        skip_keys=None):
        """
        Add the generated configs to table located immediately beneath path
        `tb_path` with name 'tb_name' inside a PyTables HDF5 file located on
        the filesystem at `db_path`.

        If there is no file located at `db_path`, it is created. If no table
        with matching location exists inside the HDF5 file, all intermediate
        groups along the path and the table at the end of the path are created
        """

        skip_keys = skip_keys if skip_keys is not None else []
        # Generate table description
        # flat_conf = self.confs[0].flatten(sep='__')
        print(self.confs[0])
        desc_dict = {'ID': self.confs[0].ID}
        # desc_dict.update(flat_conf)
        desc_dict.update(self.confs[0])
        # desc = get_pytables_desc(desc_dict, skip_keys=skip_keys)
        desc = get_pytables_desc(desc_dict)
        desc['ID']._v_pos = 0
        pprint.pprint(desc)
        # Open in append mode!
        hdf = tb.open_file(db_path, mode='a')
        try:
            table = hdf.create_table(where=tb_path, name=tb_name,
                                     description=desc, createparents=True)
            existing_ids = []
        # Table already exists
        except tb.NodeError:
            table = hdf.get_node(where=tb_path, name=tb_name,
                                 classname='Table')
            existing_ids = table.read(field='ID')
        # conf_row = table.row
        for conf in self.confs:
            write_dict = {'ID': conf.ID}
            write_dict.update(conf)
            if conf.ID.encode('utf-8') in existing_ids:
                self.log.info('ID %s already in table', conf.ID)
                continue
            add_row(table, write_dict)
            # conf_row['ID'] = conf.ID
            # flat = conf.flatten(skip_branch=skip_keys, sep='__')
            # for k, v in flat.items():
            #     conf_row[k] = v
            # conf_row.append()
        table.flush()
        hdf.close()
