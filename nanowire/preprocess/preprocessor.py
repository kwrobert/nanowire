import os
import posixpath
import itertools
import tables as tb
import logging
import ast
import operator as op
import pint
import unqlite
import pickle
from collections import MutableMapping
from nanowire.utils.utils import (
    get_pytables_desc,
    add_row,
    update_row,
    ureg,
    Q_,
    find_lists,
    open_pytables_file
)
from nanowire.utils.config import Config
from .parser import Parser

UNSAFE_OPERATORS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                    ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
                    ast.Pow: op.pow, ast.Mod: op.mod,
                    ast.Eq: op.eq, ast.NotEq: op.ne,
                    ast.Gt: op.gt, ast.Lt: op.lt,
                    ast.GtE: op.ge, ast.LtE: op.le,
                    ast.Not: op.not_,
                    ast.USub: op.neg, ast.UAdd: op.pos,
                    ast.In: lambda x, y: op.contains(y, x),
                    ast.NotIn: lambda x, y: not op.contains(y, x),
                    ast.Is: lambda x, y: x is y,
                    ast.IsNot: lambda x, y: x is not y,
                    }
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
    quant = Q_(*args)
    return quant

def fn_pint_magnitude(q):
    """
    Return the magnitude of a pint Quantity
    """
    if not isinstance(q, pint.quantity._Quantity):
        raise ValueError("Argument of F.mag must be a pint Quantity")
    return q.magnitude


class Preprocessor:
    """
    This class consumes a configuration template and an optional params file
    and generates all the simulation configuration files, placing them in
    subdirectories named by the Simulation ID
    """
    # These are the operators used by the expression evaluator inside Parser
    # object. This replaces the safe_add, safe_mult, and safe_pow operators
    # with the unsafe python equivalents. This is fine because our config files
    # should only ever come from a trusted source, and hopefully we can trust
    # users to no accidentally compute a massive power or something
    DEFAULT_OPERATORS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                         ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
                         ast.Pow: op.pow, ast.Mod: op.mod,
                         ast.Eq: op.eq, ast.NotEq: op.ne,
                         ast.Gt: op.gt, ast.Lt: op.lt,
                         ast.GtE: op.ge, ast.LtE: op.le,
                         ast.Not: op.not_,
                         ast.USub: op.neg, ast.UAdd: op.pos,
                         ast.In: lambda x, y: op.contains(y, x),
                         ast.NotIn: lambda x, y: not op.contains(y, x),
                         ast.Is: lambda x, y: x is y,
                         ast.IsNot: lambda x, y: x is not y,
                         }

    def __init__(self, template):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
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
        ops = {'simpleeval': {'operators': self.DEFAULT_OPERATORS}}
        parser = Parser(fns={'Q': fn_pint_quantity,
                             'mag': fn_pint_magnitude},
                        params=ops, cache_graph=True)
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
        parser = Parser(fns={'Q': fn_pint_quantity,
                             'mag': fn_pint_magnitude},
                        params=ops, cache_graph=True)
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
            names = Config({'P': in_pars})
            parser.update_names(names)
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
            self.log.info('Writing conf %s to %s', conf.ID, outfile)
            conf.write(outfile)

    def add_to_database(self, db, tb_name='simulations',
                        skip_keys=None, update=False):
        """
        Add the generated configs to table located immediately beneath path
        `tb_path` with name 'tb_name' inside an UnQLite database `db`.

        Parameters
        ----------

        db : str
            A path to the database file. If it does not exist, it is created
            automatically.

        update : bool
            Update configs that already exist in the database
        """

        skip_keys = skip_keys if skip_keys is not None else []
        db = unqlite.UnQLite(db)
        col = db.collection(tb_name)
        # Only creates a new collection if one does not already exist
        col.create()
        existing_ids = set(rec['ID'].decode() for rec in col.all())
        for conf in self.confs:
            ID = conf.ID
            if (ID in existing_ids) and not update:
                self.log.info('ID %s already in table, skipping', conf.ID)
                continue
            elif (ID in existing_ids) and update:
                self.log.info('Updating ID %s', conf.ID)
                conf.update_in_db(db, col)
            else:
                self.log.info('Adding new ID %s ', conf.ID)
                conf.store_in_db(db, col)
        db.close()
