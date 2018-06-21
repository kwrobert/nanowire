import os
import conff
import posixpath
import itertools
import pprint
from collections import MutableMapping, OrderedDict
from ..utils.utils import get_combos, do_profile, make_hash
from .config import Config, find_lists
from line_profiler import LineProfiler
from yaml import load as yload, dump as ydump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import json


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

    def __init__(self, template, params=None):
        if not os.path.isfile(template):
            raise ValueError('Template must be a path to a regular file')
        if type(params) == str:
            if not os.path.isfile(params):
                raise ValueError('params must be a path to a regular file or '
                                 'a dict')
            parser = conff.Parser(fns={'Q': fn_pint_quantity})
            self.in_pars = parser.load(params)
        elif isinstance(params, dict):
            parser = conff.Parser(fns={'Q': fn_pint_quantity})
            self.in_pars = parser.parse(params)
        else:
            self.in_pars = {}
        self.template = template
        self.variable = []
        self.optimized = []
        self.confs = []

    def generate_configs(self):
        """
        Generate all the unique Config objects containing a single set of
        parameters. Note that the Config objects generated by this function
        have not necessarily been rendered yet
        """

        locs, lists = find_lists(self.in_pars)
        paths = [posixpath.join(*l) for l in locs]
        combos = itertools.product(*lists)
        # print(list(combos))
        parser = conff.Parser(fns={'Q': fn_pint_quantity})
        names = Config({'P': self.in_pars})
        for combo in combos:
            for i, val in enumerate(combo):
                path = posixpath.join('P', paths[i])
                names[path] = val
            parser.update_names(names)
            conf = Config(parser.load(self.template),
                          skip_keys=['Postprocessing', 'General'])
            self.confs.append(conf)

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
