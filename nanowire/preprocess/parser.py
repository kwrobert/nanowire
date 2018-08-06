import json
import logging
import os
import collections
import copy
import warnings
import posixpath
import networkx as nx
import re
from jinja2 import Template
from boltons.iterutils import remap, default_enter, get_path
from cryptography.fernet import Fernet
from nanowire.preprocess import simpleeval
from nanowire.preprocess.utils import (
    update_recursive,
    yaml_safe_load,
    filter_value,
    FancyDict,
)
from collections import MutableMapping, ItemsView, Sequence
# For visual inspection of the graph during debugging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph, write_dot
# import pygraphviz as pgv


class Visitor:
    """
    Callable to be used in remap for building a directed graph from a nested
    data structure
    """

    def __init__(self, d):
        self.graph = nx.DiGraph()
        self.ref_regex = re.compile('R(?:\.[^\d\W]\w*)+', re.MULTILINE)
        self.split_regex = re.compile('(?<!(F))\.')
        self.d = d

    def _get_refs_in_str(self, s):
        refs = re.findall(self.ref_regex, s)
        return refs

    def _get_refs(self, val):
        refs = []
        if isinstance(val, str):
            refs.extend(self._get_refs_in_str(val))
        elif isinstance(val, list):
            for el in val:
                if isinstance(el, str):
                    refs.extend(self._get_refs_in_str(el))
        return refs

    def _add_deps(self, nodepath, refs):
        """
        If val contains any references to other items in the config, add them
        to the graph as directed edges from nodepath to reference in val to
        indicate them as dependencies of nodepath
        """
        for ref in refs:
            # Don't split up function keys like 'F.extend'
            ref = ref.strip('R')
            seq = [el for el in re.split(self.split_regex, ref) if el]
            dep_path = posixpath.join(*seq)
            self.graph.add_node(dep_path)
            self.graph.add_edge(nodepath, dep_path)

    def _add_deps_recursive(self, nodepath, refs):
        """
        Add dependency on all refs in refs to all nodes below nodepath
        """
        root = self.d[nodepath]
        for k, v in root.items():
            childpath = posixpath.join(nodepath, k)
            if isinstance(v, MutableMapping):
                self._add_deps_recursive(childpath, refs)
            else:
                self._add_deps(childpath, refs)

    def __call__(self, path, key, val):
        if not path:
            parentpath = '/'
        else:
            parentpath = posixpath.join(*path)
        nodepath = posixpath.join(*path, key)
        # Get a list of all the references in val
        refs = self._get_refs(val)
        if key == '_depends':
            self._add_deps_recursive(parentpath, refs)
            return False
        else:
            self.graph.add_node(nodepath)
            self._add_deps(nodepath, refs)
            self.graph.add_edge(parentpath, nodepath)
            return True

    def enter(self, path, key, value):
        if isinstance(value, Sequence):
            return value.__class__(), False
        elif isinstance(value, MutableMapping):
            return value, ItemsView(value)
        else:
            return default_enter(path, key, value)


class Parser:
    # default params
    default_params = {
        'etype': 'fernet',
        # list of simpleeval library parameters
        'simpleeval': {
            # by default operators = simpleeval.DEFAULT_OPERATORS,
            'operators': {},
            'options': {
                'max_power': simpleeval.MAX_POWER,
                'max_string_length': simpleeval.MAX_STRING_LENGTH,
                'disallow_prefixes': simpleeval.DISALLOW_PREFIXES
            }
        }
    }

    def __init__(self, names=None, fns=None, params=None, cache_graph=False):
        """
        :param params: A dictionary containing some parameters that will modify
        how the builtin functions run. For example, the type of encryption to
        use and the encrpyption key to use or simpleeval library parameters
        """
        self.errors = []
        self.logger = self.prepare_logger()
        self.params = self.prepare_params(params=params)
        self.fns = self.prepare_functions(fns=fns)
        self.names = self.prepare_names(names=names)
        self._evaluator = self.prepare_evaluator()
        self.cache_graph = cache_graph
        self.graph = None

    def prepare_logger(self):
        logger = logging.getLogger('conff')
        logger.addHandler(logging.NullHandler())
        return logger

    def prepare_params(self, params: dict = None):
        """
        Setup parameters for the library

        :param params: A dictionary containing some parameters that will modify
        how the builtin functions run. For example, the type of encryption to
        use and the encrpyption key to use or simpleeval library parameters

        :return: Prepared parameters
        """
        # ensure not to update mutable params
        params = copy.deepcopy(params or {})
        # inject with default params with exception for simpleeval.operators
        params = update_recursive(params, self.default_params)
        return FancyDict(params)

    def prepare_functions(self, fns: dict = None):
        fns = fns or {}
        cls_fns = {fn[3:]: getattr(self, fn) for fn in dir(self) if 'fn_' in fn}
        result = {'F': update_recursive(fns, cls_fns)}
        return FancyDict(result)

    def prepare_names(self, names: dict = None):
        names = names or {}
        return FancyDict(names)

    def prepare_evaluator(self):
        """
        Setup evaluator engine

        :return: Prepare evaluator engine
        """
        simpleeval_params = self.params.get('simpleeval', {})
        # update simpleeval safety options
        for k, v in simpleeval_params.get('options', {}).items():
            setattr(simpleeval, k.upper(), v)
        evaluator = simpleeval.EvalWithCompoundTypes()
        # self._evals_functions should mirror self.fns
        # TODO: Make a test to ensure proper mirroring
        evaluator.functions = self.fns
        evaluator.names = self.names
        # set the operators
        if simpleeval_params.get('operators'):
            evaluator.operators = simpleeval_params.get('operators')
        return evaluator


    def load(self, fs_path: str, fs_root: str = '', fs_include: list = None):
        """
        Parse configuration file on disk.

        :param fs_path: The path to the file on disk. If fs_root is specified,
        this will be interpreted as a path relative to fs_root
        :type fs_path: str
        :param fs_root: Root directory to use when parsing. Defaults to the
        directory of the input file.
        :type fs_root: str
        :param fs_include: A list of additional directories in which to
        search for included files. Always contains the directory of the input
        file, and will also contain fs_root if specified.
        :type fs_include: list
        """
        fs_file_path = os.path.join(fs_root, fs_path)
        _, fs_file_ext = os.path.splitext(fs_file_path)
        fs_root = fs_root if fs_root is None else os.path.dirname(fs_file_path)
        self.params.update({'fs_path': fs_path, 'fs_root': fs_root})
        with open(fs_file_path) as stream:
            if fs_file_ext in ('.yml', '.yaml'):
                data = FancyDict(yaml_safe_load(stream))
                names = FancyDict({'R': data})
                self.update_names(names)
                data = self._process(data)
            elif 'json' in fs_file_ext:
                data = FancyDict(json.loads(stream.read()))
                names = FancyDict({'R': data})
                self.update_names(names)
                data = self._process(data)
            else:
                # load_yaml initial structure
                data = '\n'.join(stream.readlines())
        # Delete anything specific to this file so we can reuse the parser
        for k in ('fs_path', 'fs_root', 'R'):
            if k in self.params:
                del self.params[k]
            if k in self.names:
                del self.names[k]
        return data

    def parse(self, data):
        """
        Main entry point to parse arbitary data type
        :param data: Input can be any data type such as dict, list, string, int
        :return: Parsed data
        """
        if isinstance(data, MutableMapping):
            data = FancyDict(data)
            new_names = FancyDict({'R': data})
            self.update_names(new_names)
            result = self._process(data)
        else:
            result = self.parse_expr(data)
        return result

    def parse_expr(self, expr: str):
        """
        Parse an expression in string
        """
        try:
            v = self._evaluator.eval(expr=expr)
        except SyntaxError as ex:
            v = expr
            # TODO: feature T2
            self.logger.warn("simpleeval SyntaxError exception:\n"
                             "  Expression: %s  Message: %s",
                             ex.text, ex.msg)
            self.errors.append([expr, ex])
        except simpleeval.InvalidExpression as ex:
            v = expr
            # TODO: feature T2
            self.logger.warn("simpleeval InvalidExpression exception:\n"
                             "  Expression: %s\n  Message: %s\n Return: %s",
                             ex.expression, ex.message, str(v))
            self.errors.append(ex)
        except Exception as ex:
            v = expr
            # TODO: feature T2
            self.errors.append(ex)
            msg = "Expression: {}".format(expr)
            print(msg)
            ex.args = ex.args + (msg,)
            raise
        # TODO: feature T4: include this part of the classes so user could override
        v = filter_value(v)
        return v

    def build_graph(self, d):
        if self.cache_graph and self.graph is not None:
            return self.graph
        visit = Visitor(d)
        d = remap(d, visit=visit, enter=visit.enter)
        if not nx.is_directed_acyclic_graph(visit.graph):
            raise ValueError('Your config has circular dependencies!')
        # write_dot(visit.graph, 'graph.dot')
        # os.system('dot -Tsvg graph.dot -o /tmp/graph.svg')
        # A = to_agraph(visit.graph)
        # A.layout('dot')
        # A.draw('graph.png')
        # img = mpimg.imread('graph.png')
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # plt.show()
        # os.remove('graph.png')
        if self.cache_graph:
            self.graph = visit.graph
        return visit.graph

    def _process_list(self, l):
        for i, v in enumerate(l):
            if isinstance(v, str):
                res = self.parse_expr(v)
            elif isinstance(v, list):
                res = self._process_list(v)
            elif isinstance(v, (float, int)):
                res = v
            else:
                raise ValueError('Unhandled type: {}'.format(type(v)))
            l[i] = res
        return l

    def _process(self, root):
        """
        The main parsing function
        """
        g = self.build_graph(root)
        sorted_nodes = list(reversed(list(nx.topological_sort(g))))
        for nodepath in sorted_nodes:
            if nodepath == '/':
                continue
            path = tuple(nodepath.split('/'))
            key = path[-1]
            parentpath = path[:-1]
            # Remove _depends keys as we go
            if key == '_depends':
                try:
                    del root[parentpath]['_depends']
                except KeyError:
                    pass
                continue
            # We might have removed this item after running some function
            try:
                item = root[path]
            except KeyError:
                continue
            parent = root[parentpath]
            if 'F.extend' == key:
                if isinstance(item, str):
                    item = self.parse_expr(item)
                root[parentpath] = self.fn_extend(item, root[parentpath])
                if isinstance(parent, MutableMapping):
                    del root[parentpath]['F.extend']
            elif 'F.template' == key:
                item = self.fn_template(item)
                if isinstance(item, MutableMapping):
                    self.fn_update(item, root[parentpath])
                else:
                    root[parentpath] = item
                if isinstance(parent, MutableMapping):
                    del parent['F.template']
            elif 'F.update' == key:
                self.fn_update(item, parent)
                del root[parentpath]['F.update']
            elif 'F.foreach' == key:
                for k in ('values', 'template'):
                    if k not in parent['F.foreach']:
                        raise ValueError('F.foreach missing key: {}'.format(k))

                self.fn_foreach(item, root[parentpath])
                del parent['F.foreach']
            elif isinstance(item, list):
                item = self._process_list(item)
                root[nodepath] = item
            elif isinstance(item, str):
                value = self.parse_expr(item)
                if len(path) == 1:
                    root[path[0]] = value
                elif len(path) > 1:
                    nested_d = get_path(root, path[:-1])
                    nested_d[path[-1]] = value
                else:
                    raise ValueError('This path does not make sense: '
                                     '{}'.format(path))
        # Remove all _depends keys
        root = remap(root, visit=lambda p, k, v: k != '_depends')
        return dict(root)

    def add_functions(self, funcs: dict):
        """
        Add functions to the list of available parsing function. Funcs should
        be a dict whose keys are the name you would like the function to have,
        and whose value is a callable that maps to that name. The functions
        will be callable via F.name_of_func(args_go_here)
        """

    def update_names(self, names: dict):
        """
        Add names to the dictionary of names available when parsing. These
        names are accessible via the syntax R.path.to.name. Any overlapping
        keys between the existing names dict and the argument to this function
        will be replaced using the values in the argument dict.
        """
        # self.names = update_recursive(self.names, names)
        self.names.update(names)

    def generate_crypto_key(self):
        """
        Generate a cryptographic key for encrypting data. Stores the key in
        self.params['ekey'] so it is accessible to encrypt parsing functions.
        Also returns the key
        """
        etype = self.params.get('etype')
        if etype == 'fernet':
            key = Fernet.generate_key()
        else:
            key = None
        self.params['ekey'] = key
        return key

    def fn_str(self, val):
        return str(val)

    def fn_float(self, val):
        return float(val)

    def fn_int(self, val):
        return int(val)

    def fn_has(self, val, name):
        if isinstance(val, collections.Mapping):
            return val.get(name, False) is not False
        else:
            return name in val

    def fn_next(self, vals, default=None):
        vals = [vals] if type(vals) != list else vals
        val = next(iter(vals), default)
        return val

    def fn_join(self, vals, sep=' '):
        vals = [val for val in vals if val]
        return sep.join(vals)

    def fn_trim(self, val: str, cs: list = None):
        cs = cs if cs else ['/', ' ']
        for c in cs:
            val = val.strip(c)
        return val

    def fn_linspace(self, start, end, steps):
        delta = (end - start) / (steps - 1)
        return [start + delta * i for i in range(steps)]

    def fn_arange(self, start, end, delta):
        vals = [start]
        while vals[-1] + delta <= end:
            vals.append(vals[-1] + delta)
        return vals

    def fn_extend(self, val, val2):
        val = copy.deepcopy(val)
        if isinstance(val, list) and isinstance(val2, list):
            val.extend(val2)
        elif isinstance(val, MutableMapping) and isinstance(val2,
                                                            MutableMapping):
            for k, v in val2.items():
                if k not in val:
                    val[k] = v
        return val

    def fn_update(self, update, parent):
        def walk(u, p):
            if isinstance(u, MutableMapping) and isinstance(p, MutableMapping):
                for k, v in u.items():
                    p[k] = walk(v, p.get(k, v))
                return p
            else:
                return u
        walk(update, parent)

    def fn_encrypt(self, data):
        etype = self.params.get('etype', None)
        ekey = self.params.get('ekey', None)
        token = None
        if etype == 'fernet' and ekey is not None:
            f = Fernet(ekey)
            token = f.encrypt(data=str(data).encode()).decode()
        else:
            raise ValueError('Encryption key cannot be of type None')
        return token

    def fn_decrypt(self, data):
        etype = self.params.get('etype', None)
        ekey = self.params.get('ekey', None)
        message = None
        if etype == 'fernet' and ekey is not None:
            f = Fernet(ekey)
            message = f.decrypt(token=str(data).encode()).decode()
        else:
            raise ValueError('Encryption key cannot be of type None')
        return message

    def fn_inc(self, fs_path, fs_root: str = None):
        fs_root = fs_root if fs_root else self.params['fs_root']
        # Make sure to pass on any modified options to the sub parser
        sub_parser = Parser(params=self.params)
        data = sub_parser.load(fs_path=fs_path, fs_root=fs_root)
        return data

    def fn_foreach(self, foreach, parent):
        template = foreach['template']
        if not isinstance(template, MutableMapping):
            raise ValueError('template item of F.foreach must be a dict')
        for i, v in enumerate(foreach['values']):
            self.names.update({'loop': {'index': i, 'value': v,
                                        'length': len(foreach['values'])}})
            result = {}
            for key, val in template.items():
                pkey = self.parse_expr(key)
                pval = self._process(copy.copy(val))
                result[pkey] = pval
            parent.update(result)
        # remove this specific foreach loop info from names dict so we don't
        # break any subsequent foreach loops
        del self.names['loop']

    def fn_template(self, template: str, root=None):
        template = self.parse_expr(template)
        engine = Template(template)
        obj_str = engine.render(**self.names)
        # TODO: feature T2
        obj = yaml_safe_load(obj_str)
        return obj


class ParserPlugin(object):
    pass
