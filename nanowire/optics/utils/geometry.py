import itertools
import numpy as np

from sympy import Point, Circle, Ellipse, Triangle, Polygon, RegularPolygon
from sympy.utilities.lambdify import lambdify
from sympy.abc import x as _x
from sympy.abc import y as _y
from matplotlib.path import Path
from collections import OrderedDict
from .utils import get_nk


def get_mask(shape, xcoords, ycoords):
    """
    Given a sympy shape object (Circle, Triangle, etc) and two 1D arrays
    containing the x and y coordinates, return a mask that determines whether
    or not a given (x, y) point in within the shape. True means inside, False
    means outside
    """
    if isinstance(shape, Ellipse):
        expr = shape.equation()
        circ_f = lambdify((_x, _y), expr, modules='numpy', dummify=False)
        xv, yv = np.meshgrid(ycoords, xcoords)
        e = circ_f(xv, yv)
        mask = np.where(e > 0, 0, 1)
    else:
        polygon = Path(shape.vertices)
        mask = polygon.contains_points(list(itertools.product(xcoords,
                                                              ycoords)))
    return mask


def get_layers(sim):
    """
    Creates an OrderedDict of layer objects sorted with the topmost layer first

    :param sim: :py:class:`nanowire.optics.simulate.Simulator` or
                :py:class:`nanowire.optics.postprocess.Simulation`

    :returns: An OrderedDict of :py:class:`Layer` instances. The keys are the
              layer names from the config and the values are the instances
    :rtype: OrderedDict
    """

    ordered_layers = sim.conf.sorted_dict(sim.conf['Layers'])
    start = 0
    layers = OrderedDict()
    materials = sim.conf['Materials']
    for layer, ldata in ordered_layers.items():
        # Dont add the layer if we don't have field data for it because its
        # beyond max_depth
        max_depth = sim.conf[('Simulation', 'max_depth')]
        if max_depth and start >= max_depth:
            break
        layer_t = ldata['params']['thickness']
        # end = start + layer_t + sim.dz
        end = start + layer_t
        # Things are discretized, so start needs to be a location that we
        # have a grid point on, and not the continuous starting point of
        # the real physical layer
        start_ind = np.searchsorted(sim.Z, start)
        quantized_start = sim.Z[start_ind]
        end_ind = np.searchsorted(sim.Z, end)
        quantized_end = sim.Z[end_ind]
        if 'geometry' in ldata:
            g = ldata['geometry']
        else:
            g = {}
        layers[layer] = Layer(layer, start, end,
                              start_ind, end_ind, sim.period,
                              sim.xsamps, sim.ysamps, sim.dz,
                              base_material=ldata['base_material'],
                              geometry=g, materials=materials)
        start = end
    return layers

class Layer:

    def __init__(self, name, start, end, istart, iend, period, xsamples,
                 ysamples, dz, base_material=None, materials={}, geometry={}):
        self.name = name
        self.start = start
        self.end = end
        self.thickness = end - start
        self.period = period
        self.istart = istart
        self.iend = iend
        self.slice = (slice(self.istart, self.iend), ...)
        if geometry:
            self.collect_shapes(geometry)
        else:
            self.shapes = {}
        self.x_samples = xsamples
        self.y_samples = ysamples
        self.dx = period/xsamples
        self.dy = period/ysamples
        self.dz = dz
        self.base_material = base_material
        self.materials = materials

    def get_nk_dict(self, freq):
        """
        Build dictionary where keys are material names and values is a tuple
        containing (n, k) at a given frequency
        """
        nk = {mat: (get_nk(matpath, freq)) for mat, matpath in
              self.materials.items()}
        nk['vacuum'] = (1, 0)
        return nk

    def is_in_layer(self, z):
        """
        Determine if a given z coordinate lies within this layer
        """
        return self.start < z < self.end

    def collect_shapes(self, d):
        """
        Collect and instantiate the dictionary stored in the shapes attribute
        given a dictionary describing the shapes in the layer
        """

        shapes = OrderedDict()
        sorted_d = OrderedDict(sorted(d.items(),
                                      key=lambda tup: tup[1]['order']))
        for name, data in sorted_d.items():
            shape = data['type'].lower()
            if shape == 'circle':
                center = Point(data['center']['x'], data['center']['y'])
                radius = data['radius']
                material = data['material']
                shape_obj = Circle(center, radius)
            else:
                raise NotImplementedError('Can only handle circles right now')
            shapes[name] = (shape_obj, material)
        self.shapes = shapes
        return shapes

    def get_nk_at_point(self, x, y, freq):
        """
        Given the x and y coordinates of a point and the frequency of interest,
        return the n and k values at that point
        """
        nk = self.get_nk_dict(freq)
        # If this layer is just a slab of material, return the nk values of the
        # base material
        if not self.shapes:
            return nk[self.base_material]
        # First we need to get all the shapes that enclose point. tup[0] is the
        # actual shape object
        p = Point(x, y)
        enclosing_shapes = [tup for name, tup in
                            self.shapes.items() if tup[0].encloses_point(p)]
        if len(enclosing_shapes) == 0:
            return nk[self.base_material]
        # Now we need to find the innermost shape containing the point
        # not any([]) == True
        while len(enclosing_shapes) > 0:
            print("Inside while")
            innermost, material_name = enclosing_shapes.pop()
            if not any(innermost.encloses(tup[0]) for tup in enclosing_shapes):
                break
        print("Enclosing material: {}".format(material_name))
        return nk[material_name]

    def get_nk_matrix(self, freq, xcoords=None, ycoords=None):
        """
        Returns two 2D matrices containing the real component n and the
        imaginary component k of the index of refraction at a given frequency
        at each point in space in the x-y plane. This function handles internal
        geometry properly
        """

        # Get the nk values for all the materials in the layer
        nk = self.get_nk_dict(freq)
        # Create the matrix and fill it with the values for the base material
        if xcoords is None: 
            xcoords = np.linspace(0, self.period, self.x_samples)
        if ycoords is None:
            ycoords = np.linspace(0, self.period, self.y_samples)
        n_matrix = np.ones((len(xcoords),
                            len(ycoords)))*nk[self.base_material][0]
        k_matrix = np.ones((len(xcoords),
                            len(ycoords)))*nk[self.base_material][1]
        for name, (shape, mat) in self.shapes.items():
            n = nk[mat][0]
            k = nk[mat][1]
            # Get a mask that is True inside the shape and False outside
            mask = get_mask(shape, xcoords, ycoords)
            # print(mask)
            shape_nvals = mask*n
            shape_kvals = mask*k
            # print("Mask Shape: {}".format(mask.shape))
            # print("Nvals Shape: {}".format(shape_nvals.shape))
            # print("N Matrix Shape: {}".format(n_matrix.shape))
            n_matrix = np.where(mask, shape_nvals, n_matrix)
            # print(n_matrix)
            k_matrix = np.where(mask, shape_kvals, k_matrix)
        return n_matrix, k_matrix
