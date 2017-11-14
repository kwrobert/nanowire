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
        xv, yv = np.meshgrid(xcoords, ycoords)
        e = circ_f(xv, yv) 
        mask = np.where(e > 0, False, True)
    else:
        polygon = Path(shape.vertices)
        mask = polygon.contains_points(list(itertools.product(xcoords, 
                                                              ycoords)))
    return mask


class Layer:

    def __init__(self, name, start, end, period, xsamples, ysamples, dz, 
                 base_material=None, materials={}, geometry={}):
        self.name = name
        self.start = start
        self.end = end
        self.thickness = end - start
        self.period = period
        self.istart = int(round(start/dz))
        self.iend = int(round(end/dz))
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

    def get_nk_matrix(self, freq):
        """
        Returns two 2D matrices containing the real component n and the
        imaginary component k of the index of refraction at a given frequency
        at each point in space in the x-y plane. This function handles internal
        geometry properly
        """

        # Get the nk values for all the materials in the layer
        nk = {mat: (get_nk(matpath, freq)) for mat, matpath in
              self.materials.items()}
        nk['vacuum'] = (1, 0)
        # Create the matrix and fill it with the values for the base material
        n_matrix = np.ones((self.x_samples,
                            self.y_samples))*nk[self.base_material][0]
        k_matrix = np.ones((self.x_samples,
                            self.y_samples))*nk[self.base_material][1]
        xcoords = np.arange(0, self.period, self.dx)
        ycoords = np.arange(0, self.period, self.dy)
        for name, (shape, mat) in self.shapes.items():
            n = nk[mat][0]
            k = nk[mat][1]
            # Get a mask that is True inside the shape and False outside
            mask = get_mask(shape, xcoords, ycoords)
            # print(mask)
            shape_nvals = mask*n
            shape_kvals = mask*k
            n_matrix = np.where(mask, shape_nvals, n_matrix)
            # print(n_matrix)
            k_matrix = np.where(mask, shape_kvals, k_matrix)
        return n_matrix, k_matrix
