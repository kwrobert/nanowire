import sympy
from sympy import Point, Circle, Ellipse, Triangle, Polygon, RegularPolygon
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, y as _x, _y
from matplotlib.path import Path
import itertools
import numpy as np


def get_mask(shape, xcoords, ycoords):
    """
    Given a sympy shape object (Circle, Triangle, etc) and two 1D arrays
    containing the x and y coordinates, return a mask that determines whether
    or not a given (x, y) point in within the shape. True means inside, False
    means outside
    """
    if isinstance(shape, Ellipse):
        expr = shape.equation()
        circ_f = lambdify((x, y), expr, modules='numpy', dummify=False)
        xv, yv = np.meshgrid(xcoords, ycoords)
        e = circ_f(xv, yv) 
        mask = np.where(e > 0, False, True)
    else:
        polygon = Path(shape.vertices)
        mask = polygon.contains_points(list(itertools.product(xcoords, 
                                                              ycoords)))
    return mask


class Layer:

    def __init__(self, name, start, end, period, dx, dy, dz, material=None, geometry={}):
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
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def collect_shapes(self, d):
        """
        Collect and instantiate the dictionary stored in the shapes attribute
        given a dictionary describing the shapes in the layer
        """

        shapes = {}
        for name, data in d.items():
            shape = data['type'].lower()
            if shape == 'circle':
                center = Point(data['center']['x'], data['center']['y'])
                radius = data['radius']
                material = data['Material']
                shape_obj = Circle(center, radius)
            else:
                raise NotImplementedError('Can only handle circles right now')
            shapes[name] = (shape_obj, material)
        self.shapes = shapes
        return shapes
