import itertools
import numpy as np

from sympy import Point, Circle, Ellipse, Triangle, Polygon, RegularPolygon
from sympy.utilities.lambdify import lambdify
from sympy.abc import x as _x
from sympy.abc import y as _y
from matplotlib.path import Path
from collections import OrderedDict
from nanowire.optics.utils.utils import get_nk
from nanowire.utils.utils import sorted_dict


def get_mask_by_shape(shape, xcoords, ycoords):
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
    return mask.astype(int)


def get_mask_by_material(layer, material, x, y):
    """
    Given a layer object, the name of a material in the layer and two 1D arrays
    containing the x and y coordinates, return a mask that determines whether
    or not a given (x, y) point in within the given material. True means
    inside, False means outside
    """
    if material == layer.base_material:
        mask = np.ones((len(x), len(y)), dtype=int)
    else:
        mask = np.zeros((len(x), len(y)), dtype=int)

    for shape_name, (shape_obj, mat_name) in layer.shapes.items():
        shape_mask = get_mask_by_shape(shape_obj, x, y)
        if mat_name == material:
            mask[shape_mask == 1] = 1
        else:
            mask[shape_mask == 1] = 0
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

    ordered_layers = sorted_dict(sim.conf['Layers'])
    start = 0
    layers = OrderedDict()
    materials = sim.conf['Materials']
    max_depth = sim.conf[('General', 'max_depth')]
    for layer, ldata in ordered_layers.items():
        # Dont add the layer if we don't have field data for it because its
        # beyond max_depth
        if max_depth and start >= max_depth:
            break
        layer_t = ldata['thickness']
        # end = start + layer_t + sim.dz
        end = start + layer_t
        # Things are discretized, so start needs to be a location that we
        # have a grid point on, and not the continuous starting point of
        # the real physical layer
        if 'geometry' in ldata:
            g = ldata['geometry']
        else:
            g = {}
        layers[layer] = Layer(layer, start, end, sim.period,
                              base_material=ldata['base_material'],
                              materials=materials, geometry=g)
        start = end
    return layers


class Layer:
    """
    Object for representing a layer within an RCWA simulation. It contains
    sympy.Shape objects representing it's internal geometry, and a dictionary
    of materials within the layer
    """

    def __init__(self, name, start, end, period, base_material=None,
                 materials={}, geometry={}):
        self.name = name
        self.start = start
        self.end = end
        self.thickness = end - start
        self.period = period
        self.base_material = base_material
        self.materials = {}
        if base_material == 'vacuum':
            self.materials[base_material] = None
        else:
            self.materials[base_material] = materials[base_material]
        if geometry:
            self._collect_shapes(geometry, materials)
        else:
            self.shapes = OrderedDict()

    def _collect_shapes(self, d, materials):
        """
        Collect and instantiate the dictionary stored in the shapes attribute
        given a dictionary describing the shapes in the layer
        """

        shapes = OrderedDict()
        sorted_shapes = sorted_dict(d)
        for name, data in sorted_shapes.items():
            shape = data['type'].lower()
            if shape == 'circle':
                center = Point(data['center']['x'], data['center']['y'])
                radius = data['radius']
                material = data['material']
                if material not in self.materials:
                    self.materials[material] = materials[material]
                shape_obj = Circle(center, radius)
            else:
                raise NotImplementedError('Can only handle circles right now')
            shapes[name] = (shape_obj, material)
        self.shapes = shapes
        return shapes

    def get_slice(self, zcoords):
        """
        Given a set of z coordinates, get a length 3 tuple can be used to
        retrieve the chunk of a 3D array that falls within this layer. This
        assumes the z direction is along the zeroth axis (i.e arr[z, x, y]).
        So, one can slice out the chunk of the 3D array with
        arr[layer.get_slice()]
        """

        start_ind = np.searchsorted(zcoords, self.start)
        end_ind = np.searchsorted(zcoords, self.end)
        return (slice(start_ind, end_ind), ...)

    def get_nk_dict(self, freq):
        """
        Build dictionary where keys are material names and values is a tuple
        containing (n, k) at a given frequency
        """
        nk = {mat: (get_nk(matpath, freq)) for mat, matpath in
              self.materials.items()}
        return nk

    def is_in_layer(self, z):
        """
        Determine if a given z coordinate lies within this layer
        """
        return self.start < z < self.end

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

    def get_nk_matrix(self, freq, xcoords, ycoords):
        """
        Returns two 2D matrices containing the real component n and the
        imaginary component k of the index of refraction at a given frequency
        at each point in space in the x-y plane. This function handles internal
        geometry properly
        """

        # Get the nk values for all the materials in the layer
        nk = self.get_nk_dict(freq)
        # Create the matrix and fill it with the values for the base material
        n_matrix = np.ones((len(xcoords),
                            len(ycoords)))*nk[self.base_material][0]
        k_matrix = np.ones((len(xcoords),
                            len(ycoords)))*nk[self.base_material][1]
        for name, (shape, mat) in self.shapes.items():
            n = nk[mat][0]
            k = nk[mat][1]
            # Get a mask that is True inside the shape and False outside
            mask = get_mask_by_shape(shape, xcoords, ycoords)
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
