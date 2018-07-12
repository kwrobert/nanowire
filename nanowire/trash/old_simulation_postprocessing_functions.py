"""
A bunch of old postprocessing functions that were ugly and needed to be
rewritten
"""

class Simulation:

    def integrate_nanowire(self, zvals, nkEsq=None):
        """
        Use natural neighbor interpolation to integrate nk|E|^2 inside the
        nanowire in shell in polar coordinates for optimum accuracy
        """

        # if len(self.X) % 2 == 0 or len(self.Y) % 2 == 0:
        #     raise ValueError("Need and odd number of x-y samples to use this "
        #                      " function")
        self.log.info("CALLING INTEGRATE_NANOWIRE")
        nw_layer = self.layers['NW_AlShell']
        core_rad = self.conf['Layers']['NW_AlShell']['params']['core_radius']
        shell_rad = self.conf['Layers']['NW_AlShell']['params']['shell_radius']
        period = self.conf['Simulation']['params']['array_period']
        # shell_rad = self.conf['Layers']['NW_AlShell']['params']['shell_radius']
        # Extract nkEsq data in layer
        if nkEsq is None:
            nkEsq = self.data['nknormEsq'][nw_layer.get_slice(self.Z)]
        # Get masks for each region
        core_mask = get_mask_by_material(nw_layer, 'GaAs', self.X, self.Y)
        shell_mask = get_mask_by_material(nw_layer, 'AlInP', self.X, self.Y)
        cyc_mask = np.logical_not(np.logical_or(core_mask, shell_mask))
        core_mask3d = np.broadcast_to(core_mask,
                                      (nkEsq.shape[0],
                                       core_mask.shape[0],
                                       core_mask.shape[1]))
        shell_mask3d = np.broadcast_to(shell_mask,
                                       (nkEsq.shape[0],
                                        shell_mask.shape[0],
                                        shell_mask.shape[1]))
        cyc_mask3d = np.broadcast_to(cyc_mask,
                                     (nkEsq.shape[0],
                                      cyc_mask.shape[0],
                                      cyc_mask.shape[1]))
        # Finally the cyclotene
        cyc_vals = nkEsq*cyc_mask3d
        cyc_result = integrate3d(cyc_vals, zvals, self.X, self.Y,
                                 meth=intg.simps)
        # Extract vals in each region from nkEsq array
        core_inds = np.where(core_mask3d)
        shell_inds = np.where(shell_mask3d)
        core_vals = nkEsq[core_inds]
        shell_vals = nkEsq[shell_inds]
        pts = cartesian_product((zvals, self.X, self.Y))
        # Shift x and y values so origin is at center of nanowire
        # core_pts_inds = np.where((core_pts[:, 1] - period/2)**2 + (core_pts[:, 2] - period/2)**2 <= core_rad**2)
        p2 = period/2.0
        pts[:, 2] -= p2
        pts[:, 1] -= p2
        core_pts_inds = np.where(pts[:, 1]**2 + pts[:, 2]**2 <= core_rad**2)
        shell_pts_inds = np.where((core_rad**2 < pts[:, 1]**2 + pts[:, 2]**2)
                                  &
                                  (pts[:, 1]**2 + pts[:, 2]**2 <= shell_rad**2))
        # core_pts_inds = np.where((pts[:, 1]-p2)**2 + (pts[:, 2]-p2)**2 <= core_rad**2)
        # shell_pts_inds = np.where((core_rad**2 <= (pts[:, 1]-p2)**2 + (pts[:, 2]-p2)**2)
        #                           &
        #                           ((pts[:, 1]-p2)**2 + (pts[:, 2]-p2)**2 <= shell_rad**2))
        core_pts = pts[core_pts_inds[0], :]
        shell_pts = pts[shell_pts_inds[0], :]
        # core_pts = np.column_stack((xx[core_inds[1], core_inds[2], core_inds[0]],
        #                        yy[core_inds[1], core_inds[2], core_inds[0]],
        #                        zz[core_inds[1], core_inds[2], core_inds[0]]))
        # Transform cartesian points into polar coordinates.
        # polar_pts[r, theta, z]
        core_polar_pts = np.zeros_like(core_pts)
        core_polar_pts[:, 0] = np.sqrt(core_pts[:, 2]**2 + core_pts[:, 1]**2)
        # This returns angles on [-pi, pi], so shift them
        core_polar_pts[:, 1] = np.arctan2(core_pts[:, 2], core_pts[:, 1])
        # core_polar_pts[:, 1][core_polar_pts[:, 1] < 0] += 2*np.pi
        core_polar_pts[:, 2] = core_pts[:, 0]
        # Same for the shell
        shell_polar_pts = np.zeros_like(shell_pts)
        shell_polar_pts[:, 0] = np.sqrt(shell_pts[:, 2]**2 + shell_pts[:, 1]**2)
        # This returns angles on [-pi, pi], so shift them
        shell_polar_pts[:, 1] = np.arctan2(shell_pts[:, 2], shell_pts[:, 1])
        # shell_polar_pts[:, 1][shell_polar_pts[:, 1] < 0] += 2*np.pi
        shell_polar_pts[:, 2] = shell_pts[:, 0]
        ###########
        # Insert S4 data at r = 0 here and all theta values.
        # Odd numbers of points guarantee an S4 point at the center of the unit
        # cell. Use an odd number of points, and take the center value and
        # replicate it across all thetas for r = 0
        # 1) Does the weird start in the center go away
        # 2) Compare integral by material to plain old integral and flux method
        ###########
        # Get function value at r = 0
        # center_inds = np.where((pts[:, 1] == .125) & (pts[:, 2] == .125))
        center_inds = np.where(core_polar_pts[:, 0] == 0)
        rzero_core_vals = core_vals[center_inds[0]]
        extra_theta = 180
        extra_pts = cartesian_product((np.array([0]),
                                       np.linspace(-np.pi, np.pi, extra_theta),
                                       core_polar_pts[center_inds[0], 2]))
        repeated_zero_vals = np.concatenate([rzero_core_vals for i in range(extra_theta)])
        core_polar_pts = np.concatenate((core_polar_pts, extra_pts))
        core_vals = np.concatenate((core_vals, repeated_zero_vals))

        # Extract interpolated points on a polar grid
        rstart = 0
        core_numr = 180
        shell_numr = 60
        numtheta = 360
        numz = 200
        # If the last element of each range is complex, the ranges behave like
        # np.linspace
        ranges = [[rstart, core_rad, 1j*core_numr], [-np.pi, np.pi, 1j*numtheta],
                  [nw_layer.start, nw_layer.end, 1j*numz]]
        # ranges = [[rstart, core_rad, 1j*core_numr], [0, 2*np.pi, 1j*numtheta],
        #           [nw_layer.start, nw_layer.end, 1j*numz]]
        core_interp = nn.griddata(core_polar_pts, core_vals, ranges)
        # ranges = [[core_rad, shell_rad, 1j*shell_numr], [0, 2*np.pi, 1j*numtheta],
        #           [nw_layer.start, nw_layer.end, 1j*numz]]
        ranges = [[core_rad, shell_rad, 1j*shell_numr], [-np.pi, np.pi, 1j*numtheta],
                  [nw_layer.start, nw_layer.end, 1j*numz]]
        shell_interp = nn.griddata(shell_polar_pts, shell_vals, ranges)
        # Multiply by area factor in polar coords to get integrand
        core_rvals = np.linspace(rstart, core_rad, core_numr)
        thetavals = np.linspace(-np.pi, np.pi, numtheta)
        intzvals = np.linspace(nw_layer.start, nw_layer.end, numz)
        rr, tt = np.meshgrid(core_rvals, thetavals, indexing='ij')
        # xx = rr*np.cos(tt)
        # yy = rr*np.sin(tt)
        # __import__('pdb').set_trace()
        integrand = core_interp*rr[:, :, None]
        core_result = integrate3d(integrand, thetavals, intzvals, core_rvals,
                                  meth=intg.simps)
        # Shell integral
        shell_rvals = np.linspace(core_rad, shell_rad, shell_numr)
        rr, tt = np.meshgrid(shell_rvals, thetavals, indexing='ij')
        integrand = shell_interp*rr[:, :, None]
        # integrand = core_interp*rr[:, :, None]
        shell_result = integrate3d(integrand, thetavals, intzvals, shell_rvals,
                                   meth=intg.simps)
        #plt.matshow(shell_mask3d[1, :, :])
        #plt.show()
        #plt.matshow(shell_mask)
        #plt.show()
        self.log.info("Core Integral Result = {}".format(core_result))
        self.log.info("Shell Integral Result = {}".format(shell_result))
        self.log.info("Cyc Integral Result = {}".format(cyc_result))
        return sum((core_result, shell_result, cyc_result))
        # return (core_rvals, shell_rvals, thetavals, zvals, core_interp,
        #         shell_interp, core_result, shell_result, cyc_result)

    def integrate_layer(self, lname, layer):
        freq = self.conf[('Simulation', 'params', 'frequency')]
        n_mat, k_mat = layer.get_nk_matrix(freq, self.X, self.Y)
        try:
            Esq = self.data['normEsquared']
        except KeyError:
            Esq = self.normEsquared()
        nkEsq = n_mat*k_mat*Esq[layer.get_slice(self.Z)]
        results = {}
        for mat in layer.materials.keys():
            mask = get_mask_by_material(layer, mat, self.X, self.Y)
            values = nkEsq*mask
            points = (self.Z[layer.get_slice(self.Z)], self.X, self.Y)
            rgi = interpolate.RegularGridInterpolator(points, values,
                                                      method='linear',
                                                      bounds_error=True)
            z = self.Z[layer.get_slice(self.Z)]
            # x = self.X
            # y = self.Y
            z = np.linspace(self.Z[layer.get_slice(self.Z)][0],
                            self.Z[layer.get_slice(self.Z)][-1], len(z)*2)
            x = np.linspace(self.X[0], self.X[-1], len(self.X)*2)
            y = np.linspace(self.Y[0], self.Y[-1], len(self.Y)*2)
            pts = cartesian_product((z, x, y))
            vals = rgi(pts).reshape((len(z), len(x), len(y)))
            z_integral = intg.trapz(vals, x=z, axis=0)
            x_integral = intg.trapz(z_integral, x=x, axis=0)
            y_integral = intg.trapz(x_integral, x=y, axis=0)
            results[mat] = y_integral
        result = sum(results.values())
        return result


class SimulationGroup:

    def get_plane(self, scalar, plane, pval):
        """
        Gets data along a 2D plane/slice through the 3D data array for a given
        quantity

        :param str plane: Any of 'xy', 'yz', or 'xz'. Determines the plane
        along which the slice is taken
        :param int pval: The index along the final unspecified direction. If
        plane='xy' then index would index along the z direction.
        :param str quantity: The quantity whose data array you wish to take a
        line cut through
        """

        if plane == 'yz' or plane == 'zy':
            # z along rows, y along columns
            return scalar[:, pval, :]
        elif plane == 'xz' or plane == 'zx':
            # x along columns, z along rows
            return scalar[:, :, pval]
        elif plane == 'xy' or plane == 'yx':
            # x along rows, y along columns
            return scalar[pval, :, :]

    def diff_sq(self, x, y):
        """Returns the magnitude of the difference vector squared between two vector fields at each
        point in space"""
        # Calculate the magnitude of the difference vector SQUARED at each point in space
        # This is mag(vec(x) - vec(y))^2 at each point in space. This should be a 1D array
        # with # of elements = # sampling points
        mag_diff_vec = sum([np.absolute(v1 - v2)**2 for v1, v2 in zip(x, y)])
        return mag_diff_vec

    def get_slice(self, sim):
        """Returns indices for data that strip out air and substrate regions"""
        # TODO: This function is definitely not general. We need to get a list
        # of layers to exclude from the user. For now, just assume we want to
        # exclude the top and bottom regions
        # sorted_layers is an OrderedDict, and thus has the popitem method
        sorted_layers = sim.conf.sorted_dict(sim.conf['Layers'])
        first_layer = sorted_layers.popitem(last=False)
        last_layer = sorted_layers.popitem()
        # We can get the starting and ending planes from their heights
        start_plane = int(round(first_layer[1]['params'][
                          'thickness'] / sim.dz))
        end_plane = int(round(last_layer[1]['params'][
                        'thickness'] / sim.dz))
        return start_plane, end_plane

    def get_comp_vec(self, sim, field, start, end):
        """Returns the comparison vector"""
        # Compare all other sims to our best estimate, which is sim with highest number of
        # basis terms (last in list cuz sorting)

        # Get the proper file extension depending on the field.
        norm = 'norm'+field
        # Get the comparison vector
        vecs = [sim.data[field+comp][start:end] for comp in ('x', 'y', 'z')]
        normvec = sim.get_scalar_quantity('normE')
        normvec = normvec[start:end]**2
        return vecs, normvec

    def difference_squared(self, field, layer_name, sim1, sim2):
        """
        Computes the difference squared between the given field in the given
        layer of two simulations at each point in space, for example normE or
        genRate. If any vector components are passed in, this is also handled
        appropriately (for example Ex, Ey, or Ez)

        :param field: The field or field component you wish to compare
        :type field: str
        :param layer_name: The name of the layer in which you wish to make the
        comparison
        :type layer_name: str
        :return: An array containing the difference squared at each point in
        space.
        :rtype: np.ndarray
        """

        if field in ('Ex', 'Ey', 'Ez'):
            field = '{}_{}'.format(layer_name, field)
        self.log.info('Running the difference squared computation for quantity %s', field)
        layer_obj = self.layers[layer_name]
        # Set the reference sim
        # Get the comparison vector
        ref_arr = sim1.data[field][layer_obj.get_slice()]
        # For all other sims in the groups, compare to best estimate
        # and write to error file
        comp_arr = sim2.data[field][layer_obj.get_slice()]
        # Get the array containing the magnitude of the difference vector
        # at each point in space
        diff_sq = np.absolute(ref_arr - comp_arr)**2
        return diff_sq

    def global_error(self, field, exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell"""

        self.log.info('Running the global error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        if exclude:
            start, end = self.get_slice(self.sims[0])
            excluded = '_excluded'
        else:
            start = 0
            end = None
            excluded = ''
        # base = self.sims[0].conf['General']['base_dir']
        base = self.sims[0].conf['General']['results_dir']
        errpath = osp.join(base, 'globalerror_%s%s.dat' % (field, excluded))
        with open(errpath, 'w') as errfile:
            self.log.info('Computing global error for sweep %s', base)
            # Set reference sim
            ref_sim = self.sims[-1]
            # Get the comparison vector
            vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
            # For all other sims in the groups, compare to best estimate
            # and write to error file
            for i in range(0, self.num_sims - 1):
                sim2 = self.sims[i]
                vecs2, normvec2 = self.get_comp_vec(sim2, field, start, end)
                self.log.info("Computing global error between numbasis %i and numbasis %i",
                              ref_sim.conf['Simulation'][ 'params']['numbasis'],
                              sim2.conf['Simulation']['params']['numbasis'])
                # Get the array containing the magnitude of the difference vector at each point
                # in space
                mag_diff_vec = self.diff_sq(vecs1, vecs2)
                # Check for equal lengths between norm array and diff mag
                # array
                if len(mag_diff_vec) != len(normvec):
                    self.log.error( "The normalization vector has an incorrect number of elements!!!")
                    raise ValueError
                # Error as a percentage should be the square root of the ratio of sum of mag diff vec
                # squared to mag efield squared
                error = np.sqrt(np.sum(mag_diff_vec) / np.sum(normvec))
                errfile.write('%i,%f\n' % (sim2.conf['Simulation']['params']['numbasis'], error))
                sim2.clear_data()
            ref_sim.clear_data()

    def adjacent_error(self, field, exclude=False):
        """Computes the global error between the vector fields of two simulations. This is the sum
        of the magnitude squared of the difference vectors divided by the sum of the magnitude
        squared of the comparison efield vector over the desired section of the simulation cell.
        This computes error between adjacent sims in a sweep through basis terms."""

        self.log.info('Running the adjacent error computation for quantity %s', field)
        # If we need to exclude calculate the indices
        if exclude:
            start, end = self.get_slice(self.sims[0])
            excluded = '_excluded'
        else:
            start = 0
            end = None
            excluded = ''
        base = self.sims[0].conf['General']['results_dir']
        errpath = osp.join(base, 'adjacenterror_%s%s.dat' % (field, excluded))
        with open(errpath, 'w') as errfile:
            self.log.info('Computing adjacent error for sweep %s', base)
            # For all other sims in the groups, compare to best estimate
            # and write to error file
            for i in range(1, self.num_sims):
                # Set reference sim
                ref_sim = self.sims[i]
                # Get the comparison vector
                vecs1, normvec = self.get_comp_vec(ref_sim, field, start, end)
                sim2 = self.sims[i - 1]
                vecs2, normvec2 = self.get_comp_vec(sim2, field, start, end)
                self.log.info("Computing adjacent error between numbasis %i and numbasis %i",
                              ref_sim.conf['Simulation'][ 'params']['numbasis'],
                              sim2.conf['Simulation']['params']['numbasis'])
                # Get the array containing the magnitude of the difference vector at each point
                # in space
                mag_diff_vec = self.diff_sq(vecs1, vecs2)
                # Check for equal lengths between norm array and diff mag
                # array
                if len(mag_diff_vec) != len(normvec):
                    self.log.error("The normalization vector has an incorrect number of elements!!!")
                    raise ValueError
                # Error as a percentage should be thkkk square root of the ratio of sum of mag diff vec
                # squared to mag efield squared
                error = np.sqrt(np.sum(mag_diff_vec) / np.sum(normvec))
                # self.log.info(str(error))
                errfile.write('%i,%f\n' % (sim2.conf['Simulation']['params']['numbasis'], error))
                sim2.clear_data()
                ref_sim.clear_data()

