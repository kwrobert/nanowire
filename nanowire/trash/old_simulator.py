

class Simulator:

    @ureg.wraps(('', '', ''),
                None)
    def compute_fields_by_layer(self, sample_dict):
        """
        Compute fields within each layer such that a z sample falls exactly on
        the start and end of a layer. The purpose of this it to reduce
        integration errors

        :param sample_dict: A dictionary whose keys are strings of layer names
        and whose values are integers indicating the number of z sampling
        points to use in that layer
        :type sample_dict: dict[str] int
        :return: A dictionary whose keys are layers names, and whose values are
        dictionaries containing all the field components
        :rtype: dict[str] dict
        """

        results = {}
        for lname, layer in self.layers.items():
            if lname not in sample_dict:
                self.log.info("Layer %s not in sample dict, skipping", lname)
                continue
            if isinstance(sample_dict[lname], int):
                z = np.linspace(layer.start, layer.end, sample_dict[lname])
            else:
                args = [layer.start, layer.end, *sample_dict[lname]]
                z = arithmetic_arange(*args)
            self.log.info("Computing fields in layer %s using %i samples",
                          lname, len(z))
            Ex, Ey, Ez, Hx, Hy, Hz = self.compute_fields(zvals=z)
            # results[lname] = {'Ex':Ex, 'Ey':Ey, 'Ez':Ez, 'Hx':Hx, 'Hy':Hy,
            #                   'Hz':Hz}
            results.update({'{}_{}'.format(lname, fname): arr for fname, arr in
                           (('Ex', Ex), ('Ey', Ey), ('Ez', Ez), ('Hx', Hx),
                            ('Hy', Hy), ('Hz', Hz))})
        return results
