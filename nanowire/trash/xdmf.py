"""
This used to be a method on the Simulator object.

Whenever I tried to import HDF5 file using this XDMF xml file Paraview would
crash so I gave up. Idk why I even wanted to use Paraview anyway
"""

def write_xdmf(self):
    """
    Writes an XMDF file for the electric fields, allowing import into
    Paraview for visualization
    """

    grid = E.Grid
    domain = E.Domain
    topo = E.Topology
    geo = E.Geometry
    ditem = E.DataItem
    attr = E.Attribute
    base = 'sim.hdf5:/sim_{}'.format(self.id[0:10])
    dims = '{} {} {}'.format(self.zsamps, self.xsamps, self.ysamps)
    doc = (
    E.Xdmf({'Version': '3.0'},
        domain(
            grid({'GridType': 'Uniform', 'Name': 'FullGrid'},
                topo({'TopologyType': '3DRectMesh'}),
                geo({'GeometryType': 'VXVYVZ'},
                   ditem(base+'/xcoords', {'Name': 'xcoords',
                                           'Dimensions': str(self.xsamps),
                                           'NumberType': 'Float',
                                           'Precision': '4',
                                           'Precision': '4',
                                           'Format': 'HDF',
                                           'Compression': 'Zlib'}),
                   ditem(base+'/ycoords', {'Name': 'ycoords',
                                           'Dimensions': str(self.ysamps),
                                           'NumberType': 'Float',
                                           'Precision': '4',
                                           'Format': 'HDF',
                                           'Compression': 'Zlib'}),
                   ditem(base+'/zcoords', {'Name': 'zcoords',
                                           'Dimensions': str(self.zsamps),
                                           'NumberType': 'Float',
                                           'Precision': '4',
                                           'Format': 'HDF',
                                           'Compression': 'Zlib'}),
                ),
                attr({'Name': 'Electric Field Components', 'AttributeType': 'Scalar',
                      'Center': 'Node'},
                    ditem(base+'/Ex', {'Dimensions': dims}),
                    ditem(base+'/Ey', {'Dimensions': dims}),
                    ditem(base+'/Ez', {'Dimensions': dims})
                )
            )
        )
    )
    )
    path = osp.join(self.path, 'sim.xdmf')
    with open(path, 'wb') as out:
        out.write(etree.tostring(doc, pretty_print=True))

def save_data(self):
    """Saves the self.data dictionary to an npz file. This dictionary
    contains all the fields and the fluxes dictionary"""

    start = time.time()
    if self.hdf5 is None:
        self.open_hdf5()
    if self.conf['General']['save_as'] == 'npz':
        self.log.debug('Saving fields to NPZ')
        if self.conf['General']['adaptive_convergence']:
            if self.converged[0]:
                out = osp.join(self.path, 'converged_at.txt')
            else:
                out = osp.join(self.path, 'not_converged_at.txt')
            self.log.debug('Writing convergence file ...')
            with open(out, 'w') as outf:
                outf.write('{}\n'.format(self.converged[1]))
        out = osp.join(self.path, self.conf["General"]["base_name"])
        # Compression adds a small amount of time. The time cost is
        # nonlinear in the file size, meaning the penalty gets larger as the
        # field grid gets finer. However, the storage gains are enormous!
        # Compression brought the file from 1.1G to 3.9M in a test case.
        # I think the compression ratio is so high because npz is a binary
        # format, and all compression algs benefit from large sections of
        # repeated characters
        np.savez_compressed(out, **self.data)
    elif self.conf['General']['save_as'] == 'hdf5':
        compression = self.conf['General']['compression']
        if compression:
            # filter_obj = tb.Filters(complevel=8, complib='blosc')
            filter_obj = tb.Filters(complevel=4, complib='zlib')
        gpath = '/sim_'+self.id[0:10]
        for name, arr in self.data.items():
            # Check for recarrays first because they are subclass of
            # ndarrays
            if isinstance(arr, np.recarray):
                self.log.info("Saving record array %s", name)
                num_rows = len(list(self.conf['Layers'].keys()))
                table = self.hdf5.create_table(gpath, name,
                                               description=arr.dtype,
                                               expectedrows=num_rows,
                                               createparents=True)
                row = table.row
                fields = arr.dtype.names
                for record in arr:
                    for (i, el) in enumerate(record):
                        row[fields[i]] = el
                    row.append()
                table.flush()
            elif isinstance(arr, np.ndarray):
                self.log.debug("Saving array %s", name)
                if compression:
                    self.hdf5.create_carray(gpath, name, createparents=True,
                                       atom=tb.Atom.from_dtype(arr.dtype),
                                       obj=arr, filters=filter_obj)
                else:
                    self.hdf5.create_array(gpath, name, createparents=True,
                                      atom=tb.Atom.from_dtype(arr.dtype),
                                      obj=arr)

        # Write XMDF xml file for importing into Paraview
        self.write_xdmf()
        end = time.time()
        diff = end - start
        self.log.info('Time to write data to disk: %f seconds', diff)
                # # Save the field arrays
                # self.log.debug('Saving fields to HDF5')
                # path = '/sim_'+self.id[0:10]
                # for name, arr in self.data.items():
            # self.log.debug("Saving array %s", name)
            # tup = ('create_array', (path, name),
                #    {'compression': self.conf['General']['compression'],
                #     'createparents': True, 'obj': arr,
                #     'atom': tb.Atom.from_dtype(arr.dtype)})
            # self.q.put(tup, block=True)
        # # Save the flux dict to a table
        # self.log.debug('Saving fluxes to HDF5')
        # self.log.debug(self.flux_dict)
        # tup = ('create_flux_table', (self.flux_dict, path, 'fluxes'),
               # {'createparents': True,
                # 'expectedrows': len(list(self.conf['Layers'].keys()))})
        # self.q.put(tup, block=True)
        self.hdf5.flush()
    else:
        raise ValueError('Invalid file type specified in config')
