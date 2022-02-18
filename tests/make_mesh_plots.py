import pyvista as pv
import numpy as np
import shelve
import vtk

dir = './mesh_size_sweep/'

pl = pv.Plotter(shape=(1, 7), border=False)


for count, mesh in enumerate(['1', '2', '3', '4', '5', '6', '7']):
    mesh_filename = './mesh_size_sweep/mesh_processed_'+mesh
    sol_filename = './mesh_size_sweep/sol_mesh_processed_'+mesh

    with shelve.open(mesh_filename) as shelf:
        grid_data = shelf['grid_data']
        conn_mat = shelf['tet_data']-1
        tri_data = shelf['tri_data']-1
        sizes_dict = shelf['sizes']
        surf_data = shelf['surfaces']
        for key in surf_data:
            surf_data[key] -= 1  # fix!

    with shelve.open(sol_filename) as shelf:
        u = shelf['u']

    theta = np.concatenate(
        (4*np.ones((len(conn_mat), 1)), conn_mat[:, 1:]), axis=1)
    theta = np.hstack(theta).astype(int)

    # each cell is a VTK_TETRA
    celltypes = np.empty(len(conn_mat), dtype=np.uint8)
    celltypes[:] = vtk.VTK_TETRA

    # Build mesh
    mesh = pv.UnstructuredGrid(theta, celltypes, grid_data)

    # Generate plot
    # plot = pv.Plotter(window_size=[2000, 1500])
    pl.set_background('white')
    pl.subplot(0, count)

    # For now, set the "solution" field to be the point's x-value and associate with the mesh
    mesh.point_data['sol_data'] = u

    # Normal vector for clipping
    normal = (0, 1, 0)
    origin = (0, 0, 0)
    # clipped_mesh = mesh.clip(normal)

    pl.add_mesh(mesh, scalars='sol_data',
                show_edges=True,
                line_width=0.75,
                )

pl.show(cpos='xy')
