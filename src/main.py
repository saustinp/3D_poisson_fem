import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as splinalg
import shelve
# import pyvista as pv
# import vtk
import sys
import time

mesh_dir = sys.argv[1]
meshfile = mesh_dir + '/' + sys.argv[2]
outfile = mesh_dir + '/sol_' + sys.argv[2]

# Turn on for plotting
pltflag=0

wc_tb0 = time.perf_counter()
cpu_tb0 = time.process_time()

print(meshfile)
with shelve.open(meshfile) as shelf:
    grid_data = shelf['grid_data']
    conn_mat = shelf['tet_data']-1
    tri_data = shelf['tri_data']-1
    sizes_dict = shelf['sizes']
    surf_data = shelf['surfaces']
    for key in surf_data:
        surf_data[key] -= 1 # fix!

n_nodes = grid_data.shape[0]

A = lil_matrix((n_nodes, n_nodes))
F = lil_matrix((n_nodes, 1))

# Hard-coding values for the dirichlet BCs and Neumann_BCs
BCs = {1: ('Dirichlet', 0),
        2: ('Neumann', 0),
        # Note that the boundary points are overwritten as Neumann in this case
        3: ('Neumann', 0),
        4: ('Neumann', 0),
        5: ('Neumann', 0),
        6: ('Dirichlet', 1)}

# Interior
print(conn_mat.shape[0])
count = 0

for i, elem in enumerate(conn_mat):
    if not i%10000:
        print(i)
    phi = elem[1:]      # Take all but the first element in the row
    coords = grid_data[phi, :]  # Combined advanced/basic indexing, returns a copy
    Q = np.concatenate((np.ones((4,1)), coords), axis=1)
    Q_inv = np.linalg.inv(Q)
    vol = abs(np.linalg.det(Q))/2
    A_local = vol*(Q_inv[1:2,:].transpose() * Q_inv[1:2,:] + Q_inv[2:3,:].transpose() * Q_inv[2:3,:] + Q_inv[3:,:].transpose() * Q_inv[3:,:])
    A[phi[:, None], phi] += A_local

# BCs
# print(F.shape)
for surf in BCs:
    # print(surf)
    if BCs[surf][0] == 'Dirichlet':
        for node in surf_data[surf]:
            A[node,:] = 0    # Might be an issue if a copy is generated instead of in-place
            A[node, node] = 1  # Broadcast
            F[node,:] = BCs[surf][1]

    elif BCs[surf][0] == 'Neumann':
        for node in surf_data[surf]:
            # Need to write this as = for now, but change to += in the future for a general Poisson solver, not restricted to Laplace
            F[node,:] = BCs[surf][1]

wc_matrix_build = time.perf_counter() - wc_tb0
cpu_matrix_build = time.process_time() - cpu_tb0

print('Finished assembling matrix....')

wc_tb1 = time.perf_counter()
cpu_tb1 = time.process_time()

# Solve with "backslash"
try:
    u = splinalg.spsolve(A.asformat('csc'), F.asformat('csc'))
    wc_solve_backslash = time.perf_counter() - wc_tb1
    cpu_solve_backslash = time.process_time() - cpu_tb1
    print('Successfully solved with backslash....')
except Exception as e:
    u='NaN'
    print(e.message), e.args
    wc_solve_backslash = 'NaN'
    cpu_solve_backslash = 'NaN'

wc_tb2 = time.perf_counter()
cpu_tb2 = time.process_time()

A = A.asformat('csr')

def Ax(x):
    return A.dot(x)

lin = splinalg.LinearOperator(A.shape, Ax)

# Solve with CG - A can be sparse, but b needs to be dense
try:
    # F_dense = F.todense()
    # u_cg = splinalg.cg(A.asformat('csc'), F_dense)
    # if not u_cg[1]:    # Successful exit
    #     u_cg = u_cg[0]
    # wc_solve_cg = time.perf_counter() - wc_tb2
    # cpu_solve_cg = time.process_time() - cpu_tb2
    # print('Successfully solved with CG...')

    F_dense = F.todense()
    u_cg = splinalg.cg(lin, F_dense)
    if not u_cg[1]:    # Successful exit
        u_cg = u_cg[0]
    wc_solve_cg = time.perf_counter() - wc_tb2
    cpu_solve_cg = time.process_time() - cpu_tb2
    print('Successfully solved with CG...')
    
except Exception as e:
    u_cg = 'NaN'
    print(e.message), e.args
    wc_solve_backslash = 'NaN'
    cpu_solve_backslash = 'NaN'

# Writes stats out to data file
log=False
if log:
    with open("solution_benchmarking.txt", "a") as file:
        file.write('File,' + meshfile + 
                    ',wc_matrix_build,' + repr(wc_matrix_build) +
                    ',cpu_matrix_build,' + repr(cpu_matrix_build) + 
                    ',wc_solve_backslash,' + repr(wc_solve_backslash) +
                    ',cpu_solve_backslash,' + repr(cpu_solve_backslash) +
                    ',wc_solve_cg,' + repr(wc_solve_cg) +
                    ',cpu_solve_cg,' + repr(cpu_solve_cg) + '\n')

print(wc_solve_cg)
print(cpu_solve_cg)

# Saves solution to disk
with shelve.open(outfile) as shelf:
    shelf['u'] = u
    shelf['u_cg'] = u_cg

# if pltflag:
#     # Plotting

#     theta = np.concatenate((4*np.ones((len(conn_mat),1)), conn_mat[:,1:]), axis=1)
#     theta = np.hstack(theta).astype(int)

#     # each cell is a VTK_TETRA
#     celltypes = np.empty(len(conn_mat), dtype=np.uint8)
#     celltypes[:] = vtk.VTK_TETRA

#     # Build mesh
#     mesh = pv.UnstructuredGrid(theta, celltypes, grid_data)

#     # Generate plot
#     plot = pv.Plotter(window_size=[2000,1500])
#     # plot.set_background('white')

#     # For now, set the "solution" field to be the point's x-value and associate with the mesh
#     mesh.point_data['sol_data'] = u

#     # Normal vector for clipping
#     normal = (0, 1, 0)
#     origin = (0, 0, 0)
#     # clipped_mesh = mesh.clip(normal)


#     plot.add_mesh(mesh, scalars = 'sol_data',
#                         show_edges = True,
#                         line_width=0.75,
#                         )

#     plot.show(cpos='xy')

#     # Conj grad w/ keeping matrix structure
#     # Conj grad w/o keeping matrix structure
    # Add in second order elements
    # We

    # Conj grad - works for SPD matrices
    # For linear elements use a diagonal preconditioner - the identity
    # Lumped mass matrix - 1/3 1/3 1/3
    # For quadratic elemtns you're going to have to get the mass matrix and add the abs terms to the diagonal
    # Sum of the mass matrix is the voluem
    # Preconditioning with quadratic elements is harder
