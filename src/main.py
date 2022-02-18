import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as splinalg
import shelve
import sys
import time
from viz import plot_sol
import pathlib

"""
#TODO
Case run directory
Subdirectories for inputs and output
In it goes the geometry, mesh, and exported mesh
Auxiliary script contains paths to the inputs and outputs, and main code path
Change mulitple file loads to .npz instead of .npy

IMPORTANT: Need to calculate Ex, Ey, Ez (interpolate the potential solution onto a uniform grid to calculate it?)

# Fix bounding box/unit issue in gmsh - automate the bounding box generation?
# Automate picking the normals to be more robust

E-field definition - can vary
"""

def Ax(x):
    """
    Needed to compute the matrix-vector product for the conjugate gradient solver
    """
    return A.dot(x)

# Turn on for plotting
pltflag=1

wc_tb0 = time.perf_counter()
cpu_tb0 = time.process_time()

# First path is the path to the mesh
# Absolute filepaths only!
mshfile = pathlib.Path(sys.argv[1])
outdir = pathlib.Path(sys.argv[2])

outfile_sol = str(mshfile.stem).replace('.msh', '') + '_sol.npy'
outfile_sol = outdir / outfile_sol

print('Reading mesh from ' + str(mshfile))

with open(mshfile, 'rb') as f:
    # Order needs to match that of the gmshcall.py
    p = np.load(f)
    t = np.load(f)
    pg = np.load(f, allow_pickle=True).item()

genA = 0

if sys.argv[3] == '-b':
    genA = 1

if genA:
    # Construct A, no BCs

    outfile_mat = str(mshfile.stem).replace('.msh', '') + '_mat'
    outfile_mat = outdir / outfile_mat

    n_nodes = p.shape[0]

    A = lil_matrix((n_nodes, n_nodes))
    F = np.zeros((n_nodes, 1))

    # Interior

    for i, elem in enumerate(t):
        if not i % 1000:  # Prints status every 1000 elements
            print(i)
        phi = elem      # Take all but the first element in the row

        # print(elem)
        # print(p[phi, :])

        coords = p[phi, :]  # Combined advanced/basic indexing, returns a copy
        Q = np.concatenate((np.ones((4, 1)), coords), axis=1)
        Q_inv = np.linalg.inv(Q)
        vol = abs(np.linalg.det(Q))/2
        A_local = vol*(Q_inv[1:2, :].transpose() * Q_inv[1:2, :] + Q_inv[2:3,
                    :].transpose() * Q_inv[2:3, :] + Q_inv[3:, :].transpose() * Q_inv[3:, :])
        A[phi[:, None], phi] += A_local

    # save_sparse_csr(str(outfile_mat) + 'A.npz', A.asformat('csr'))
    # save_sparse_csr(str(outfile_mat) + 'F.npz', F.asformat('csr'))

    with open(str(outfile_mat) + 'F.npy', 'wb') as file:
        np.save(file, F)

    scipy.sparse.save_npz(str(outfile_mat) + 'A.npz', A.asformat('csr'))
    # scipy.sparse.save_npz(str(outfile_mat) + 'F.npz', F.asformat('csr'))

    print('Saved A and F to '+ str(outfile_mat))


else:
    # Load A from file
    # First path is the path to the matrix
    matfile = pathlib.Path(sys.argv[3])

    print('Reading matrix from ' + str(matfile))
    print('loaded A')

    with open(str(matfile) + 'F.npy', 'rb') as file:
        F = np.load(file)

    A = scipy.sparse.load_npz(str(matfile) + 'A.npz')
    # F = sparse_matrix = scipy.sparse.load_npz(str(matfile) + 'A.npz')

A = A.asformat('lil')
# BCs code:     0 = Dirichlet, 1 = Neumann
bc_array = np.zeros((len(pg.keys()), 3))
bc_array[:, 0] = np.arange(bc_array.shape[0])
bc_array[:41, 1] = 0
bc_array[:41, 2] = 0
bc_array[41:, 1] = 1
bc_array[41:, 2] = 1        # Note that the global Neumann BC can be overwritten later on the element by element leve

# Set external electric field vector
E_field = np.array([0, 0, 1])

for surface in bc_array:
    surface_idx = surface[0]
    if surface[1] == 0:     # Dirichlet
        for node in pg[surface_idx+1]['nodes']:     # bc_array is 0-indexed, while the face indices from the mesh data structure pg are 1-indexed
            A[node,:] = 0
            A[node, node] = 1  # Broadcast
            F[node, :] = surface[2]

    elif surface[1] == 1:        # Neumann
        for node in pg[surface_idx+1]['nodes']:
            if node.shape[0] > 1:   # If the normals were pre-specified
                F[int(node[0]), :] += np.dot(E_field, node[-3:])     # E-field dot the surface normal at that node
            else:
                print('Strange case while assigning Neumann BC, exiting!')
                exit()

wc_matrix_build = time.perf_counter() - wc_tb0
cpu_matrix_build = time.process_time() - cpu_tb0

print('Finished assembling matrix')

wc_tb1 = time.perf_counter()
cpu_tb1 = time.process_time()

# Solver
A = A.asformat('csr')
lin = splinalg.LinearOperator(A.shape, Ax)

# Solve with CG - A can be sparse, but b needs to be dense
try:
    # F_dense = F.todense()
    u = splinalg.cg(lin, F)
    if not u[1]:    # Successful exit
        u = u[0]
    # wc_solve_cg = time.perf_counter() - wc_tb2
    # cpu_solve_cg = time.process_time() - cpu_tb2
    print('Successfully solved with CG...')
    
except Exception as e:
    u = 'NaN'
    print(e.message), e.args
    # wc_solve_backslash = 'NaN'
    # cpu_solve_backslash = 'NaN'

# print(wc_solve_cg)
# print(cpu_solve_cg)

# Saves solution to disk
with open(outfile_sol, 'wb') as f:
    np.save(f, p)
    np.save(f, t)
    np.save(f, u)

print('Saved solution to ' + str(outfile_sol))

if pltflag:
    print('Plotting')
    plot_sol(p, t, u)


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
