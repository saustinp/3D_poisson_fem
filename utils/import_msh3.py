import numpy as np
import sys
import re


def import_msh3(filename, ndim, elemtype, elemtype_face=None):
    """
    inputs:
    - filename: mesh filename, expects file extension as well
    - nd: dimensionality of geometry
    - elemtype: element type of the elements that you would like to pull from the mesh. Expects an integer value corresponding to elm-type in Sec 9.1 of http://gmsh.info/dev/doc/texinfo/gmsh.pdf. Note that in the majority of cases, you will want to use the elements of the highest dimensionality in the mesh (for example, tets for a 3D mesh even though both tets and triangular surface elements are avialable). In some cases, you may want to only access the surface elements.
    - elemtype_face: element type of the surface element that you will use to assign the boundary conditions/physical groups. Same format as elemtype

    According to the .msh version 3 standard, the ASCII mesh file will conform to the following format. See Sec 9.1 of http://gmsh.info/dev/doc/texinfo/gmsh.pdf for information on version 4.1 (can't find a reference for V3).
    $MeshFormat
    <Mesh format information>
    $EndMeshFormat
    $PhysicalNames
    <Physical groups and names>
    $EndPhysicalNames
    $Entities
    <Physical entity informatin>
    $EndEntities
    $Nodes
    <Node coordinates>
    $EndNodes
    $Elements
    <Mesh elements, includes both surface (2D) and volume (3D) elements
    $EndElements

    For the purposes of the ultimate mesh data structure, we don't care about the Entities or mesh elements of dimension less than the highest dimension (surface elements in a 3D mesh, for example)

    NOTE: Modified from the Exasim version to remove the mesh generation functionality - it is assumed that this is already taken care of in another script
    """

    if '.msh' not in filename:
        raise ValueError('Please specify a .msh file, exiting!')

    case = 'null'
    firstline = 0

    # Initialize mesh and surface triangulation data structures as empty for now, will populate later
    mesh_t = 0
    if elemtype_face:
        face_tria = 0

    with open(filename) as meshfile:
        for line in meshfile:
            line = line.strip()

            # "State machine" case switcher for parser
            if (not firstline) and ('$' in line):   # Header or footer detected
                if 'End' in line:               # Only the following header/footer labels are important in the mesh file
                    case = 'null'
                    continue
                elif 'MeshFormat' in line:
                    case = 'mesh_format'
                    continue
                elif 'PhysicalNames' in line:
                    case = 'physical_names'
                    firstline = 1
                    continue
                elif 'Nodes' in line:
                    case = 'nodes'
                    firstline = 1
                    continue
                elif 'Elements' in line:
                    case = 'elements'
                    firstline = 1
                    continue
                continue   # No need to execute following code if header/footer parsed
            # Make sure case of ignored data block (entities for example) are treated correctly

            # Parser state machine
            if case != 'null':
                if case == 'mesh_format':
                    version = np.array(line.split()).astype(float)[0]
                    if version != 3:
                        raise ValueError(
                            'The Gmsh parser only accepts .msh files of version 3, exiting!')
                elif case == 'physical_names':
                    if firstline:   # Create physical groups dict
                        phys_grp_dict = {}
                        firstline = 0
                    else:
                        phys_grp = line.split()
                        if not int(phys_grp[0]) == ndim:
                            phys_grp_dict[phys_grp[2].strip('"')] = {'dim': int(
                                phys_grp[0]), 'idx': int(phys_grp[1])}
                elif case == 'nodes':
                    if firstline:
                        npts = int(line)
                        mesh_p = np.zeros((npts, ndim))
                        firstline = 0
                    else:
                        pt = np.array(line.split()).astype(float)
                        # Offset by 1 to avoid including the point index
                        mesh_p[int(pt[0])-1, :] = pt[1:ndim+1]
                elif case == 'elements':
                    if firstline:
                        nelem = int(line)
                        firstline = 0
                    else:
                        # Only need to store connectivity matrix as 32 bit integers because the point indices will not exceed the the storage capacity for a 32 bit int
                        elem = np.array(line.split()).astype(np.int32)

                        # The number of elements given at the top of the elements section does not distinguish between volume and surface elements. Thus, arrays for the volume and surface triangulations are each preallocated with the number of combined elements, and trimmed at the end.
                        # Only want to further manipulate elements that are of the desired element type
                        if elem[1] == elemtype:
                            # Only create array if it hasn't been initialized (see declaration of mesh_t at the top of the function)
                            if type(mesh_t) == int:
                                # The number of points per element is stored in the fourth value of the element entry in .msh V3. Electing to not store the physical_group ID for volume elements
                                mesh_t = np.zeros(
                                    (nelem, elem[3])).astype(np.int32)
                            mesh_t[elem[0]-1, :] = elem[4:]

                        # Associate faces with physical groups if the flag is passed in to the function
                        elif elemtype_face and (elem[1] == elemtype_face):
                            if type(face_tria) == int:
                                # Stores both the nodes in the surface elements and the physical group
                                face_tria = np.zeros(
                                    (nelem, elem[3]+1)).astype(np.int32)
                            face_tria[elem[0]-1,
                                      :] = np.delete(elem, [0, 1, 3])

    # Remove rows of both surf_triangulation and the mesh that are all zeros and subtract 1 to obtain zero-indexed triangulation
    mesh_p = mesh_p.transpose()
    mesh_t = mesh_t[np.all(mesh_t, axis=1)].transpose() - \
        1     # Nodes are zero-indexed
    face_tria = face_tria[np.all(
        face_tria, axis=1)]

    # Clean up the surf_data dict
    for key in phys_grp_dict:
        phys_grp_dict[key]['nodes'] = np.unique(
            face_tria[face_tria[:, 0] == phys_grp_dict[key]['idx'], 1:])-1      # Nodes are zero-indexed

    return mesh_p, mesh_t, phys_grp_dict


if __name__ == '__main__':

    import pathlib
    import shelve

    # arg1: .msh file path
    # arg2: output director. Absolute paths only!

    # msh_file = sys.argv[1]
    # write_dir = sys.argv[2]

    msh_file = '/media/homehd/saustin/lightning_research/3D_poisson_fem/data/mesh_gmsh_format/h1.0_tets24.msh'
    write_dir = '/media/homehd/saustin/lightning_research/3D_poisson_fem/data/mesh_processed'

    mshfile = pathlib.Path(msh_file).stem

    p, t, pg = import_msh3(msh_file, 3, 4, 2)

    print(p)
    print('')
    print(t)
    print('')
    print(pg)

    with shelve.open('mesh') as shelf:
        shelf['p'] = p
        # shelf['t'] = t
        # shelf['pg'] = pg
