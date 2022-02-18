import time
import shelve
from memory_profiler import profile
import os
import numpy as np
import pandas as pd
import sys
# cdir = '/media/homehd/saustin/Exasim'
# exec(open('/media/homehd/saustin/Exasim/Installation/setpath.py').read())

# import Preprocessing

def find(list, substr):
    for idx, string in enumerate(list):
        if substr in string:
            return idx
    return None

def bisection_search(input_lines, elemtype1, numel, global_first_elem_idx):
    """
    Returns the 0-indexed first index where the element type changes
    """
    
    if numel == 1:
        return global_first_elem_idx

    half_idx = numel//2
    etype = int(input_lines[half_idx - 1].split()[1])   # 0-indexed
    
    if etype == elemtype1:
        global_first_elem_idx += half_idx
        return bisection_search(input_lines[half_idx:], elemtype1, numel-half_idx, global_first_elem_idx)
    elif etype != elemtype1:
        return bisection_search(input_lines[:half_idx], elemtype1, half_idx, global_first_elem_idx)

    

# @profile
def gmshcall(filename, ndim, elemtype, elemtype_face=None):
    """
    inputs:
    -pde: Exasim pde dictionary object
    -filename: .geo mesh filename
    -nd: dimensionality of geometry
    -elemtype: tri, quad, tet, hex, etc. See http://gmsh.info/dev/doc/texinfo/gmsh.pdf 9.1 for a complete list of possibilities

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

    """

    tb0 = time.perf_counter()

    if ('.geo' not in filename) and ('.msh' not in filename):
        raise ValueError(
            'Please specify a valid mesh file extension, either .geo or .msh, exiting!')

    if '.geo' in filename:
        # Format .msh version 3, silent output
        opts = "-format msh3 -v 0"

        # find gmsh executable
        gmsh = Preprocessing.findexec(pde['gmsh'], pde['version'])

        print("Gmsh mesh generator...\n")
        mystr = gmsh + " " + filename + ".geo -" + str(ndim) + " " + opts
        os.system(mystr)
        filename.replace('.geo', '.msh')

    with open(filename, 'r') as meshfile:
        lines = meshfile.readlines()

    # Version check
    if float(lines[1].split()[0]) != 3:
        raise ValueError('The Gmsh parser only accepts .msh files of version 3, exiting!')

    ptr = 0     # "Pointer" to current location in file line list - prevents searching through the whole file

    # Physical group names
    if elemtype_face is not None:
        ptr_tmp = find(lines, '$PhysicalNames')

        if ptr_tmp != None:    
            num_phys_grps = int(lines[ptr_tmp + 1].strip())
            phys_grp_dict = {}

            for physgrp in lines[ptr_tmp+2: ptr_tmp+2+num_phys_grps]:
                phys_grp = physgrp.split()
                if not int(phys_grp[0]) == ndim:
                    phys_grp_dict[phys_grp[2].strip('"')] = {'dim': int(phys_grp[0]), 'idx': int(phys_grp[1])}

            # Update ptr to end of block
            ptr = ptr_tmp + num_phys_grps + 2   

        else:   # No physical group names specified
            ptr_tmp = find(lines, '$Entities')
            if ptr_tmp:
                l = np.asarray(lines[ptr_tmp+1].strip().split(), dtype=np.int32)
                num_phys_grps = l[ndim-1]
                phys_grp_dict = {key:{'idx':key, 'dim':ndim-1} for key in np.arange(1,num_phys_grps+1)}
            else:
                print('No physical groups found')

    # Nodes
    # Update ptr to the start of the nodes list
    ptr += find(lines[ptr:], '$Nodes')
    nnodes = int(lines[ptr + 1].strip())

    nodes = pd.read_csv(filename, delimiter=' ', header=None, dtype=np.float64,
                        skiprows=ptr+2, nrows=nnodes).to_numpy()[:, 1:-1]

    # Elements
    # Update ptr to the start of the nodes list
    ptr = ptr + nnodes + 2 + find(lines[ptr+nnodes+2:], '$Elements')
    numel = int(lines[ptr + 1].strip())
    first_elem_line = ptr + 2

    elemtype1 = int(lines[first_elem_line].split()[1])

    elem_change_idx1 = bisection_search(lines[first_elem_line:first_elem_line+numel], elemtype1, numel, 0)

    # Assumption: Pulls the first two types of elements listed - this is usually the highest two dimensions.
    elem1 = pd.read_csv(filename, delimiter=' ', header=None, dtype=np.int64,
                        skiprows=first_elem_line, nrows=elem_change_idx1).to_numpy()

    elemtype2 = int(lines[first_elem_line+elem_change_idx1].split()[1])

    elem_change_idx2 = bisection_search(lines[first_elem_line+elem_change_idx1:first_elem_line+numel], elemtype2, numel-elem_change_idx1, 0)

    elem2 = pd.read_csv(filename, delimiter=' ', header=None, dtype=np.int64,
                        skiprows=first_elem_line+elem_change_idx1, nrows=elem_change_idx2).to_numpy()

    elemtype2 = elem2[0,1]

    if elemtype1 == elemtype:
        mesh_t = elem1[:,4:] - 1    # Subtracting 1 to make nodes 0-indexed
    elif elemtype2 == elemtype:
        mesh_t = elem2[:, 4:] - 1

    if elemtype_face:
        if elemtype1 == elemtype_face:
            surf_elem = np.delete(elem1, [0, 1, 3], axis=1)
        if elemtype2 == elemtype_face:
            surf_elem = np.delete(elem2, [0, 1, 3], axis=1)

        surf_elem[:,1:] -= 1 # Nodes must be 0-indexed

        
        # ## Desired data structure: replace the "node" list in each surface with a node array where each row has [node#, nx, ny, nz]
        # surf_elem = np.concatenate((surf_elem, np.zeros((surf_elem.shape[0], 3))), axis=1)

        # # Loop through each surface node
        # for i_elem, elem in enumerate(surf_elem):
        #     if elem[0] >= 42:        # Don't need the normal vectors for the outer boundary
        #         tri = elem[1:-3].astype(np.int)      # Nodes in the surface element (remember that the first column is the surface index)
                
        #         # if not np.array_equal(tri, np.array([175, 176, 545])):
        #         #     if not np.array_equal(tri, np.array([170, 543, 173])):
        #         #         if not np.array_equal(tri, np.array([34, 473, 148])):
        #         #             continue            
        #         # print(i_elem)
        #         # print(elem)

        #         if i_elem % 100 == 0:
        #             print(i_elem)

        #         # Compute the normal vector using the cross product # Have to get indexing right
        #         v1 = nodes[tri[1], :] - nodes[tri[0], :]
        #         v2 = nodes[tri[2], :] - nodes[tri[0], :]
        #         nvec = np.cross(v1, v2)
        #         nvec /= np.linalg.norm(nvec)

        #         # print(nvec)

        #         # Find its corresponding volume element in the main connectivity matrix.
        #         for tet in mesh_t:
        #             bl = tet == tri[:, None]
        #             if np.all(np.any(bl, axis=1)):
        #                 missing_node = np.setdiff1d(tet, tri)   # Set subtraction to find the last element in the tet that's not on the given face
        #                 # Ensures that the normal vector is pointint INTO the volume element/mesh, away from the surface
        #                 outward_vec = np.squeeze(nodes[missing_node, :] - nodes[tri[0], :])     # Pick any point in the surface element for the last one
        #                 # Append normal vector to the surface element array
        #                 surf_elem[i_elem, -3:] = np.squeeze(nvec*np.sign(np.dot(nvec, outward_vec)))
        #                 break

        ## Construction of the nodes list for each surface element is going to look a bit different. Instead of taking the "grab bag" of unique faces in the set of surface elements for a given surface,
        ## we'll need to loop through the data and find the surface element where the node matches.


        # print('*************')

        for key in phys_grp_dict:       # Key is the surface # (1-indexed)
            nodes_on_surf = surf_elem[surf_elem[:, 0] == phys_grp_dict[key]['idx'], :]      # This list includes the face index and normal vectors
            unique_nodes_on_surf = np.unique(nodes_on_surf[:, 1:4])        # surface node info is contained in cols 1, 2, and 3 of surf_elem
            
            if key >= 42:        # Don't need the normal vectors for the outer boundary
                # if key != 11:
                #     continue
                #     if key !=15: 
                #         if key != 16:
                #             continue

                # print(key)

                # Create a new zeros matrix whose length corresponds to the number of unique elements. Put the element indices running down the 0th column.
                surf_unique_pts_and_normals = np.zeros((unique_nodes_on_surf.shape[0], 4))
                surf_unique_pts_and_normals[:,0] = unique_nodes_on_surf

                pts = nodes[unique_nodes_on_surf,:]
                # print(pts)
                # exit()

                # test for zero row
                col_idx = np.where(np.all(pts==pts[0,:], axis=0))[0][0]

                # print(pts == pts[0, :])
                # print(col_idx)
                surf_unique_pts_and_normals[:, 1 +
                                            col_idx] = np.sign(pts[0, col_idx])

                if pts[0, col_idx] == 0:
                    surf_unique_pts_and_normals[:, 1 +col_idx] = -1       # Possible vulnerability here if the max BBox is at 0, NEED FIX!
                elif pts[0, col_idx] == 200:
                    surf_unique_pts_and_normals[:, 1 +col_idx] = -1       # Possible vulnerability here if the max BBox is at 0, NEED FIX!


                # print(surf_unique_pts_and_normals)
                # print()

                # if col_idx.shape[0] > 0:
                #     surf_unique_pts_and_normals[:,1+col_idx] = np.sign(pts[0, col_idx])
                # else:
                #     print("strange case")

                # # test for 1 row
                # else:
                #     col_idx = np.where(np.invert(np.any(pts-1, axis=0)))[0]
                #     if col_idx.shape[0] > 0:
                #         surf_unique_pts_and_normals[:, 1+col_idx] = 1

                # print(np.round(nodes_on_surf, 3))
                # print(unique_nodes_on_surf)
                # print(surf_unique_pts_and_normals)
                # print()
                # # exit()
                
                # For each row (node)
                # for i_node, unique_pt in enumerate(unique_nodes_on_surf):
                #     # Find the first occurence of the node in an elem and assign its normal vector to the point in that row.
                #     # if unique_pt != 148:
                #     #     continue
                #     # print(unique_pt)
                #     # print(nodes_on_surf[:, 1:4])
                #     node = np.argwhere(nodes_on_surf[:, 1:4] == unique_pt)[0][0]
                #     # print(node)

                #     surf_unique_pts_and_normals[i_node, 1:] = nodes_on_surf[node, -3:]

                # print(surf_unique_pts_and_normals)
                # print('')
                phys_grp_dict[key]['nodes'] = surf_unique_pts_and_normals    # 0-indexing moved to above
            else:   # Outer boundary
                phys_grp_dict[key]['nodes'] = unique_nodes_on_surf   # 0-indexing moved to above


        return nodes, mesh_t, phys_grp_dict
    else:
        return nodes, mesh_t


if __name__ == '__main__':
    # pde, mesh = Preprocessing.initializeexasim()

    # p, t, pg = gmshcall(pde, "h1.0_tets24.msh", 3, 4, 2)
    # p, t = gmshcall(pde, "h1.0_tets24.msh", 3, 4)

    # p, t, pg = gmshcall(pde, "h0.01_tets4504962.msh", 3, 4, 2)
    # p, t = gmshcall(pde, "h0.01_tets4504962.msh", 3, 4)
    # p, t = gmshcall(pde, "h0.5_tets101.msh", 3, 4)

    filename = sys.argv[1]
    p, t, pg = gmshcall(filename, 3, 4, 2)

    # print(p.T)
    # print()
    # print(t.T)
    # print()
    # print(pg)

    # print(p.transpose())
    # print(t.transpose())

    # for key in pg:
    #     print(key)
    #     print(pg[key]['nodes'])
    #     print()

    with open(filename + 'NML_QUICK.npy', 'wb') as f:
        np.save(f, p)
        np.save(f, t)
        np.save(f, pg)

    print('Saved to ' + filename + 'NML_QUICK.npy')
