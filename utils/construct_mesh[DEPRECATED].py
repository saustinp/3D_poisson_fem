import numpy as np
import shelve
import re
import sys
import time

mesh_dir = sys.argv[1]
meshfile = sys.argv[2] + '.bdf'
outfile = mesh_dir + '/mesh_processed_' + sys.argv[2]

sizes_dict = {'GRID': 0, 'CTETRA': 0, 'CTRIA3': 0, 'PSHELL': 0, 'PSOLID': 0}
remaining = len(sizes_dict)
r = re.compile('\$ NUM (\w+) (\d+)')
grid1_re = re.compile('GRID,1')
tet1_re = re.compile('CTETRA,1')
tria1_re = re.compile('CTRIA3,' + str(sizes_dict['CTETRA']+1))

grid_re = re.compile('GRID,(\d+),,([\d\.]+),([\d\.]+),([\d\.]+)')
tet_re = re.compile('CTETRA,(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)')
tri_re = re.compile('CTRIA3,(\d+),(\d+),(\d+),(\d+),(\d+)')
surf_re = re.compile('PSHELL,(\d+)')


# Build sizes dict
wc_tb0 = time.perf_counter()
cpu_tb0 = time.process_time()

mesh_file = open(mesh_dir + '/' + meshfile, mode='r')
# mesh_file = open('old/cube_mesh_coarse8.bdf', mode='r')

while remaining:
    line = mesh_file.readline()
    match = re.search(r, line)
    if match != None:
        sizes_dict[match.group(1)] = int(match.group(2))
        remaining -= 1

# print('Object sizes: ' + repr(sizes_dict))

grid_data = np.zeros((sizes_dict['GRID'], 3))
tet_data = np.zeros((sizes_dict['CTETRA'], 5)).astype(int)
tri_data = np.zeros((sizes_dict['CTRIA3']-sizes_dict['CTETRA'], 4)).astype(int)
surf_data = np.zeros((sizes_dict['PSHELL']))
surface_node_list = {}

count = 0
for line in mesh_file:
    count += 1

    # Try GRID match
    mo = re.search(grid_re, line)
    if mo != None:
        data = np.array(mo.groups()).astype(float)
        grid_data[int(data[0]-1),:] = data[1:]

    # Try CTETRA match
    mo = re.search(tet_re, line)
    if mo != None:
        data = np.array(mo.groups()).astype(int)
        tet_data[data[0]-1, :] = data[1:]

    # Try CTRIA3 match
    mo = re.search(tri_re, line)
    if mo != None:
        data = np.array(mo.groups()).astype(int)
        tri_data[data[0]-sizes_dict['CTETRA'] - 1, :] = data[1:]

    # Try PSHELL match
    mo = re.search(surf_re, line)
    if mo != None:
        data = np.array(mo.groups()).astype(int)
        surf_data[data[0]-1] = data[0]
        surface_node_list[data[0]] = []     # Initialize dictionary location

    if 'ENDDATA' in line:
        break

mesh_file.close()

# Sorting and extracting surface nodes

# sort_idx = np.argsort(tri_data[:,0])
# sorted = tri_data[sort_idx,:] # Combined advanced/basic indexing

for surf in surf_data:
    surface_node_list[surf] = np.unique(tri_data[tri_data[:,0]==surf,1:])

# print(surface_node_list.keys())
# print(surface_node_list)

# For future: check for corner/edge nodes

# print('Saving data...')
with shelve.open(outfile) as shelf:
    shelf['grid_data'] = grid_data
    shelf['tet_data'] = tet_data
    shelf['tri_data'] = tri_data
    shelf['sizes'] = sizes_dict
    shelf['surfaces'] = surface_node_list

wc_time = time.perf_counter() - wc_tb0
cpu_time = time.process_time() - cpu_tb0

with open("mesh_convert_time.txt", "a") as file:
    file.write('File,' + meshfile + ',wall_clock,' + repr(wc_time) + ',cpu_time,' + repr(cpu_time) + '\n')

with open('mesh_sizes.txt', 'a') as file:
    file.write(repr(sizes_dict) + '\n')

#TODO: remove the human input to the file in the beginning