import re
import pathlib

# This is a test script to try out the many different ways to parse and return the filename from a file path, without the extension. But, can easily be modified to use the extension too.

filepath = '/media/homehd/saustin/lightning_research/3D_poisson_fem/data/mesh_gmsh_format/h0.015_tets1363205.msh'

print('With regex')
r = re.compile(r'[^\/]+(?=\.)')
match = r.search(filepath)
if match:
    print(match.group(0))

print('')
print('With splitting')
print(filepath.split('/')[-1].split('.msh')[0])

print('')
print('With pathlib')
print(pathlib.Path(filepath).stem)
# Pathlib is probably the most robust method