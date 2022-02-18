#! /bin/bash

path_to_geo=/media/homehd/saustin/lightning_research/3D_poisson_fem/data/geo_files/cube.geo
save_dir=/media/homehd/saustin/lightning_research/3D_poisson_fem/data/mesh_gmsh_format

# Array of element sizes that are desired to generate
mesh_array=(1.0 0.5 0.1 0.05 0.02 0.015 0.01 0.005)

for mesh_size in ${mesh_array[@]}
do
    sed -i "4s/.*/h = $mesh_size;/" $path_to_geo
    gmsh $path_to_geo -3 -o $save_dir/h$mesh_size.msh -format msh3 > stats.txt
    nelem=$(cat stats.txt | grep -oP '[[:digit:]]+(?= tetrahedra created)')
    mv $save_dir/h$mesh_size.msh $save_dir/h${mesh_size}_tets${nelem}.msh
    echo "Wrote h${mesh_size}_tets${nelem}.msh"
done