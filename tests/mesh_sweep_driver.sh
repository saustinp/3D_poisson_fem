#!/bin/bash

mesh_dir=./mesh_size_sweep

# Pulls all mesh filenames (without extension)
readarray mesh_filename_array < <(ls ./mesh_size_sweep | grep -iE ".bdf" | grep -oP '.*(?=\.)')

# echo ${mesh_filename_array[@]}

for mesh in ${mesh_filename_array[@]}
do
    python3 construct_mesh.py $mesh_dir $mesh
    # processed_mesh=$(ls ./mesh_size_sweep | grep -iE "mesh_processed_$mesh")
    # echo $processed_mesh
    # python3 main.py $mesh_dir $processed_mesh
done