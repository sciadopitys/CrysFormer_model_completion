#!/bin/bash -l

mkdir ../ps_alphafold_randdrop
for j in {0..19}
do
    python3 scaling_tensor_ed.py example_id_files/split_${j}.txt 6.462 -0.922 ${j} &
    
done
