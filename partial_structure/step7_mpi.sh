#!/bin/bash -l

for j in {0..19}
do
    mkdir 7_electron_density_pt_${j}
    python3 convert_ccp4_to_pt_ed.py example_id_files/split_${j}.txt ${j} &
done
