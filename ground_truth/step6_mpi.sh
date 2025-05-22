#!/bin/bash -l

for j in {0..19}
do
    mkdir 6_patterson_pt_rand_${j}
    python3 convert_ccp4_to_pt_pat.py example_id_files/split_${j}.txt ${j} &
done
