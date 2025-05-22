#!/bin/bash -l

mkdir ../patterson_scaled
for j in {0..19}
do
    python3 scaling_tensor_pat.py example_id_files/split_${j}.txt 2517.483 -249.376 ${j} &
done

