#!/bin/bash -l

source /path/to/ccp4-<version>/bin/ccp4.setup-sh

for j in {0..19}
do
    export split=${j}
    mkdir 3_electron_density_mtz_res_${j}
    cat example_id_files/split_${j}.txt | parallel -j 25 'gemmi sfcalc --dmin 1.5 --to-mtz=3_electron_density_mtz_res_${split}/{}.gemmi.mtz 2_pdb/{}.pdb'
done
