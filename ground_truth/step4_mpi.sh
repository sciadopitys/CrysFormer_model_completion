#!/bin/bash -l

source /path/to/ccp4-<version>/bin/ccp4.setup-sh

for j in {0..19}
do
    export split=${j}
    mkdir 4_patterson_ccp4_rand_${j}
    cat example_id_files/split_rand_${j}.txt | parallel --colsep ' ' -j 25 'echo -e "LABIN F1=FC PHI=PHIC \n GRID {2} {3} {4} \n RESOLUTION 40. {5} \n PATT \n" | fft hklin 3_electron_density_mtz_res_${split}/{1}.gemmi.mtz mapout 4_patterson_ccp4_rand_${split}/{1}_patterson.ccp4'
done
