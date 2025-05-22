#!/bin/bash -l

source /path/to/ccp4-<version>/bin/ccp4.setup-sh

for k in {0..19}
do
    export split=${k}
    mkdir 5_electron_density_ccp4_rand_${k}
    cat example_id_files/split_rand_${k}.txt | parallel --colsep ' ' -j 25 'echo -e "LABIN F1=FC PHI=PHIC \n GRID {2} {3} {4} \n RESOLUTION 40. {5} \n" | fft hklin 3_electron_density_mtz_res_${split}/{1}.gemmi.mtz mapout 5_electron_density_ccp4_rand_${split}/{1}_fft.ccp4'
done
