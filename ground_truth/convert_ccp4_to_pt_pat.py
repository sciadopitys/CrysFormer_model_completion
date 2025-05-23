import gemmi
import numpy as np
import sys
import torch

list_file = sys.argv[1]
split = sys.argv[2]
with open(list_file) as myfile2: #select the first n_train examples as the training set, rest as validation set
    ids = myfile2.readlines()
idlist  = [x.rstrip() for x in ids]

for x in idlist:
    input_file = '4_patterson_ccp4_rand_' + split + '/' + x + '_patterson.ccp4'
    ccp4_map = gemmi.read_ccp4_map(input_file, setup = True)
    tensor = torch.Tensor(np.array(ccp4_map.grid, copy=False))
    torch.save(tensor, '6_patterson_pt_rand_' + split + '/' + x + '_patterson.pt')
