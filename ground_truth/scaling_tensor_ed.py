import sys
import torch

input_file = sys.argv[1]
max_value = sys.argv[2]
min_value = sys.argv[3]
split = sys.argv[4]
max_value = float(max_value)
min_value = float(min_value)
diff = max(max_value, abs(min_value))

with open(input_file) as myfile2: #select the first n_train examples as the training set, rest as validation set
    ids = myfile2.readlines()
idlist  = [x.rstrip() for x in ids]

for x in idlist:
    input_file = '7_electron_density_pt_rand_' + split + '/' + x + '_fft.pt'
    matrix = torch.load(input_file)
    new_matrix = torch.div(matrix, diff)
    torch.save(new_matrix, '../electron_density_scaled/' + x + '_fft.pt')
