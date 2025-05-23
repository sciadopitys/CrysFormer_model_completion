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
    input_file = '6_patterson_pt_rand_' + split + '/' + x + '_patterson.pt'
    matrix = torch.load(input_file)
    new_matrix = torch.div(matrix, diff)
    torch.save(new_matrix, '../patterson_scaled/' + x + '_patterson.pt')
