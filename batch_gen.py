
import torch 
import torch.fft
import torch.cuda

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import random
def shuffle_slice(a, start, stop):
    i = start
    while (i < (stop-1)):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1


def create_batches():
    
    with open("train.txt") as myfile1: 
        trainlist = myfile1.readlines()
    trainlist  = [x.rstrip() for x in trainlist]

    with open("size_indices.txt") as myfile: 
        sindices = myfile.readlines()
    sindices  = [x.rstrip() for x in sindices]
    
    for i in range(len(sindices) - 1):
        start = int(sindices[i])
        end = int(sindices[i+1])
        shuffle_slice(trainlist, start, end)

    with open("training_indices.txt") as myfile2: 
        indices = myfile2.readlines()
    indices  = [x.rstrip() for x in indices]

    for i in range(len(indices) - 1):
        start = int(indices[i])
        end = int(indices[i+1])
        xlist = []
        pslist = []
        ylist = []
        start_id1 = trainlist[start]
        start_id_split = start_id1.split("_")
        new_scale = torch.tensor(int(start_id_split[-1])) 
        for j in range(start, end):
            id1 = trainlist[j]
            id_split = id1.split("_")
            id = "_".join(id_split[:3])
            id_full = "_".join(id_split[:5])
            

            new_x = torch.load('patterson_scaled/' + id + '_patterson.pt')
            new_x = torch.unsqueeze(new_x, 0)

            new_xlist = torch.load('ps_alphafold_randdrop/' + id_full + '_fft.pt')  
            new_xlist = torch.unsqueeze(new_xlist, 0)
            
            xlist.append(new_x)
            pslist.append(new_xlist)
            

            new_y = torch.load('electron_density_scaled/' + id + '_fft.pt')
            new_y = torch.unsqueeze(new_y, 0)
            ylist.append(new_y)

        data_x = torch.stack(xlist)
        data_ps = torch.stack(pslist)
        data_y = torch.stack(ylist)
        torch.save(data_x, 'batches/train_' + str(i) + '_patterson.pt')  
        torch.save(data_ps, 'batches/train_' + str(i) + '_ps.pt')
        torch.save(new_scale, 'batches/train_' + str(i) + '_scale.pt')
        torch.save(data_y, 'batches/train_' + str(i) + '.pt')
        