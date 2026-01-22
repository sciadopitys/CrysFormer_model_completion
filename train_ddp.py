import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'

from torch.utils.data import random_split

import torch 
import torch.nn as nn
import torch.fft
import torch.cuda
import numpy as np
#from numpy import sqrt

import itertools
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import batch_gen
import datetime
from schedule_free.schedulefree.adamw_schedulefree import AdamWScheduleFree

from model import ViT_vary_encoder_decoder_partial_structure
import random

# lower matrix multiplication precision
torch.set_float32_matmul_precision('high')

#torch.set_num_threads(10)
    

def set_seed(args,rank):
    random.seed(args.seed*99+rank)
    np.random.seed(args.seed*99+rank)
    torch.manual_seed(args.seed*99+rank)

# setup operations for DDP processes
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(hours = 36))

def cleanup():
    dist.destroy_process_group()

# test set dataset definition
class Dataset(torch.utils.data.Dataset):

  def __init__(self, pdbIDs):
        self.ids = pdbIDs
        
  def __len__(self):
        return len(self.ids)

  def __getitem__(self, index): # each example consists of patterson map, partial structure template, and scale identifier inputs, and output electron density 
        
        ID1 = self.ids[index]
        
        id_split = ID1.split("_")
        ID = "_".join(id_split[:3])
        ID_full = "_".join(id_split[:5])

        # load Patterson map
        X = torch.load('patterson_scaled/' + ID + '_patterson.pt')
        X = torch.unsqueeze(X, 0)

        # load partial structure
        Xlist = torch.load('ps_alphafold_randdrop/' + ID_full + '_fft.pt') 
        Xlist = torch.unsqueeze(Xlist, 0)

        # load scale ID (part of full ID)
        scale = torch.tensor(int(id_split[-1]))    

        # introduce dummy dimension for consistency with training batches
        X = torch.unsqueeze(X, 0)
        Xlist = torch.unsqueeze(Xlist, 0)
        
        # load ground truth electron density
        y = torch.load('electron_density_scaled/' + ID + '_fft.pt')
        y = torch.unsqueeze(y, 0)
        
        return X, Xlist, scale, y

# training set dataset definition
class Dataset1(torch.utils.data.Dataset):

    def __init__(self, indices): 
        self.indices = indices
        
        
    def __getitem__(self, index):

        # load stored user-generated batches
        X = torch.load('batches/train_' + str(index) + '_patterson.pt')  
        PS = torch.load('batches/train_' + str(index) + '_ps.pt')
        S = torch.load('batches/train_' + str(index) + '_scale.pt')
        y = torch.load('batches/train_' + str(index) + '.pt')
        
        return X, PS, S, y
        
        
    def __len__(self):

        # ensure even division of batches across processes/gradient accumulation (currently hardcoded)
        return len(self.indices) - 3
        


def train(rank,args, test_datasets, n_test):
    device=rank
    setup(rank, args.world_size)
    set_seed(args,rank)

    # do not tune convolution to input size due to variable size
    torch.backends.cudnn.benchmark = False
    #print(torch.get_num_threads())

    # create training set with size based on 
    with open("training_indices.txt") as myfile2:
        indices = myfile2.readlines()
    indlist  = [x.rstrip() for x in indices]

    dataset_train = Dataset1(indlist)
    n_train = len(dataset_train) / args.world_size

    # obtain current process' split of test set
    dataset_test = test_datasets[rank]
    
    # calculate Pearson r correlation coefficient for single pair
    def pearson_r_loss(output, target): #calculate pearson r coefficient for single pair

        x = output
        y = target  
        
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx))) * torch.sqrt(torch.sum(torch.square(vy)))))
        return cost

    # calculate Pearson correlation of magnitudes after taking Fourier transform of prediction and ground truth for a batch
    def pearson_r_loss2(output, target):
        x = output[:,0,:,:,:]
        if target.dim() > 5:
            y = torch.squeeze(target, 0)[:,0,:,:,:]
        else:
            y = target[:,0,:,:,:]  
        
        batch = x.shape[0]
        cost = 0.0
        
        for i in range(batch):
        
            curx = x[i,:,:,:]
            cury = y[i,:,:,:]
            
            curx1 = torch.fft.fftn(curx)
            cury1 = torch.fft.fftn(cury)
            
            curx2 = torch.abs(curx1)
            cury2 = torch.abs(cury1)
            
            vx = curx2 - torch.mean(curx2)
            vy = cury2 - torch.mean(cury2)

            cost += (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx))) * torch.sqrt(torch.sum(torch.square(vy)))))
        return (cost / batch)

    # calculate Pearson correlation for a batch
    def pearson_r_loss3(output, target):
        x = output[:,0,:,:,:]
        if target.dim() > 5:
            y = torch.squeeze(target, 0)[:,0,:,:,:]
        else:
            y = target[:,0,:,:,:]  
        
        batch = x.shape[0]
        cost = 0.0
        
        for i in range(batch):
        
            curx = x[i,:,:,:]
            cury = y[i,:,:,:]
            
            vx = curx - torch.mean(curx)
            vy = cury - torch.mean(cury)

            cost += (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx))) * torch.sqrt(torch.sum(torch.square(vy)))))
        return (cost / batch)

    # calculate a Pearson correlation comparison between transformed prediction and corresponding Patterson input as a sanity check
    def fft_loss(patterson, electron_density):
        patterson = patterson[0,0,0,:,:]
        electron_density = electron_density[0,0,:,:,:]
        f1 = torch.fft.fftn(electron_density)
        f2 = torch.fft.fftn(torch.roll(torch.flip(electron_density, [0, 1, 2]), shifts=(1, 1, 1), dims=(0, 1, 2)))
        f3 = torch.mul(f1,f2)
        f4 = torch.fft.ifftn(f3)
        f4 = f4.real

        vx = f4 - torch.mean(f4)
        vy = patterson - torch.mean(patterson)

        cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx))) * torch.sqrt(torch.sum(torch.square(vy)))))
        return cost

    # Create sampler and dataloaders. Test set has batch size 1, training set effective batch size specified by generated batches
    sampler=DistributedSampler(dataset_train, shuffle = True, drop_last = True)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, shuffle = False, batch_size = 1, num_workers = 4, pin_memory = True, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, shuffle = False, batch_size = 1, num_workers = 4, pin_memory = True)

    # create model with specified hyperparameters and send it to current process' GPU
    model = ViT_vary_encoder_decoder_partial_structure(
        args=args,
        num_partial_structure = args.max_partial_structure, 
        image_height = args.max_image_height,          
        image_width = args.max_image_width,
        frames = args.max_image_depth,               
        image_patch_size = args.patch_size,     
        frame_patch_size = args.patch_size,  
        ps_size = args.ps_size,
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim,
        same_partial_structure_emb=args.same_partial_structure_emb,
        dropout = 0.1,
        emb_dropout = 0.1,
        biggan_block_num=args.biggan_block_num
    ).to(device)

    # wrap model in DDP for data paralleliztion
    model= DDP(model, device_ids=[rank])

    # specify main loss function term, learning rate schedule, number of epochs
    criterion = nn.MSELoss()
    learning_rate = 4.5e-4
    max_learning_rate = 2.85e-3
    n_epochs = args.total_epochs
    epoch = 0
    accum = 12 // args.world_size  #gradient accumulation

    # schedule-free AdamW optimizer with one-cycle learning rate schedule
    optimizer = AdamWScheduleFree(model.parameters(), lr = learning_rate, weight_decay=3e-2, warmup_steps = 0)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_learning_rate, steps_per_epoch=(len(train_loader) // accum), epochs=n_epochs, pct_start=0.05, three_phase= False, div_factor=(max_learning_rate/learning_rate), final_div_factor=0.525)

    # loading pretrained model    
    #checkpoint = torch.load('state_final.pth')
    #model.load_state_dict(checkpoint['model_state_dict'])

    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #loss = checkpoint['loss']
    #epoch = checkpoint['epoch']
    
    # step through lr schedule if desired
    #print(scheduler.state_dict())
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.3e-3, steps_per_epoch=(len(train_loader) // accum), epochs=n_epochs, pct_start=0.05, three_phase= False, div_factor=(13.0/learning_rate), final_div_factor=0.525)
    #for i in range(288300):
    #    scheduler.step()
    #print(scheduler.state_dict())
    
    # account for dummy dimension of desired ground truth
    def mse_wrapper_loss(output, target):
        y = torch.squeeze(target, 0)
        return criterion(output, y)

    clip = 1.0 #gradient clipping value
    while epoch < n_epochs:

        #set optimizer and model to "train" mode
        optimizer.train()
        model.train() 
        
        acc = 0.0 #for reporting current training set loss
        
        sampler.set_epoch(epoch)
        
        if epoch >= 0:
            for i, (x, ps, s, y) in enumerate(train_loader):
                
                # load tensors to GPU
                x, ps, s, y= x.to(device), ps.to(device), s.to(device), y.to(device)

                # apply model to current example
                yhat = model(x, ps, s)  

                # evaluate and aadd loss function terms
                loss_1 = mse_wrapper_loss(yhat, y)                       
                if loss_1.isnan().any():
                    raise Exception("nan")                
                loss_2 = (1 - pearson_r_loss3(yhat, y))
                loss = (0.99997 * loss_1) + (1.5e-5 * loss_2)
                acc += float(loss.item())

                #compute and accumulate gradients for model parameters
                loss = loss / accum  #needed due to gradient accumulation
                loss.backward()   

                #update model parameters only on accumulation epochs
                if (i+1) % accum == 0: 
                    #gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)                 #gradient clipping
                    
                    optimizer.step()

                    #clear (accumulated) gradients
                    model.zero_grad(set_to_none = True) 
                    
                    scheduler.step()
                    torch.cuda.empty_cache()

            # every 4 epochs, re-generate training batches to mix examples between batches
            if (epoch % 4 == 0):
                
                # have only one process generate batches while the others block
                dummy = torch.zeros(2, 3).to(device)
                if rank == 0:
                            
                    batch_gen.create_batches() 
                    for i in range(args.world_size - 1):
                        dist.send(dummy, i + 1)
                else:
                    dist.recv(dummy, 0)
        
        # evaluate test set metrics after each epoch
        if True:
            model.train()
            optimizer.eval()

            # necessary for schedule-free optimizer
            with torch.no_grad():
                for x, ps, s, y in itertools.islice(train_loader, 150):
                    x, ps, s = x.to(device), ps.to(device), s.to(device)
                    model(x, ps, s)
            model.eval()        
            
            acc_pearson = 0.0
            acc_pat = 0.0
            acc_fft_pearson = 0.0

            # report metrics across process' subset of test set
            with torch.no_grad(): 
                for x, ps, s, y in test_loader: 
                    x, ps, s, y = x.to(device), ps.to(device), s.to(device), y.to(device)
                    
                    yhat = model(x, ps, s)
                    loss_pearson = pearson_r_loss(yhat, y)
                    loss_pat = fft_loss(x, yhat)
                    loss_fft_pearson = pearson_r_loss2(yhat, y)
                    acc_pearson += float(loss_pearson.item())
                    acc_pat += float(loss_pat.item())
                    acc_fft_pearson += float(loss_fft_pearson.item())
                    torch.cuda.empty_cache()

            # pass all metrics to one process 
            metrics = torch.zeros(args.world_size - 1, 5).to(device)
            if args.world_size > 1:
                if rank == 0:
                    metrics[rank][0] = acc_pearson
                    metrics[rank][1] = acc_pat
                    metrics[rank][2] = acc_fft_pearson
                    dist.send(metrics, rank + 1) 
                elif rank < args.world_size - 1:
                    dist.recv(metrics, rank - 1)
                    metrics[rank][0] = acc_pearson
                    metrics[rank][1] = acc_pat
                    metrics[rank][2] = acc_fft_pearson
                    dist.send(metrics, rank + 1)                    
                else:
                    dist.recv(metrics, rank - 1) 
            
            # one process computes and reports average value of metrics across full test set while others block
            dummy = torch.zeros(2, 3).to(device)
            if rank == args.world_size - 1:
                for i in range(args.world_size - 1):
                    acc_pearson += metrics[i][0]
                    acc_pat += metrics[i][1]
                    acc_fft_pearson += metrics[i][2]
                curacc = (acc_pearson / n_test)
                curacc2 = (acc_pat / n_test)
                curacc3 = (acc_fft_pearson / n_test)

                # report epoch number, average training set loss, standard Pearson, comparison with original Patterson, Pearson after Fourier transforms, and last learning rate
                print("%d %.10f %.6f %.6f %.6f %.10f" % (epoch, (acc / n_train), curacc, curacc2, curacc3, scheduler.get_last_lr()[0]))  

                # save current model state
                if epoch == 0 or (epoch % args.save_every == 0):
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'epoch': epoch + 1,
                        }, 'state.pth')  
                        
                for i in range(args.world_size - 1):
                    dist.send(dummy, i)
            else:
                dist.recv(dummy, args.world_size - 1)
                

        epoch += 1

def run_train(args, testset, n_test):
    mp.spawn(train,
             args=(args, testset, n_test),
             nprocs=args.world_size,
             join=True)

if __name__ == "__main__":
    import argparse   

    # load list of test set examples
    with open("test.txt") as myfile: 
        testlist = myfile.readlines()
    testlist = [x.rstrip() for x in testlist]        

    # specify default values for model and training hyperparameters (can also be specified in command line)
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=71, type=int, help='Total epochs to train the model') # number of 
    parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot of the model state')
    parser.add_argument('--world_size', default=3, type=int, help='world size (number of training processes)')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--max_image_height',default=60, type=int, help='max size of Patterson/ground truth in first spatial dimension')
    parser.add_argument('--max_image_width',default=68, type=int, help='max size of Patterson/ground truth in second spatial dimension')
    parser.add_argument('--max_image_depth',default=44, type=int, help='max size of Patterson/ground truth in third spatial dimension')
    parser.add_argument('--ps_size',default=[60, 68, 44], type=list, help='maximum size of partial structures')
    parser.add_argument('--patch_size',default=4, type=int, help='patch size (all dimensions)')
    parser.add_argument('--activation',default='tanh', type=str, help='final activation function')

    parser.add_argument('--dim',default=512, type=int, help='token embedding dimension')
    parser.add_argument('--depth',default=12, type=int, help='transformer depth')
    parser.add_argument('--heads',default=12, type=int, help='number of attention heads')
    parser.add_argument('--mlp_dim',default=2048, type=int, help='dimensionality within feedforward MLP')

    parser.add_argument('--max_partial_structure',default=1, type=int, help='max number of partial structures')
    parser.add_argument('--same_partial_structure_emb', default = True, help='whether to use a constant partial structure embedding in each transformer layer')

    parser.add_argument('--biggan_block_num',default=2, type=int, help='number of post-transformer BigGAN residual convolutional blocks')
    parser.add_argument('--downsample',default=2, type=int, help='number of times to downsample within transformer')
    parser.add_argument('--downsample_by',default=4, type=int, help='reduction in each spatial dimension for downsamples')
    args = parser.parse_args()

    assert (args.depth % 2) == 0, "depth must be even"
    assert (args.depth % (2 * (args.downsample + 1))) == 0, "downsamples must evenly divide transformer depth"
    assert (args.downsample_by % args.patch_size) == 0, "must downsample by a multiple of patch size"

    # create full test set and split across processes
    dataset_test = Dataset(testlist, -1.0)
    n_test = float(len(dataset_test))
    ws = args.world_size
    n_split = int((n_test // ws) + 1)
    splits = [n_split for i in range(ws)]
    splits[ws - 1] = int(n_test) - ((ws - 1) * n_split)

    test_datasets = random_split(dataset_test, splits)  

    # generate training batches
    batch_gen.create_batches()

    run_train(args, test_datasets, n_test)



