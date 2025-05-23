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

torch.set_float32_matmul_precision('high')

#torch.set_num_threads(10)
    

def set_seed(args,rank):
    random.seed(args.seed*99+rank)
    np.random.seed(args.seed*99+rank)
    torch.manual_seed(args.seed*99+rank)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(hours = 36))

def cleanup():
    dist.destroy_process_group()


class Dataset(torch.utils.data.Dataset):

  def __init__(self, pdbIDs, randval):
        self.ids = pdbIDs
        self.randval = randval
        
  def __len__(self):
        return len(self.ids)

  def __getitem__(self, index): #each example consists of a patterson map, electron density pair
        
        
        ID1 = self.ids[index]
        
        id_split = ID1.split("_")
        ID = "_".join(id_split[:3])
        ID_full = "_".join(id_split[:5])
        
        X = torch.load('patterson_scaled/' + ID + '_patterson.pt')
        X = torch.unsqueeze(X, 0)

        Xlist = torch.load('ps_alphafold_randdrop/' + ID_full + '_fft.pt') 
        Xlist = torch.unsqueeze(Xlist, 0)
        
        scale = torch.tensor(int(id_split[-1]))    
            
        X = torch.unsqueeze(X, 0)
        Xlist = torch.unsqueeze(Xlist, 0)
        

        y = torch.load('electron_density_scaled/' + ID + '_fft.pt')
        y = torch.unsqueeze(y, 0)
        
        return X, Xlist, scale, y


class Dataset1(torch.utils.data.Dataset):

    def __init__(self, indices): 
        self.indices = indices
        
        
    def __getitem__(self, index):

        X = torch.load('batches/train_' + str(index) + '_patterson.pt')  
        PS = torch.load('batches/train_' + str(index) + '_ps.pt')
        S = torch.load('batches/train_' + str(index) + '_scale.pt')
        y = torch.load('batches/train_' + str(index) + '.pt')
        
        return X, PS, S, y
        
        
    def __len__(self):
        return len(self.indices) - 3
        


def train(rank,args, test_datasets, n_test):
    device=rank
    setup(rank, args.world_size)
    set_seed(args,rank)
    torch.backends.cudnn.benchmark = False
    #print(torch.get_num_threads())

    with open("training_indices.txt") as myfile2:
        indices = myfile2.readlines()
    indlist  = [x.rstrip() for x in indices]
        
    dataset_val = test_datasets[rank]
    
    dataset_train = Dataset1(indlist)
    n_train = len(dataset_train) / args.world_size

    def pearson_r_loss(output, target): #calculate pearson r coefficient for single pair

        x = output
        y = target  
        
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx))) * torch.sqrt(torch.sum(torch.square(vy)))))
        return cost
        
    def pearson_r_loss2(output, target): #calculate pearson r coefficient of magnitudes after taking Fourier transform of prediction and ground truth
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
        
    def pearson_r_loss3(output, target): #calculate pearson r coefficient for batch
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

    sampler=DistributedSampler(dataset_train, shuffle = True, drop_last = True)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, shuffle = False, batch_size= args.batch_size, num_workers = 4, pin_memory = True, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_val, shuffle = False, batch_size= args.test_batch_size, num_workers = 4, pin_memory = True)

    model = ViT_vary_encoder_decoder_partial_structure(
        args=args,
        num_partial_structure = args.max_partial_structure, #max number of amino acid (partial structure) 
        image_height = args.max_image_height,          # max image size
        image_width = args.max_image_width,
        frames = args.max_frame_size,               # max number of frames
        image_patch_size = args.patch_size,     # image patch size
        frame_patch_size = args.patch_size,      # frame patch size
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
        


    model= DDP(model, device_ids=[rank])

    #specify loss function, learning rate schedule, number of epochs
    criterion = nn.MSELoss()
    learning_rate = 4.5e-4
    max_learning_rate = 2.85e-3
    
    n_epochs = args.total_epochs
    epoch = 0
    accum = 12 // args.world_size  #gradient accumulation
    
    optimizer = AdamWScheduleFree(model.parameters(), lr = learning_rate, weight_decay=3e-2, warmup_steps = 0)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_learning_rate, steps_per_epoch=(len(train_loader) // accum), epochs=n_epochs, pct_start=0.05, three_phase= False, div_factor=(max_learning_rate/learning_rate), final_div_factor=0.525)

    #loading pretrained model    
    #checkpoint = torch.load('state_15_p21_alphafold_randdrop_smallercell_recycle_5.pth')
    #model.load_state_dict(checkpoint['model_state_dict'])

    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #loss = checkpoint['loss']
    #epoch = checkpoint['epoch']
    
    
    #print(scheduler.state_dict())
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.3e-3, steps_per_epoch=(len(train_loader) // accum), epochs=n_epochs, pct_start=0.05, three_phase= False, div_factor=(13.0/learning_rate), final_div_factor=0.525)
    #for i in range(288300):
    #    scheduler.step()
    #print(scheduler.state_dict())
    

    def mse_wrapper_loss(output, target):

        y = torch.squeeze(target, 0)
        return criterion(output, y)

    clip = 1.0 #gradient clipping value
    count = 0
    while epoch < n_epochs:

        optimizer.train()
        model.train() 
        acc = 0.0
        sampler.set_epoch(epoch)
        if epoch >= 0:
            for i, (x, ps, s, y) in enumerate(train_loader):
                x, ps, s, y= x.to(device), ps.to(device), s.to(device), y.to(device)
                
                yhat = model(x, ps, s)                                              #apply model to current example
                loss_1 = mse_wrapper_loss(yhat, y)                  #evaluate loss      
                if loss_1.isnan().any():
                    raise Exception("nan")                
                loss_2 = (1 - pearson_r_loss3(yhat, y))

                loss = (0.99997 * loss_1) + (1.5e-5 * loss_2)
                acc += float(loss.item())
                loss = loss / accum                                             #needed due to gradient accumulation
                loss.backward()                                                 #compute and accumulate gradients for model parameters
                
                if (i+1) % accum == 0:                                          
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)                 #gradient clipping
                    
                    optimizer.step()                                            #update model parameters only on accumulation epochs
                    model.zero_grad(set_to_none = True)                         #clear (accumulated) gradients
                    scheduler.step()
                    torch.cuda.empty_cache()
            
            if (epoch % 4 == 0):
                dummy = torch.zeros(2, 3).to(device)
                if rank == 0:
                            
                    batch_gen.create_batches() 
                    for i in range(args.world_size - 1):
                        dist.send(dummy, i + 1)
                else:
                    dist.recv(dummy, 0)
        
        
        if True:
            model.train()
            optimizer.eval()
            with torch.no_grad():
                for x, ps, s, y in itertools.islice(train_loader, 150):
                    x, ps, s = x.to(device), ps.to(device), s.to(device)
                    model(x, ps, s)
            model.eval()        
            acc_pearson = 0.0
            acc_pat = 0.0
            acc_fft_pearson = 0.0
            with torch.no_grad(): #calculate metrics for all test set elements
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
            
            
            dummy = torch.zeros(2, 3).to(device)
            #store average value of metrics
            if rank == args.world_size - 1:
                for i in range(args.world_size - 1):
                    acc_pearson += metrics[i][0]
                    acc_pat += metrics[i][1]
                    acc_fft_pearson += metrics[i][2]
                curacc = (acc_pearson / n_test)
                curacc2 = (acc_pat / n_test)
                curacc3 = (acc_fft_pearson / n_test)
                print("%d %.10f %.6f %.6f %.6f %.10f" % (epoch, (acc / n_train), curacc, curacc2, curacc3, scheduler.get_last_lr()[0]))  
             
                if epoch >= 0:
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
        count += 1

def run_train(args, testset, n_test):
    mp.spawn(train,
             args=(args, testset, n_test),
             nprocs=args.world_size,
             join=True)

if __name__ == "__main__":
    import argparse   
        
    with open("test.txt") as myfile: #select the first n_train examples as the training set, rest as validation set
        testlist = myfile.readlines()
    testlist = [x.rstrip() for x in testlist]        
        
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=71, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=5, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--test_batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--world_size', default=3, type=int, help='world size')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--lr_lambda', default=2, type=int, help='lr scheduler')
    parser.add_argument('--max_frame_size',default=60, type=int, help='max size')
    parser.add_argument('--max_image_height',default=68, type=int, help='max size')
    parser.add_argument('--max_image_width',default=44, type=int, help='max size')
    parser.add_argument('--ps_size',default=[60, 68, 44], type=list, help='max size')
    parser.add_argument('--patch_size',default=4, type=int, help='patch size')
    parser.add_argument('--activation',default='tanh', type=str, help='activation function')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--FFT', default = False, help='FFT')
    parser.add_argument('--iFFT', default = False, help='FFT')
    parser.add_argument('--FFT_skip', default = False, help='FFT')
    parser.add_argument('--transformer', default='Nystromformer',type=str , help='transformer type: normal or Nystromformer')
    parser.add_argument('--work_dir', default='', type=str,
                    help='experiment directory.')
    parser.add_argument('--save_pth', default = True, help='save_pth')

    parser.add_argument('--dim',default=512, type=int, help='dim')
    parser.add_argument('--depth',default=12, type=int, help='depth')
    parser.add_argument('--heads',default=12, type=int, help='heads')
    parser.add_argument('--mlp_dim',default=2048, type=int, help='mlp_dim')

    parser.add_argument('--max_partial_structure',default=1, type=int, help='max number of partial_structure')
    parser.add_argument('--same_partial_structure_emb', default = True, help='whether use same partial structure embeding layer each transformer layer')

    parser.add_argument('--biggan_block_num',default=2, type=int, help='number of additional biggan block')
    parser.add_argument('--downsample',default=2, type=int, help='number of additional biggan block')
    parser.add_argument('--downsample_by',default=4, type=int, help='number of additional biggan block')
    args = parser.parse_args()

    assert (args.depth % 2) == 0, "depth must be even"
    assert (args.depth % (2 * (args.downsample + 1))) == 0, "downsamples must evenly divide depth"
    assert (args.downsample_by % args.patch_size) == 0, "must downsample by a multiple of patch size"


    #args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    #logging = create_exp_dir(args.work_dir,scripts_to_save=None, debug=args.debug)
    
    dataset_val = Dataset(testlist, -1.0)
    n_test = float(len(dataset_val))
    ws = args.world_size
    n_split = int((n_test // ws) + 1)
    splits = [n_split for i in range(ws)]
    splits[ws - 1] = int(n_test) - ((ws - 1) * n_split)

    test_datasets = random_split(dataset_val, splits)  
    
    batch_gen.create_batches()

    run_train(args, test_datasets, n_test)
