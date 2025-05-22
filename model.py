import torch
from torch import nn

import math
from einops import rearrange
from scale_equivariant_3d_convnet.layers import conv
from scale_equivariant_3d_convnet.layers import normalization
from einops.layers.torch import Rearrange
import nystrom_ps

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PreNorm_partial_structure(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, partial_structure, **kwargs):
        return self.fn(self.norm(x), self.norm_p(partial_structure), **kwargs)

class Transformer_partial_structure(nn.Module):
    def __init__(self, patch_dim, num_patches, num_patches_ps, patch_height, patch_width, frame_patch_size, num_partial_structure, dim, depth, heads, dim_head, mlp_dim, same_partial_structure_emb=True,dropout = 0., downsample = 0, downsample_by = 0, channels = 10):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches_ps, dim))
        self.layers = nn.ModuleList([])
        self.downsample_by = downsample_by
        self.downupsample = nn.ModuleList([])
        self.channels = channels
        self.depth = depth
        self.change_dim = depth // ((downsample + 1) * 2)
        self.same_partial_structure_emb=same_partial_structure_emb
        if self.same_partial_structure_emb:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm_partial_structure(dim, nystrom_ps.Nystrom_attention_partial_structure(dim, heads = heads, dim_head = dim_head, num_landmarks = 64, residual = True, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            self.partial_structure_to_patch_embedding = nn.Sequential(
                Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (c p1 p2 pf)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm_partial_structure(dim, nystrom_ps.Nystrom_attention_partial_structure(dim, heads = heads, dim_head = dim_head, num_landmarks = 64, residual = True, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            self.partial_structure_rearrange=Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (c p1 p2 pf)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size)
            self.partial_structure_emb_layers=nn.ModuleList([])
            for _ in range(depth):
                self.partial_structure_emb_layers.append(nn.Sequential(
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, dim),
                    nn.LayerNorm(dim),
                ))
        
        ksize = downsample_by + 1

        for _ in range(downsample):
            self.downupsample.append(nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=ksize, bias=True))
        for _ in range(downsample):
            self.downupsample.append(nn.ConvTranspose3d(in_channels=channels, out_channels=channels, kernel_size=ksize, bias=True))
    def forward(self, x, partial_structure, fsize, hsize, wsize, pfsize, phsize, pwsize):
        if self.same_partial_structure_emb:
            
            partial_structure = self.partial_structure_to_patch_embedding(partial_structure)
            _,total_partial_patches,_ = partial_structure.shape
            partial_structure = partial_structure + self.pos_embedding[:,:total_partial_patches,:]
            count1 = 1
            count2 = 0
            
            for attn, ff in self.layers:
                
                x = attn(x,partial_structure) + x    
                x = ff(x) + x
                
                if ((count1 % self.change_dim) == 0) and (count1 != (self.depth // 2)) and (count1 != self.depth):
                    x = rearrange(x, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=fsize // pfsize, h=hsize // phsize, w=wsize // pwsize, p1 = phsize, p2 = pwsize, pf = pfsize, c=self.channels)
                    x = self.downupsample[count2](x)
                    if count1 < (self.depth // 2):
                        hsize -= self.downsample_by
                        wsize -= self.downsample_by
                        fsize -= self.downsample_by
                    else:
                        hsize += self.downsample_by
                        wsize += self.downsample_by
                        fsize += self.downsample_by
                    x = rearrange(x, 'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = phsize, p2 = pwsize, pf = pfsize)    
                    count2 += 1
                count1 += 1
            return x
        else:
            partial_structure = self.partial_structure_rearrange(partial_structure)
            _,total_partial_patches,_ = partial_structure.shape

            count1 = 1
            count2 = 0
            
            for i in range(len(self.layers)):
                partial_structure_i=self.partial_structure_emb_layers[i](partial_structure)
                partial_structure_i = partial_structure_i + self.pos_embedding[:,:total_partial_patches,:]
                x = self.layers[i][0](x,partial_structure_i) + x
                x = self.layers[i][1](x) + x
                
                
                if ((count1 % self.change_dim) == 0) and (count1 != (self.depth // 2)) and (count1 != self.depth):
                    x = rearrange(x, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=fsize // pfsize, h=hsize // phsize, w=wsize // pwsize, p1 = phsize, p2 = pwsize, pf = pfsize, c=self.channels)
                    x = self.downupsample[count2](x)
                    if count1 < (self.depth // 2):
                        hsize -= self.downsample_by
                        wsize -= self.downsample_by
                        fsize -= self.downsample_by
                    else:
                        hsize += self.downsample_by
                        wsize += self.downsample_by
                        fsize += self.downsample_by
                    x = rearrange(x, 'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = phsize, p2 = pwsize, pf = pfsize)    
                    count2 += 1
                count1 += 1
                
            return x

class BigGANBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels

        self.relu = nn.ReLU(inplace=True)

        self.bn1=nn.BatchNorm3d(in_channels)
        self.conv1=nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

        self.bn2=nn.BatchNorm3d(in_channels)

        self.conv2=nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        
        self.bn3=nn.BatchNorm3d(out_channels)

        self.conv3=nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        
        self.bn4=nn.BatchNorm3d(out_channels)

        self.conv4=nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        
        res=x

        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.conv4(x)

        x = x + res
        return x

class ViT_vary_encoder_decoder_partial_structure(nn.Module):
    def __init__(self, args, num_partial_structure, image_height, image_width, image_patch_size, frames, frame_patch_size, ps_size, dim, depth, heads, mlp_dim, same_partial_structure_emb, channels = 10, dim_head = 64, dropout = 0., emb_dropout = 0., biggan_block_num=2):
        super().__init__()


        self.FFT=args.FFT
        self.iFFT=args.iFFT
        self.activation=args.activation
        self.FFT_skip=args.FFT_skip
        same_partial_structure_emb=args.same_partial_structure_emb

        self.conv1 = conv.SESConv_Z3_H(in_channels = 1, out_channels = 10, kernel_size = 7, effective_size = 4, scales = [0.759313,1.0,0.887459,0.987854,0.803794,0.827597,0.923694,0.918319,0.797655,0.833723,0.829428,0.801383,0.854762,0.940347,0.968662,0.892938,0.780852,0.979999,0.816765,0.751273], padding = 3, padding_mode='circular', bias = False)
        self.bn1 = normalization.SEBatchNorm(10)
        channels=10

        self.conv1_p = conv.SESConv_Z3_H(in_channels = 1, out_channels = 10, kernel_size = 7, effective_size = 4, scales = [0.759313,1.0,0.887459,0.987854,0.803794,0.827597,0.923694,0.918319,0.797655,0.833723,0.829428,0.801383,0.854762,0.940347,0.968662,0.892938,0.780852,0.979999,0.816765,0.751273], padding = 3, padding_mode='circular', bias = False)
        self.bn1_p = normalization.SEBatchNorm(10)

        patch_height, patch_width = pair(image_patch_size)

        self.patch_height=patch_height
        self.patch_width=patch_width
        self.frame_patch_size=frame_patch_size

        self.num_patches = (math.ceil(image_height / patch_height)) * math.ceil(image_width / patch_width) * math.ceil(frames / frame_patch_size)
        self.num_patches_ps = (math.ceil(ps_size[0] / patch_height)) * (math.ceil(ps_size[1] / patch_width)) * (math.ceil(ps_size[2] / frame_patch_size)) * num_partial_structure
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        downsample = args.downsample
        downsample_by = args.downsample_by

        assert dim % (patch_height * patch_width * frame_patch_size) == 0
        channels_emb = dim // (patch_height * patch_width * frame_patch_size)

        self.transformer = Transformer_partial_structure(patch_dim, self.num_patches, self.num_patches_ps, patch_height, patch_width, frame_patch_size, num_partial_structure, dim, depth, heads, dim_head, mlp_dim, same_partial_structure_emb, dropout, downsample, downsample_by, channels_emb)

        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )
        
        self.biggan_block_num=biggan_block_num
        self.bigGAN_layers=nn.ModuleList([])
        for i in range(biggan_block_num):
            self.bigGAN_layers.append(BigGANBlock(10,10))

        self.conv2 = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x, ps, scale):
        x = torch.squeeze(x,0)
        partial_structure = torch.squeeze(ps, 0)
        pad_list=[]
        res=x.shape[4]% self.patch_width
        if not(res==0):
            pad_list.append((self.patch_width-res)//2)
            pad_list.append((self.patch_width-res)-(self.patch_width-res)//2)
        else:
            pad_list.append(0)
            pad_list.append(0)


        res=x.shape[3]% self.patch_height
        if not(res==0):
            pad_list.append((self.patch_height-res)//2)
            pad_list.append((self.patch_height-res)-(self.patch_height-res)//2)
        else:
            pad_list.append(0)
            pad_list.append(0)


        res=x.shape[2]% self.frame_patch_size
        if not(res==0):
            pad_list.append((self.frame_patch_size-res)//2)
            pad_list.append((self.frame_patch_size-res)-(self.frame_patch_size-res)//2)
        else:
            pad_list.append(0)
            pad_list.append(0)

        x= torch.nn.functional.pad(x,tuple(pad_list),"constant", 0)

        partial_structure = torch.nn.functional.pad(partial_structure,tuple(pad_list),"constant", 0)

        x_shape = x.shape
        if self.FFT:
            x_size=x.size()
            if self.FFT_skip:
                x= torch.fft.fftn(x,dim=(2,3,4)).real + x
            else:
                x= torch.fft.fftn(x,dim=(2,3,4)).real
            assert x.size()==x_size

        x = self.conv1(x)
        x = self.bn1(x) 
        x = x[:,:,scale, :,:,:]
        x = torch.squeeze(x, dim=2)
        
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        _ , num_partial_structure, _ ,_ ,_ = partial_structure.shape
        partial_structure = self.conv1_p(partial_structure)
        partial_structure = self.bn1_p(partial_structure)
        partial_structure = partial_structure[:,:,scale, :,:,:]
        partial_structure = torch.squeeze(partial_structure, dim=2)
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x,partial_structure, x_shape[2], x_shape[3], x_shape[4], self.frame_patch_size, self.patch_height, self.patch_width)

        x = self.from_patch_embedding(x)

        x = rearrange(x, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=x_shape[2] // self.frame_patch_size, h=x_shape[3] // self.patch_height, w=x_shape[4] // self.patch_width, p1 = self.patch_height, p2 = self.patch_width, pf = self.frame_patch_size, c=10)
        
        for i in range(self.biggan_block_num):
            x = self.bigGAN_layers[i](x)

        x = self.conv2(x)

        if self.iFFT:
            x = torch.fft.ifftn(x).real

        if self.activation=='tanh':
            x = torch.tanh(x)
        elif self.activation=='sigmoid':
            x = torch.sigmoid(x)
        elif self.activation=='None':
            x = x
        elif self.activation=='relu':
            x = torch.torch.nn.functional.relu(x)

        x= x[:,:,pad_list[4]:(x.shape[2]-pad_list[5]),pad_list[2]:(x.shape[3]-pad_list[3]),pad_list[0]:(x.shape[4]-pad_list[1])]
        
        return(x)

