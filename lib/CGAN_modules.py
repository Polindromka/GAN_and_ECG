import torch
from torch import nn
import numpy as np

_ACTIVATION_DICT = {'relu': nn.ReLU, 
                     'tanh': nn.Tanh,
                   'leaky_relu': nn.LeakyReLU}


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 act='relu', bn=True, dropout=None, 
                 maxpool=None, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
    
        if act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.act = _ACTIVATION_DICT[act]()
               
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        self.maxpool = None if maxpool is None else nn.MaxPool1d(maxpool)
        
    def forward(self, x):
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x)
            
        x = self.act(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        if self.maxpool is not None:
            x = self.maxpool(x)
            
        return x


class UnravelLinear(nn.Module):
    
    def __init__(self, in_channels, out_channels, out_temporal, ecg_size=512):
        super().__init__()
        self.out_channels = out_channels
        self.out_temporal = out_temporal 
        self.linear = nn.Linear(in_channels, out_channels * out_temporal)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.ecg_size=ecg_size
        
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = x.view(-1, self.out_channels, self.out_temporal)
        return x

class Discriminator(nn.Module):
    
    def __init__(self, window, n_leads=12, dropout=0.5, act='relu', bn=True, num_classes=2, channels = [8, 16, 32, 64], kernels = [7, 7, 5, 5, 3]):
        super().__init__()
        self.conv_blocks = []
        conv_block = Conv1dBlock(n_leads+n_leads,  channels[0], kernels[0], dropout=dropout, maxpool=4, act=act, bn=bn)
        self.conv_blocks.append(conv_block)
        num_blocks = len(channels)
        out_kernel = window // (2 ** (num_blocks+2))
        for i in range(num_blocks):
            if i != num_blocks-1:
                conv_block = Conv1dBlock(channels[i], channels[i+1], kernels[i], dropout=dropout, maxpool=2, act=act, bn=bn)
            else:
                conv_block = Conv1dBlock(channels[i], channels[i], kernels[i], dropout=dropout, maxpool=2, act=act, bn=bn)
            self.conv_blocks.append(conv_block)
        conv_final = nn.Conv1d(channels[num_blocks-1], 1, out_kernel)
        self.conv_blocks.append(conv_final)
        self.blocks = nn.ModuleList(self.conv_blocks)
        self.act_out = nn.Sigmoid()
        self.window=window
        self.embed =nn.Embedding(num_classes, n_leads*self.window)
        self.n_leads=n_leads
        
        
    def forward(self, x, labels):
        embedding=self.embed(labels.long().cuda()).view(labels.shape[0],self.n_leads, self.window)
        x = torch.cat([x,embedding], dim=1)
        for module in self.blocks:
            x = module(x)
        x = self.act_out(x)
        assert x.shape[1] == 1, f'Shape after linear: {x.shape}'
        assert x.shape[2] == 1, f'Shape after linear: {x.shape}'
        x = x[:, 0, 0]
        return x

    
class Generator(nn.Module):

    def __init__(self, window, noise_size, n_leads=12, act='leaky_relu', bn=True, dropout=None, num_classes=2, embed_size=100,
                 channels = [32, 32, 64, 64, 32, 32, 16, 16], kernels = [7, 7, 9, 9, 5, 5, 3]):
        super().__init__() 
        self.window=window
        channels = np.array(channels) * 2
        kernels = kernels
        paddings = np.array(kernels) // 2
        num_blocks = len(channels)
        self.conv_blocks = []
        self.unravel = UnravelLinear(noise_size+embed_size, channels[0], window + paddings.sum()*2)
        for i in range(num_blocks):
            if i != num_blocks-1:
                conv_block = Conv1dBlock(channels[i], channels[i+1], kernels[i], act=act, bn=bn, padding=0)
            else:
                conv_block = Conv1dBlock(channels[i], channels[i], kernels[i], act=act, bn=bn, padding=0)
            self.conv_blocks.append(conv_block)
        out_conv = nn.Conv1d(channels[num_blocks-1],  n_leads, 1)
        self.conv_blocks.append(out_conv)
        self.blocks = nn.ModuleList(self.conv_blocks)
        self.embed =nn.Embedding(num_classes, embed_size)
        
    def forward(self, x, labels):
        embedding=self.embed(labels.long().cuda())
        x=torch.cat([x,embedding], dim=1)
        x = self.unravel(x)
        for module in self.blocks:
            x = module(x)
        return x       