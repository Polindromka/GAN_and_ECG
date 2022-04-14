import torch
from torch import nn
import numpy as np


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 act='relu', bn=True, dropout=None, 
                 maxpool=None, padding=None, stride=1):
        
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=not bn, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
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
    
    
class Conv1dDoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 act='relu', 
                 bn=True, 
                 dropout=None, 
                 maxpool=None):
        
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, act=act, maxpool=maxpool, dropout=dropout, bn=bn)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    
    
class Conv1dDoubleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 act='relu', 
                 bn=True, 
                 dropout=None, 
                 maxpool=None):
        
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    
class Conv1dTripleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 act='relu', 
                 bn=True, 
                 dropout=None, 
                 maxpool=None):
        
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, act=act, maxpool=maxpool, dropout=dropout, bn=bn)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)
        self.conv3 = Conv1dBlock(out_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    
class Conv1dTripleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 act='relu', 
                 bn=True, 
                 dropout=None, 
                 maxpool=None):
        
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, act=act, maxpool=None, dropout=dropout, bn=False)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    
    
_ACTIVATION_DICT = {'relu': nn.ReLU, 
                     'tanh': nn.Tanh}

_BLOCK_DICT = {'single': Conv1dBlock, 
               'double': Conv1dDoubleBlock,
               'triple': Conv1dTripleBlock, 
               'double_res': Conv1dDoubleResBlock, 
               'triple_res': Conv1dTripleResBlock}



    
class ECGEncoder(nn.Module):
    def __init__(self, 
                 window=1024, 
                 block_type='single', 
                 out_act='tanh', 
                 dropout=None, 
                 bn=None, 
                 in_channels=12,
                 conv_channels=(24, 32, 48, 64, 96, 128), 
                 conv_kernels=(7, 5, 5, 3, 3, 3, 3),
                 linear_channels=(256, 256)):
        
        super().__init__()
        
            
        self.in_layer = Conv1dBlock(in_channels, conv_channels[0], conv_kernels[0], bn=bn, dropout=dropout, maxpool=2)
        
        conv_layers = list()
        for i in range(1, len(conv_channels)):
            conv_layers.append(_BLOCK_DICT[block_type](conv_channels[i-1], conv_channels[i], conv_kernels[i], bn=bn, dropout=dropout, maxpool=2))                   
        self.conv_layers = nn.ModuleList(conv_layers)
        
        
        out_kernel = window // (2 ** len(conv_channels))
        assert out_kernel >= 1, f'Encoder output kernel < 1. Input window: {window}, len of channels: {len(channels)}'     
        self.flatten_layer = Conv1dBlock(conv_channels[-1], linear_channels[0], out_kernel, bn=bn, dropout=dropout, maxpool=None, padding=0)
        
        linear_act = _ACTIVATION_DICT[out_act]
        linear_layers = list()
        for i in range(1, len(linear_channels)):
            linear_layers.append(nn.Linear(linear_channels[i-1], linear_channels[i]))
            linear_layers.append(linear_act())
        self.linear_layers = nn.ModuleList(linear_layers)

            
    def forward(self, x):
   
        x = self.in_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten_layer(x)
#         assert x.shape[2] == 1, x.shape.__repr__() + '\n\n' + self.__repr__()
        x = x[:, :, 0]
        
        for layer in self.linear_layers:
            x = layer(x)
        return x