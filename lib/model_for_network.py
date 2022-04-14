from torch import nn
import torch
import network_2  
        
class ECGClassifier(nn.Module):
    def __init__(self, params):       
        super().__init__()
        
        window = params['WINDOW'] * params['SR']
        block_type = params['NET_BLOCK']
        out_act = params['ENCODER_ACT']
        dropout = params['DROPOUT']
        bn = params['BATCH_NORM']
        in_channels = params['N_LEADS']
        conv_kernels = params['CONV_KERNELS']
        conv_channels = params['CONV_CHANNELS']
        linear_channels = params['LINEAR_CHANNELS'] 
        num_classes = 1

        self.encoder = network_2.ECGEncoder(window, block_type=block_type, out_act=out_act, dropout=dropout, bn=bn, in_channels=in_channels, conv_kernels=conv_kernels, linear_channels=linear_channels)
        self.linear = nn.Linear(linear_channels[-1], num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        sigmoid = nn.Sigmoid()
        x = x[:,0]
        return x