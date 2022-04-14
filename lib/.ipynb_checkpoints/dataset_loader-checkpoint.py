import processing_2 as processing
import augmentation as aug
import numpy as np
import torch

class PytorchDS(torch.utils.data.Dataset):
    def __init__(self, ecgs, labels,  params=None):
        if params is not None:
            self.window = params['SR'] * params['WINDOW'] 
        else:
            self.window = 64*8
        self.ecgs = ecgs
        self.labels=labels
        self.params = params
        self.p_indexes=np.where(labels==1)[0]
        self.n_indexes=np.where(labels==0)[0]

    def __len__(self):
        return self.ecgs.shape[0]

    def __getitem__(self, index):
        index=self.__sample(index)
        result = dict()
        ecg = self.ecgs[index].copy()
        ecg = processing.fix_length(ecg, self.window, mode='random')
        result['ecg'] = self.augment(ecg)
        result['label']=self.labels[index]

        return result

    def __sample(self, index):
        if self.params is not None:
            if np.random.random()<self.params['PROPORTION_P']:
                index=np.random.choice(self.p_indexes)
            else:
                index=np.random.choice(self.n_indexes)
        return index
    
    def augment(self, ecg):
        if self.params is not None:
            ecg=aug.drop_random_channel(ecg, self.params['CHANNEL_DROP_P'])
            ecg=aug.add_random_noise(ecg, self.params['NOISE_TYPE'], self.params['NOISE_P'], self.params['NOISE_A'])
        return ecg
