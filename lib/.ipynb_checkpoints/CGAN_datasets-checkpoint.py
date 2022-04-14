import torch
import numpy as np
import processing_2 as processing

class GANDataset(torch.utils.data.Dataset):
    
    def __init__(self, params, test = False):
        self.params= params
        self.window = params['SR'] * params['WINDOW'] 
        self.ecgs = np.load(params['DATASET_FILE'])
        self.labels = np.load(params['LABELS'])
        if test:
            self.ecgs = np.load(params['DATASET_FILE_TEST'])
            self.labels = np.load(params['LABELS_TEST'])
        self.p_indexes=np.where(self.labels==1)[0]
        self.n_indexes=np.where(self.labels==0)[0]
    def __len__(self):
        return self.ecgs.shape[0]

    
    def __getitem__(self, index):
        index = self.__sample()
        ecg = self.ecgs[index].copy()
        ecg = processing.fix_length(ecg, self.window, mode='random')
        labels = self.labels[index]
        return ecg.astype('float32'), labels
    
    def __sample(self):
        if np.random.random()<self.params['PROPORTION_P']:
            index=np.random.choice(self.p_indexes)
        else:
            index=np.random.choice(self.n_indexes)
        return index
    
    def to_dl(self):
        bs = self.params['BATCH_SIZE']
        nw = self.params['NUM_WORKERS']
        dl = torch.utils.data.DataLoader(self, batch_size=bs, num_workers=nw, shuffle=True)
        return dl