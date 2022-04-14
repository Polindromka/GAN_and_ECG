from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import _pickle as cPickle
import warnings
import numba
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split
import CGAN_datasets, CGAN_modules, open_datasets
import processing_2 as processing
from numba import cuda
from torchinfo import summary
import re 
import json
from memory_profiler import profile
    
def run_params(params):
    try:
        arr = torch.FloatTensor([[1,2,3],[4,5,6]])
        arr_2 = torch.FloatTensor([[1,2,3],[4,5,6]])*(-10)
        fid_calc = torch.jit.trace(calculate_fid, (arr.cuda(), arr.cuda()))

        if type(params) is str:
            with open(params) as json_file:
                params = json.load(json_file)

        window = params['WINDOW'] * params['SR']

        netG = CGAN_modules.Generator(window, params['NOISE_DIM'], n_leads = params['N_LEADS'], num_classes=params['NUM_CLASSES'], 
                                      embed_size=params['GEN_EMBEDDING'], channels = params['GEN_CHANNELS'], kernels = params['GEN_KERNELS']).cuda()
        netD = CGAN_modules.Discriminator(window, n_leads = params['N_LEADS'], num_classes=params['NUM_CLASSES'], dropout=params['DROPOUT'],
                                         channels = params['DIS_CHANNELS'], kernels = params['DIS_KERNELS']).cuda()

        netG.apply(weights_init)
        netD.apply(weights_init)
        batch_size = params['BATCH_SIZE']
        noise = torch.randn(batch_size, params['NOISE_DIM']).cuda()
        labels = torch.ones(batch_size).cuda()
        print(summary(netG,input_size=(noise.shape, labels.shape)))
        print(summary(netD,input_size=((64, 6, 512), labels.shape)))
        ds = CGAN_datasets.GANDataset(params)
        dl = ds.to_dl()

        optimG = torch.optim.Adam(netG.parameters(), lr=params['G_LR'])
        optimD = torch.optim.Adam(netD.parameters(), lr=params['D_LR'])

        criterion = torch.nn.BCELoss().cuda()

        history = list()
        flag = False

        for epoch in range(params['NUM_EPOCHS']):    
            row = run_epoch(dl, netD, netG, optimD, optimG, criterion, params, fid_calc)

            history.append(row)
            pd.DataFrame(history).to_csv(params['LOG_FILE'])
            if epoch % params['EPOCHS_TO_DISPLAY'] == 0:  
                display_gan(params, netG, ds, epoch)
                print(row)
                with open(f'{params["WEIGHTS"]}/epoch_{epoch}.pkl', 'wb') as fid:
                    cPickle.dump(netG, fid) 
    except Exception as exp:
        print('Ошибка в модуле обучения GAN')
        print(exp)
        return

def calculate_fid(act1, act2):
    mu1 = (torch.sum(act1, dim=0)/ act1.shape[0])
    sigma1 =  (torch.cov(torch.transpose(act1, 0, 1)))
    mu2 = (act2.sum(axis=0)/act2.shape[0])
    sigma2 = torch.cov(torch.transpose(act2, 0, 1))
    ssdiff = torch.sum((mu1 - mu2)**2.0)
    covmean = torch.sqrt(torch.mm(sigma1, sigma2)+0J).real
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def run_epoch(dl, netD, netG, optimD, optimG, criterion, params, fid_calc):
    
    d_loss = 0
    g_loss = 0
    
    trues = list()
    preds = list()
    fids = list()
    for batch_num, (true_batch, labels) in enumerate(tqdm(dl)):
        
        batch_size = true_batch.shape[0]
        true_labels = torch.ones(batch_size).cuda()
        fake_labels = torch.zeros(batch_size).cuda()

        true_batch = true_batch.cuda()
        # -------------------
        # Fake generation
        # -------------------        
        
        noise = torch.randn(batch_size, params['NOISE_DIM']).cuda()
        fake_batch = netG(noise, labels)       
        
        # -------------------
        # Train discriminator
        # -------------------
        true_logits = netD(true_batch, labels)
        fake_logits = netD(fake_batch.detach(), labels)
    
        
        true_loss = criterion(true_logits, true_labels)
        fake_loss = criterion(fake_logits, fake_labels)
        
        d_loss += (true_loss+fake_loss).detach().cpu().numpy()
        optimD.zero_grad()
        true_loss.backward()
        fake_loss.backward()
        optimD.step()
        
        trues.append(true_labels.cpu().numpy())
        trues.append(fake_labels.cpu().numpy())
        
        preds.append(true_logits.detach().cpu().numpy())
        preds.append(fake_logits.detach().cpu().numpy())
        
        #-------------------
        # Train Generator
        #-------------------     
        fake_logits = netD(fake_batch, labels)

        fake_loss = criterion(fake_logits, true_labels)
        g_loss += torch.mean(fake_loss).detach().cpu().numpy()
        optimG.zero_grad()
        fake_loss.backward()
        optimG.step()
        arr = np.random.randint(batch_size, size = 10)
        noise = torch.randn(len(arr), params['NOISE_DIM']).cuda()
        with torch.no_grad():
            sample_ecg = netG(noise, np.take(labels, arr)).data.cpu()
        y_pred = sample_ecg
        for i in range(len(arr)):
            fid = fid_calc(y_pred[i].cuda(), true_batch[arr[i]].cuda()).cpu().numpy()
            fids.append(fid)
    trues = np.concatenate(trues)
    preds = np.concatenate(preds)    
    
    row = dict()
    row['d_loss'] = d_loss / (batch_num + 1)
    row['g_loss'] = g_loss / (batch_num + 1)
    row.update(discr_metrics(trues, preds))
    return row
    
     
def display_gan(params, netG=None, ds=None, epoch=None):
    try:
        if type(params) is str:
            with open(params) as json_file:
                params = json.load(json_file)
        if netG is None:
             with open(f'{params["GENERATOR"]}', 'rb') as fid:
                netG = cPickle.load(fid)
        if epoch is None:
            epoch = int(re.findall('\d+', re.split('/', params["GENERATOR"])[-1])[0])
        noise = torch.randn(params['NUM_CLASSES'], params['NOISE_DIM']).cuda()
        labels = torch.from_numpy(np.arange(params['NUM_CLASSES'])).long().cuda()
        with torch.no_grad():
            sample_ecg = netG(noise, labels).data.cpu()
        if ds is not None:
            arr = np.load(params['LABELS']) 
            n = np.where(arr==0)[0][10]
            p = np.where(arr==1)[0][10]
            plt.figure(figsize=(120, 4*params['N_LEADS']))
            j = 0
            for i in range(params['N_LEADS']):
                plt.subplot(params['N_LEADS'], 4, j+1)
                plt.plot(ds[n][0][i])
                plt.grid()
                plt.title(f'Real_N_{i}')
                plt.subplot(params['N_LEADS'], 4, j+2)
                plt.plot(sample_ecg[0][i])
                plt.title(f'Fake_N_{i}')
                plt.grid()
                plt.subplot(params['N_LEADS'], 4, j+3)
                plt.plot(ds[p][0][i])
                plt.title(f'Real_P_{i}')
                plt.grid()
                plt.subplot(params['N_LEADS'], 4, j+4)
                plt.plot(sample_ecg[1][i])
                plt.title(f'Fake_P_{i}')
                plt.grid()
                j+=4
        else:
            plt.figure(figsize=(60, 4*params['N_LEADS']))
            j = 0
            for i in range(params['N_LEADS']):
                plt.subplot(params['N_LEADS'], 2, j+1)
                plt.plot(sample_ecg[0][i])
                plt.title(f'Fake_N_{i}')
                plt.grid()
                plt.subplot(params['N_LEADS'], 2, j+2)
                plt.plot(sample_ecg[1][i])
                plt.title(f'Fake_P_{i}')
                plt.grid()
                j+=2
        plt.savefig(f'{params["IMAGES_PATH"]}/epoch_{epoch}.png')
        plt.close()
    except Exception as exp:
        print('Ошибка в модуле визуализации сгенерированных данных')
        print(exp)
        return

@numba.njit 
def discr_metrics(trues, preds):
    result = dict()  
    trues = trues > 0.5
    preds = preds > 0.5
    result['accuracy'] = (trues == preds).mean()
    return result


def preload_ds(params):
    try:
        if type(params) is str:
            with open(params) as json_file:
                params = json.load(json_file)
        diagnose=params['DIAGNOSE']
        df = open_datasets.get_dataframe(params['DATASET'], params['DATASET_PATH'], load_from_file=False)
        ecgs = list()
        for ecg_file in tqdm(df['ecg_file'].values):
            ecg = processing.cached_load(ecg_file, params)
            ecg = ecg[:params['N_LEADS'], :]
            ecgs.append(ecg)
        ecgs = np.stack(ecgs) 
        labels=np.zeros(ecgs.shape[0])
        labels[df[diagnose]==1]=1
        indices = np.arange(len(labels))
        X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(ecgs, labels, indices, test_size=params['PROPORTION_TEST'], stratify = labels)
        np.save(params['DATASET_FILE_TEST'], X_test)
        np.save(params['LABELS_TEST'], y_test)
        np.save(params['DATASET_FILE'], X_train)
        np.save(params['LABELS'], y_train)
    except Exception as exp:
        print('Ошибка в модуле предобработки данных')
        print(exp)
        return

    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        