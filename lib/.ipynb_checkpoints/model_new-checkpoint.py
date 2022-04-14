import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
import pandas as pd
import _pickle as cPickle
import open_datasets
import random
import processing_2 as processing
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torchinfo import summary
from sklearn.metrics import roc_curve
import  CGAN_modules
from dataset_loader import PytorchDS
from torch.utils.data import DataLoader
import json
import os
import model_for_network

def run_dl(net, dl, netG = None, criterion=None, optimizer=None, params = None, rand =42):
    epoch_loss = 0.0
    trues = list()
    preds = list()
    random.seed(rand)
    for batch_num, batch in enumerate(dl):
        
        target = batch['label'].float()
        ecg = batch['ecg']
        if netG is not None:
            for i in range(ecg.shape[0]):
                if random.random()<=params['PROPORTION']:
                    noise = torch.randn(1, params['NOISE_DIM']).cuda()
                    labels = torch.from_numpy( np.array([target[i].long()]))
                    with torch.no_grad():
                        sample_ecg = netG(noise, labels).data.cpu()
                    ecg[i] = sample_ecg[0]
        logits = net(ecg.float().cuda()).float().cuda()
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        trues.append(target.detach().cpu().numpy())
        
        if criterion is not None:
            loss = criterion(logits.cuda(), target.cuda())
            epoch_loss += loss.item()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                         
    trues = np.concatenate(trues)
    preds = np.concatenate(preds)  
    if 'LOG_FILE_RES' in params:
        pd.DataFrame(preds, columns=['preds']).to_csv( params['LOG_FILE_RES'], index=False)
    epoch_loss /= len(dl)
    
    result = dict()
    result['loss'] = epoch_loss
    result['preds'] = preds
    result['trues'] = trues
    return result

def calc_metric(trues, preds):
    result = dict()
    result['ap'] = average_precision_score(trues, preds)
    result['auc'] = roc_auc_score(trues, preds)
    return result

def train_nn(params, net =None, train_dl=None, valid_dl=None, lr=0.03, epoches=10, rand = 42):
    try:
        if type(params) is str:
            with open(params) as json_file:
                params = json.load(json_file)
        params['LOG_FILE'] =f'LOGS/log_{params["PROPORTION"]}.csv'
        if not os.path.exists(f'CLASSIFIER/PROPORTION_{params["PROPORTION"]}'):
            os.makedirs(f'CLASSIFIER/PROPORTION_{params["PROPORTION"]}')
        params['CLASSIFIER_WEIGHTS'] = params['CLASSIFIER_WEIGHTS']+f'/PROPORTION_{params["PROPORTION"]}'
        if net is None:
            net = model_for_network.ECGClassifier(params)
        if train_dl is None or valid_dl is None:
            ecgs = np.load(params['DATASET_FILE'])
            labels = np.load(params['LABELS'])
            X_train, X_valid, y_train, y_valid = train_test_split(ecgs, labels, test_size=params['TEST_SIZE'], random_state=random.randint(15,115), stratify=labels)
            train_ds = PytorchDS(X_train, y_train,  params=params)
            valid_ds = PytorchDS(X_valid, y_valid, params=None)
            train_dl = DataLoader(train_ds, batch_size=params['BATCH_SIZE'], shuffle=True)
            valid_dl = DataLoader(valid_ds, batch_size=params['BATCH_SIZE'], shuffle=False)
        random.seed(rand)
        net.cuda()
        netG = None
        if params['PROPORTION']>0:
            with open(f'{params["GENERATOR"]}', 'rb') as fid:
                netG = cPickle.load(fid) 
        criterion = torch.nn.BCEWithLogitsLoss()
        if params['epoches'] is not None:
            epoches=params['epoches']
        if params['lr'] is not None:
            lr=params['lr']
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)   

        history = list()

        for epoch in tqdm(range(epoches)):
            log = dict()
            log['epoch'] = epoch
            net.train()
            result = run_dl(net, train_dl, netG, criterion=criterion, optimizer=optimizer, params = params, rand=rand)
            log['train_loss'] = result['loss']
            result = calc_metric(result['trues'], result['preds'])
            log.update({f'train_{key}': val for key, val in result.items()})


            net.eval()
            with torch.no_grad():
                result = run_dl(net, valid_dl, criterion=criterion, optimizer=None, params = params, rand = rand)
            log['valid_loss'] = result['loss']
            result = calc_metric(result['trues'], result['preds'])
            log.update({f'valid_{key}': val for key, val in result.items()})            
            history.append(log)
            pd.DataFrame(history).to_csv(params['LOG_FILE'], index=False)
            if epoch > 20:
                history_loss = pd.DataFrame(history)['valid_loss']
                if np.mean(history_loss[-10:]) >  np.mean(history_loss[-20:]):
                    print ('ПЕРЕОБУЧЕНИЕ')
                    return
            with open(f'{params["CLASSIFIER_WEIGHTS"]}/epoch_{epoch}.pkl', 'wb') as fid:
                cPickle.dump(net, fid) 
    except Exception as exp:
        print('Ошибка в модуле обучения классификатора')
        print(exp)
        return

def preload_ds(params):
    ecgs = list()
    diagnose=params['DIAGNOSE']
    df = open_datasets.get_dataframe(params['DATASET'], params['DATASET_PATH'], load_from_file=False)
    for ecg_file in tqdm(df['ecg_file'].values):
        ecg = processing.cached_load(ecg_file, params)
        ecg = ecg[:params['N_LEADS'], :]
        ecgs.append(processing.fix_length(ecg,  params['SR'] * params['WINDOW'], mode='random'))
    ecgs = np.stack(ecgs) 
    labels=np.zeros(ecgs.shape[0])*(-1)
    labels[df[diagnose]==1]=1
    indexes = np.where(labels!=-1)
    ecgs = ecgs[indexes]
    labels = labels[indexes]
    indices = np.arange(len(labels))
    X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(ecgs, labels, indices, test_size=0.2, stratify = labels)
    np.save(params['PATH']+params['DATASET']+'/'+params['DIAGNOSE']+'/'+params['PATH_TEST']+params['DATASET_FILE'], X_test)
    np.save(params['PATH']+params['DATASET']+'/'+params['DIAGNOSE']+'/'+params['PATH_TEST']+params['LABELS'], y_test)
    np.save(params['PATH']+params['DATASET']+'/'+params['DIAGNOSE']+'/'+params['DATASET_FILE'], X_train)
    np.save(params['PATH']+params['DATASET']+'/'+params['DIAGNOSE']+'/'+params['LABELS'], y_train)
    
def test_model(params, test_dl = None, rand = 42):
    try:
        if type(params) is str:
            with open(params) as json_file:
                params = json.load(json_file)
        if test_dl is None:
            ecgs_test = np.load(params['DATASET_FILE'])
            labels_test = np.load(params['LABELS'])
            test_ds = PytorchDS(ecgs_test, labels_test, params=None)
            test_dl = DataLoader(test_ds, batch_size=params['BATCH_SIZE'], shuffle=False)
        log = dict()
        netG=None
        if 'PROPORTION' in params:
            with open(f'{params["GENERATOR"]}', 'rb') as fid:
                netG = cPickle.load(fid)
        criterion = torch.nn.BCEWithLogitsLoss()
        with open(f'{params["CLASSIFIER"]}', 'rb') as fid:
            net = cPickle.load(fid) 
        net.cuda()
        net.eval()
        with torch.no_grad():
            result = run_dl(net, test_dl, netG, criterion=criterion, optimizer=None, params = params, rand=rand)
        log['valid_loss'] = result['loss']
        result = calc_metric(result['trues'], result['preds'])
        log.update({f'test_{key}': val for key, val in result.items()}) 
        return log
    except Exception as exp:
        print('Ошибка в модуле тестирования классификатора')
        print(exp)
        return