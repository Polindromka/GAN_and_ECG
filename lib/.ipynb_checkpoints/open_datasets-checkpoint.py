import os
import urllib
import tarfile
from tqdm import tqdm
import pandas as pd
import numpy as np
import wfdb

from utils import TqdmUpTo, download_dataset, extract_dataset

import functools

NAME_TO_URL = dict()
NAME_TO_URL['WFDB_CPSC2018'] = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz/'
NAME_TO_URL['WFDB_CPSC2018_2'] = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018_2.tar.gz/'
NAME_TO_URL['WFDB_Ga'] = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz/'
NAME_TO_URL['WFDB_PTB'] = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTB.tar.gz/'
NAME_TO_URL['WFDB_PTBXL'] = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTBXL.tar.gz/'
# NAME_TO_URL['WFDB_ShaoxingUniv'] = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_ShaoxingUniv.tar.gz/'
NAME_TO_URL['WFDB_StPetersburg'] = 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_StPetersburg.tar.gz/'

NAME_TO_URL['WFDB_ChapmanShaoxing'] = 'https://storage.cloud.google.com/physionetchallenge2021-public-datasets/WFDB_ChapmanShaoxing.tar.gz'
NAME_TO_URL['WFDB_Ningbo'] = 'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_Ningbo.tar.gz'

NAMES = tuple(sorted(list(NAME_TO_URL.keys())))

LEADS = dict()
LEADS[12] = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
LEADS[6] = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
LEADS[4] = ('I', 'II', 'III', 'V2')
LEADS[3] = ('I', 'II', 'V2')
LEADS[2] = ('I', 'II')


def download_end_extract_dataset(name, path, remove_archieve=True):
    filename =  os.path.join(path, name + '.tar.gz')
    
    download_dataset(NAME_TO_URL[name], filename)
    extract_dataset(filename, path)       
        
@functools.lru_cache(maxsize=100000)
def load_codes():
    print("here?")
    try:
        dx_codes = pd.read_csv('../docs/dx_codes.csv')
        dx_codes['Dx'] = dx_codes['Dx'].str.lower()
        return dx_codes
    except Exception as e:
        print("load_codes.Exception %s " % e)

@functools.lru_cache(maxsize=100000)
def code_to_names(code):
    dx_codes = load_codes().set_index('SNOMED CT Code', drop=True)
    if type(code) == str:
        code = int(code.lstrip('Dx_'))

    try:
        row = dx_codes.loc[code]
        return row['Abbreviation'].strip(), row['Dx'].strip()
    except KeyError:
        return None, None

@functools.lru_cache(maxsize=100000)
def longname_to_code(longname):
    dx_codes = load_codes().set_index('Dx', drop=True)
    return 'Dx_' + str(dx_codes.loc[longname, 'SNOMED CT Code'])

@functools.lru_cache(maxsize=100000)
def shortname_to_code(shortname):
    dx_codes = load_codes().set_index('Abbreviation', drop=True)
    return 'Dx_' + str(dx_codes.loc[shortname, 'SNOMED CT Code'])


def load_wsdb(file):
    record = wfdb.io.rdrecord(file.rstrip('.mat'))
    ecg = record.p_signal.T.astype('float32')
    leads = np.array(record.sig_name)
    sr = record.fs
    
    ecg[np.isnan(ecg)] = 0.0
    
    return ecg, leads, sr


def load_ann(file):
    ann = dict()
    with open(file, 'rt') as f:
        lines = f.readlines()

    num_leads = int(lines.pop(0).split()[1])

    leads = list()
    for i in range(num_leads):
        leads.append(lines.pop(0).split()[-1])

    ann['num_leads'] = num_leads
    
    for index, lead in enumerate(leads):
        ann[f'lead_{lead}'] = index
#     ann['leads'] = np.array(leads)

    for line in lines:
        if line.startswith('#Age'):
            ann['age'] = float(line.split(' ')[-1].strip())
        if line.startswith('#Sex'):
            ann['sex'] = line.split(' ')[-1].strip()
        if line.startswith('#Dx'):
            diagnoses = line.split()[-1].strip()
            if diagnoses != '':
                for diagnosis in diagnoses.split(','):
                    if diagnosis != '':
                        ann[f'Dx_{diagnosis.strip()}'] = 1
    return ann




def get_dataframe(ds_name, ds_path, save=False, load_from_file=True, folder='../docs'):
    data_filename = f'{folder}/{ds_name}.parquet'
    
    if load_from_file and os.path.isfile(data_filename):
        return pd.read_parquet(data_filename)
    
    data = list()
    dataset_path = os.path.join(ds_path, ds_name)
    for filename in tqdm(os.listdir(dataset_path)):
        if filename.endswith('.hea'): 
            full_name = os.path.join(dataset_path, filename)
            row = dict()
            row['dataset'] = ds_name
            row['ann_file'] = full_name
            row['ecg_file'] = full_name.replace('.hea', '.mat')
            
            ecg, leads, sr = load_wsdb(row['ecg_file'])
            ann = load_ann(row['ann_file'])
            row['sr'] = sr
            row['length'] = ecg.shape[1]
            row['time'] = ecg.shape[1] / sr
                       
            row.update(ann)

            data.append(row)
            
    data = pd.DataFrame(data)
    dx_cols = [col for col in data.columns if col.startswith('Dx_')]
    data[dx_cols] = data[dx_cols].fillna(0.0).astype('int32')
    
    if save:
        data.to_parquet(data_filename, index=False)
    
    return data


def get_dataframes(ds_path, names=None, save=False, load_from_file=True, folder='../docs', unite_CPSC2018=True):
    if names is None:
        names = list(NAMES)
    elif type(names) == str:
        names = [names]
        
    if unite_CPSC2018:
        for i in range(len(names)):
            if 'WFDB_CPSC2018' in names[i]:
                names.append('WFDB_CPSC2018')
                names.append('WFDB_CPSC2018_2')
        names = sorted(list(set(names)))
                
        
    data = list()
    for name in names:
        try:
            df = get_dataframe(name, ds_path, save=save, load_from_file=load_from_file, folder=folder)
            data.append(df)
            
        except Exception as e:
            print(e)
            print('Error with dataset', name)
            print(e)
    data = pd.concat(data).reset_index(drop=True)
    dx_cols = [col for col in data.columns if col.startswith('Dx_')]
    data[dx_cols] = data[dx_cols].fillna(0.0).astype('int32')
    
    if unite_CPSC2018:
        data.loc[data['dataset'] == 'WFDB_CPSC2018_2', 'dataset'] = 'WFDB_CPSC2018'
    
    return data