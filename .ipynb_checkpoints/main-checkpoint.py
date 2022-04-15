import wfdb
import _pickle as cPickle
import torch
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import sys
import fire
import os

def classify(path):
    try:
        with open(path) as json_file:
                    final = json.load(json_file)
        if 'PATH_TO_LIB' not in final:
            print('Необходимо указать путь до папки с библиотекой')
            return
        sys.path.append(final['PATH_TO_LIB'])
        from processing_2 import preprocess
        ecgs = list()
        file_names = list()

        if 'FILE' not in final:
            print('Необходимо указать путь до папки с файлами .mat и .hea')
            return
        if not os.path.isdir(final['FILE']):
            print('Некорректный путь до директории')
            return
        for filename in tqdm(os.listdir(final['FILE'])):
            if filename.endswith('.mat'): 
                file_names.append(filename.rstrip('.mat'))
                full_name = os.path.join(final['FILE'], filename)
                try:
                    record = wfdb.io.rdrecord(full_name.rstrip('.mat'))
                except Exception as exp:
                    print('Произошла ошибка при работе с данными формата .mat и .hea')
                    print(exp)
                    return
                try:
                    sr = record.fs
                    ecg = record.p_signal.T.astype('float32')
                except Exception as exp:
                    print('Произошла ошибка при изъятии из файла формата .mat частоты дискретизации и ЭКГ-сигнала. Возможно, битый файл')
                    print(exp)
                    return
                try:
                    ecg = preprocess(ecg, sr, final)
                except Exception as exp:
                    print('Произошла ошибка при предобработке сигнала. Возможно, некорректные данные в файле .json')
                    print(exp)
                    return
                if 'N_LEADS' not in final:
                    print('Необходимо указать число каналов')
                    return
                if int(final['N_LEADS'])<1 or int(final['N_LEADS'])>12:
                    print('Число каналов должно быть целым положительным числом <=12')
                    return
                ecg = ecg[:final['N_LEADS'], :]
                ecgs.append(ecg)
        ecgs = np.stack(ecgs)
        if "GENERATOR" not in final:
            print('Необходимо указать путь до весов генератора')
            return
        if not os.path.isfile(final["GENERATOR"]):
            print('Неверный путь до весов генератора')
            return
        filename, file_extension = os.path.splitext(final["GENERATOR"])
        if not file_extension=='.pkl':
            print('Неверный формат файла с весами генератора. Должен быть .pkl')
            return
        try:
            with open(f'{final["GENERATOR"]}', 'rb') as fid:
                        netG = cPickle.load(fid)
        except Exception as exp:
                    print('Произошла ошибка загрузке файла с весами генератора. Возможно битый файл')
                    print(exp)
                    return
        criterion = torch.nn.BCEWithLogitsLoss()
        if "CLASSIFIER" not in final:
            print('Необходимо указать путь до весов классификатора')
            return
        if not os.path.isfile(final["CLASSIFIER"]):
            print('Неверный путь до весов классификатора')
            return
        filename, file_extension = os.path.splitext(final["CLASSIFIER"])
        if not file_extension=='.pkl':
            print('Неверный формат файла с весами классификатора. Должен быть .pkl')
            return
        try:
            with open(f'{final["CLASSIFIER"]}', 'rb') as fid:
                    net = cPickle.load(fid) 
        except Exception as exp:
                    print('Произошла ошибка загрузке файла с весами классификатора. Возможно битый файл')
                    print(exp)
                    return
        net.cuda()
        net.eval()
        try:
            result = net(torch.Tensor(ecgs).float().cuda()).float().cuda()
        except Exception as exp:
                    print('Произошла ошибка при классификации данных. Возможно ошибка во входных данных .json')
                    print(exp)
                    return
        result = torch.sigmoid(result).detach().cpu().numpy()
        dictionary = dict(zip(file_names, result))
        df = pd.DataFrame.from_dict(dictionary, orient='index',   columns = ['probability']) 
        df.to_csv(final['RESULT'])
    except Exception as exp:
                        print('Произошла ошибка при классификации данных. Возможно ошибка во входных данных .json')
                        print(exp)

if __name__ == '__main__': 
    fire.Fire(classify) 