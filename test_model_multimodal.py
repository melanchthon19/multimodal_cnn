#!/usr/bin/env python3
from datetime import datetime
import sys
import os
from os.path import join
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset_multimodal
from dataset_multimodal import Data
import parameters
import models
import compute_metrics
from statistics import mean
import pickle


def test_model_multimodal(i, net, loss_function):

    global tiny, save, device, task, modality, name_model

    print(f'testing model: {name_model}')
    if tiny:
        print(f'testing on tiny {tiny}')

    loss_function = loss_function.to(device)
    threshold = 0.5

    net.to(device)
    net.eval()
    running_loss = []
    running_em_metrics = compute_metrics.EM_metrics()
    running_sb_metrics = compute_metrics.SB_metrics()

    for batch, labels in tqdm(test_generator):

        if task == 'sentiment':
            labels = labels.unsqueeze(1)
        elif task == 'emotion':
            labels = labels
        else:  # task == 'sentiment_binary' or 'sentiment_trinary'
            labels = labels.long()

        labels = labels.to(device)

        if modality == 'unimodal':
            uni = batch
            uni = uni.to(device)
            outputs = net.forward(uni)
        else:  # multimodal
            a, t, v = batch
            a = a.to(device)
            t = t.to(device)
            v = v.to(device)
            outputs = net.forward(a, t, v)
        #print('outputs', outputs)
        loss = loss_function(outputs, labels)
        running_loss.append(loss.item())

        if task == 'emotion':
            compute_metrics.EM(running_em_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)
        if task == 'sentiment_binary':
            compute_metrics.SB2(running_sb_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)
        if task == 'sentiment_trinary':
            compute_metrics.SB3(running_sb_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)

    loss_test = mean(running_loss)

    print(f'Loss test: {loss_test}')

    if save:
        with open(join(dir,'evaluation.txt'), 'a') as f:
            f.write(f'test loss: {loss_test}\n')

    if task == 'sentiment_binary' or task == 'sentiment_trinary':
        acc = mean(running_sb_metrics.acc)
        bacc = mean(running_sb_metrics.bacc)
        acc_not_normalize = sum(running_sb_metrics.acc_not_normalize)
        f1_macro = mean(running_sb_metrics.f1_macro)
        f1_micro = mean(running_sb_metrics.f1_micro)
        f1_weighted = mean(running_sb_metrics.f1_weighted)
        print('metrics SB on test set')
        print(f'acc: {acc:.3f}, bacc: {bacc:.3f}, acc_not_normalize: {acc_not_normalize}, \
        f1_macro: {f1_macro:.3f}, f1_micro: {f1_micro:.3f}, f1_weighted: {f1_weighted:.3f}')

        if save:
            with open(join(dir,'evaluation.txt'), 'a') as f:
                f.write(f'accuracy: {acc}\nbalanced accuracy: {bacc}\naccuracy not normalize: {acc_not_normalize}\
                \nf1_macro: {f1_macro}\nf1_micro: {f1_micro}\nf1_weighted: {f1_weighted}\n')

    if task == 'emotion':
        acc_fraction = mean(running_em_metrics.acc_fraction)
        acc_number = sum(running_em_metrics.acc_number)
        f1_micro = mean(running_em_metrics.f1_micro)
        f1_macro = mean(running_em_metrics.f1_macro)
        f1_samples = mean(running_em_metrics.f1_samples)
        f1_weighted = mean(running_em_metrics.f1_weighted)
        f1_none_h = mean(running_em_metrics.f1_none['happiness'])
        f1_none_sa = mean(running_em_metrics.f1_none['sadness'])
        f1_none_a = mean(running_em_metrics.f1_none['anger'])
        f1_none_su = mean(running_em_metrics.f1_none['surprise'])
        f1_none_d = mean(running_em_metrics.f1_none['disgust'])
        f1_none_f = mean(running_em_metrics.f1_none['fear'])
        print('metrics EM train')
        print(f'acc_frac: {acc_fraction:.3f}, acc_n: {acc_number}, f1_micro: {f1_micro}, f1_none_h: {f1_none_h:.3f} and others...')

        if save:
            with open(join(dir, 'evaluation.txt'), 'a') as f:
                f.write(f'accuracy fraction: {acc_fraction}\naccuracy number: {acc_number}\
                \nf1_macro: {f1_macro}\nf1_micro: {f1_micro}\nf1_weighted: {f1_weighted}\
                \nf1_none_h: {f1_none_h}\nf1_none_sa: {f1_none_sa}\nf1_none_a: {f1_none_a}\
                \nf1_none_su: {f1_none_su}\nf1_none_d: {f1_none_d}\nf1_none_f: {f1_none_f}\n')


if __name__ == '__main__':

    #model = 'UCNN_S_egemapsc_all.pth'  # model to test
    name_model = sys.argv[1]
    folder = sys.argv[2]
    path = f'/group/project/cstr1/mscslp/2019-20/s1872685_daniel_mora/{folder}'
    dir = join(path, name_model[:-4])
    model = open(join(dir,'net.txt'), 'r').readlines()[2][:-2]
    tiny = False
    gpu = True
    save = True

    params_dataset = pickle.load(open(join(dir, 'params_dataset.p'), 'rb'))
    params_model = pickle.load(open(join(dir, 'params_model.p'), 'rb'))
    params_dataset['tiny'] = tiny
    task = params_dataset['task']
    modality = params_dataset['modality']
    aligned2word = params_dataset['aligned2word']

    print(params_dataset)
    print(params_model)
    if 'bert_c_4l' in params_dataset['selected_features'].keys():
        feat = params_dataset['selected_features'].pop('bert_c_4l')
        params_dataset['selected_features']['bert_4l_c'] = feat

    # CUDA
    if gpu:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.set_device(0)
            print("device:", torch.cuda.current_device())
    else: device = 'cpu'

    # DATASET
    _, _, test_generator = dataset_multimodal.get_data_generators(**params_dataset)

    # MODEL
    batch_size = params_dataset['batch_size']
    if model == 'MultiCNN_w':
        net = models.MultiCNN_w(batch_size=batch_size, task=task, **params_model)
    if model == 'MultiCNN_c':
        net = models.MultiCNN_c(batch_size=batch_size, task=task, **params_model)
    if model == 'UniCNN_w':
        params_model['max_len'] = 30
        net = models.UniCNN_w(batch_size=batch_size, task=task, **params_model)
    if model == 'UniCNNSemiDeep_w':
        params_model['max_len'] = 30
        net = models.UniCNNSemiDeep_w(batch_size=batch_size, task=task, **params_model)
    if model == 'UniCNNDeep_w':
        params_model['max_len'] = 30
        net = models.UniCNNDeep_w(batch_size=batch_size, task=task, **params_model)
    if model == 'UniCNNSuperDeep_w':
        params_model['max_len'] = 30
        net = models.UniCNNSuperDeep_w(batch_size=batch_size, task=task, **params_model)
    if model == 'UniCNN_c':
        net = models.UniCNN_c(batch_size=batch_size, task=task, **params_model)

    if task == 'sentiment': loss_function = nn.MSELoss()
    elif task == 'sentiment_binary': loss_function = nn.CrossEntropyLoss()
    elif task == 'sentiment_trinary': loss_function = nn.CrossEntropyLoss()
    elif task == 'emotion': loss_function = nn.BCELoss()

    print(net)
    net.load_state_dict(torch.load(join(dir,name_model)))
    if save:
        with open(join(dir,'evaluation.txt'), 'a') as f:
            f.write(f'model: {name_model}\nmodality: {modality}\ntask: {task}\ntiny:{tiny}\n')

    for i in range(1):
        test_model_multimodal(i, net, loss_function)
