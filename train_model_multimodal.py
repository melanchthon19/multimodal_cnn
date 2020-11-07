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
from statistics import mean
import pickle
from itertools import chain
import argparse
import compute_metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model_multimodal(net, loss_function, training_generator, validation_generator, device):

    global batch_size, save, dir, modality, task

    net.to(device)
    net = net.float()
    print(net)
    # Getting number of parameters
    num_parameters = count_parameters(net)
    print(f'The model has {num_parameters} trainable parameters')
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    print(f'optimizer: {optimizer}')
    loss_function = loss_function.to(device)

    loss_training, loss_valid = [], []
    df_outputs = pd.DataFrame()#columns=range(50))

    max_epochs = 100
    best_loss = 100  # initial random high number
    threshold = 0.5  # greater than 0.5 is considered 1.0 for classification

    if save:
        with open(join(dir, 'net.txt'), 'w') as f:  # saving model's parameters
            f.write(f'{"="*20}\n{datetime.now()}\n{str(net)}\n\noptimizer: {optimizer}\
            \nloss function: {loss_function}\nmax epochs: {max_epochs}\ntiny: {params_dataset["tiny"]}\
            \nnumber of trainable parameters: {num_parameters}\nbalanced polarity: {params_dataset["balance_polarity"]}\
            \nmax length: {params_dataset["max_len"]}\nselected features: {params_dataset["selected_features"]}\
            \nnumber of parameters: {num_parameters}\ntask: {task}\nmodality: {modality}\nbatch_size: {batch_size}\n\n')

        with open(join(dir,'metrics_loss_function.csv'), 'w') as f:
            f.write('epoch,loss_training,loss_valid\n')

        if task == 'sentiment_binary' or task == 'sentiment_trinary':
            with open(join(dir, 'metrics_SB_train.csv'), 'w') as f:
                f.write('epoch,accuracy,balancedacc,accnotnormalize,f1_macro,f1_micro,f1_weighted\n')
            with open(join(dir, 'metrics_SB_valid.csv'), 'w') as f:
                f.write('epoch,accuracy,balancedacc,accnotnormalize,f1_macro,f1_micro,f1_weighted\n')

        if task == 'emotion':
            with open(join(dir, 'metrics_EM_train.csv'), 'w') as f:
                f.write('epoch,acc_fraction,acc_number,f1_macro,f1_micro,f1_weighted,f1_h,f1_sa,f1_a,f1_su,f1_d,f1_f\n')
            with open(join(dir, 'metrics_EM_valid.csv'), 'w') as f:
                f.write('epoch,acc_fraction,acc_number,f1_macro,f1_micro,f1_weighted,f1_h,f1_sa,f1_a,f1_su,f1_d,f1_f\n')

    for epoch in range(max_epochs):
        # Training
        net.train()
        running_loss = []
        running_em_metrics = compute_metrics.EM_metrics()
        running_sb_metrics = compute_metrics.SB_metrics()

        for batch, labels in tqdm(training_generator):

            if task == 'sentiment':
                labels = labels.unsqueeze(1)
            elif task == 'emotion':
                labels = labels
            else:  # task == 'sentiment_binary' or 'sentiment_trinary'
                labels = labels.long()
            # Set parameters to zero
            net.zero_grad()

            # Transfer to GPU
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
            #print('labels', labels)
            loss = loss_function(outputs, labels)
            #print('loss', loss.item())
            running_loss.append(loss.item())

            loss.backward()  # Compute gradients
            optimizer.step()  # Does the update

            if task == 'emotion':
                compute_metrics.EM(running_em_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)
            if task == 'sentiment_binary':
                compute_metrics.SB2(running_sb_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)
            if task == 'sentiment_trinary':
                compute_metrics.SB3(running_sb_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)

        loss_training.append(mean(running_loss))
        print(f'Epoch: {epoch}. Loss training: {loss_training[epoch]}')

        if task == 'sentiment_binary' or task == 'sentiment_trinary':
            acc = mean(running_sb_metrics.acc)
            bacc = mean(running_sb_metrics.bacc)
            acc_not_normalize = sum(running_sb_metrics.acc_not_normalize)
            f1_macro = mean(running_sb_metrics.f1_macro)
            f1_micro = mean(running_sb_metrics.f1_micro)
            f1_weighted = mean(running_sb_metrics.f1_weighted)
            print(f'metrics SB train\nacc: {acc:.3f}, bacc: {bacc:.3f}, acc_not_normalize: {acc_not_normalize}, f1_macro: {f1_macro:.3f}, f1_micro: {f1_micro:.3f}, f1_weighted: {f1_weighted:.3f}')

            if save:
                with open(join(dir,'metrics_SB_train.csv'), 'a') as f:
                    f.write(f'{epoch},{acc},{bacc},{acc_not_normalize},{f1_macro},{f1_micro},{f1_weighted}\n')

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
            print(f'metrics EM train\nacc_frac: {acc_fraction:.3f}, acc_n: {acc_number}, f1_micro: {f1_micro}, f1_none_h: {f1_none_h:.3f} and others...')

            if save:
                with open(join(dir, 'metrics_EM_train.csv'), 'a') as f:
                    f.write(f'{epoch},{acc_fraction},{acc_number},{f1_macro},{f1_micro},{f1_weighted},\
                    {f1_none_h},{f1_none_sa},{f1_none_a},{f1_none_su},{f1_none_d},{f1_none_f}\n')

        # Validation
        with torch.set_grad_enabled(False):
            running_loss = []
            running_sb_metrics = compute_metrics.SB_metrics()
            running_em_metrics = compute_metrics.EM_metrics()
            net.eval()

            for batch, labels in tqdm(validation_generator):

                if task == 'sentiment':
                    labels = labels.unsqueeze(1)
                elif task == 'emotion':
                    labels = labels
                else:  # task == 'sentiment_binary' or 'sentiment_trinary'
                    labels = labels.long()
                # Transfer to GPU
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

                if save:
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(net.state_dict(), join(dir, model_name))

                if task == 'emotion':
                    compute_metrics.EM(running_em_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)
                if task == 'sentiment_binary':
                    compute_metrics.SB2(running_sb_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)
                if task == 'sentiment_trinary':
                    compute_metrics.SB3(running_sb_metrics, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy(), threshold)

            loss_valid.append(mean(running_loss))
            print(f'Epoch: {epoch}. Loss validation: {loss_valid[epoch]}')

            if task == 'sentiment_binary' or task == 'sentiment_trinary':
                acc = mean(running_sb_metrics.acc)
                bacc = mean(running_sb_metrics.bacc)
                acc_not_normalize = sum(running_sb_metrics.acc_not_normalize)
                f1_macro = mean(running_sb_metrics.f1_macro)
                f1_micro = mean(running_sb_metrics.f1_micro)
                f1_weighted = mean(running_sb_metrics.f1_weighted)
                print(f'metrics SB valid\nacc: {acc:.3f}, bacc: {bacc:.3f}, acc_not_normalize: {acc_not_normalize}, f1_macro: {f1_macro:.3f}, f1_micro: {f1_micro:.3f}, f1_weighted: {f1_weighted:.3f}')

                if save:
                    with open(join(dir,'metrics_SB_valid.csv'), 'a') as f:
                        f.write(f'{epoch},{acc},{bacc},{acc_not_normalize},{f1_macro},{f1_micro},{f1_weighted}\n')

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
                print(f'metrics EM valid\nacc_frac: {acc_fraction:.3f}, acc_n: {acc_number}, f1_micro: {f1_micro}, f1_none_h: {f1_none_h:.3f} and others...')

                if save:
                    with open(join(dir, 'metrics_EM_valid.csv'), 'a') as f:
                        f.write(f'{epoch},{acc_fraction},{acc_number},{f1_macro},{f1_micro},{f1_weighted},\
                        {f1_none_h},{f1_none_sa},{f1_none_a},{f1_none_su},{f1_none_d},{f1_none_f}\n')

            if save:
                with open(join(dir,'metrics_loss_function.csv'), 'a') as f:
                    f.write(f'{epoch},{loss_training[epoch]},{loss_valid[epoch]}\n')

    if save:
        with open(join(dir, 'net.txt'), 'a') as f:
            f.write(f'END TRAINING: {datetime.now()} {"="*20}')


if __name__ == '__main__':
    print('started training', datetime.now())
    torch.manual_seed(15)
    np.random.seed(15)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model to use", type=str)
    args = parser.parse_args()

    # IMPORTING PARAMETERS
    params_model = parameters.params_models[args.model]
    params_dataset = parameters.get_params_dataset(args.model)
    save = parameters.save
    model_name = parameters.model_name
    modality = params_dataset['modality']
    task = params_dataset['task']
    batch_size = params_dataset['batch_size']
    print('\nparameters model')
    for k, v in params_model.items(): print('\t',k, v)
    print('\nparamaters dataset')
    for k, v in params_dataset.items(): print('\t',k, v)

    # TO SAVE MODEL
    if save:
        path = '/group/project/cstr1/mscslp/2019-20/s1872685_daniel_mora/models_d/'
        dir = join(path, model_name[:-4])
        if os.path.exists(dir):
            if input(f'folder {dir} already exists.. continue? [yes/no]') != 'yes':
                sys.exit()
        else:
            os.mkdir(dir)
        pickle.dump(params_dataset, open(join(dir, 'params_dataset.p'), 'wb'))
        pickle.dump(params_model, open(join(dir, 'params_model.p'), 'wb'))
    else:
        dir = None

    # CUDA
    gpu = True
    if gpu:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.set_device(0)
            print("device:", torch.cuda.current_device())
    else: device = 'cpu'

    # DATASET

    training_generator, validation_generator, test_generator = \
        dataset_multimodal.get_data_generators(**params_dataset)

    # MODEL
    if args.model == 'MultiCNN_w':
        params_model['max_len'] = params_dataset['max_len']
        net = models.MultiCNN_w(batch_size=batch_size, task=task, **params_model)
    if args.model == 'MultiCNN_c':
        net = models.MultiCNN_c(batch_size=batch_size, task=task, **params_model)
    if args.model == 'BiCNN_w':
        params_model['max_len'] = params_dataset['max_len']
        net = models.BiCNN_w(batch_size=batch_size, task=task, **params_model)
    if args.model == 'BiCNN_c':
        net = models.BiCNN_c(batch_size=batch_size, task=task, **params_model)
    if args.model == 'UniCNN_w':
        params_model['max_len'] = params_dataset['max_len']
        net = models.UniCNN_w(batch_size=batch_size, task=task, **params_model)
    if args.model == 'UniCNNSemiDeep_w':
        params_model['max_len'] = params_dataset['max_len']
        net = models.UniCNNSemiDeep_w(batch_size=batch_size, task=task, **params_model)
    if args.model == 'UniCNNDeep_w':
        params_model['max_len'] = params_dataset['max_len']
        net = models.UniCNNDeep_w(batch_size=batch_size, task=task, **params_model)
    if args.model == 'UniCNNSuperDeep_w':
        params_model['max_len'] = params_dataset['max_len']
        net = models.UniCNNSuperDeep_w(batch_size=batch_size, task=task, **params_model)
    if args.model == 'UniCNN_c':
        net = models.UniCNN_c(batch_size=batch_size, task=task, **params_model)

    if task == 'sentiment': loss_function = nn.MSELoss()
    elif task == 'sentiment_binary': loss_function = nn.CrossEntropyLoss()
    elif task == 'sentiment_trinary': loss_function = nn.CrossEntropyLoss()
    elif task == 'emotion': loss_function = nn.BCELoss()
    #loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor([16576/23259, 6683/23259]), reduction='mean')
    # 11476+5100 positive + neutral | 6683 negative

    # TRAINING
    train_model_multimodal(net, loss_function, training_generator, validation_generator, device)

    print('finished training', datetime.now())
