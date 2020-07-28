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
from dataset_multimodal import get_data_generators
import parameters_multimodal
from convnet_multimodal import SentConvNet, EmoConvNet, SentBiConvNet
from sklearn import metrics
from statistics import mean


def calculate_metrics_emotion(prediction, target, threshold=0.5):
    labels = ['happiness', 'sadness','anger', 'surprise', 'disgust', 'fear']
    pred_cpu = prediction.to('cpu')
    score = np.array(pred_cpu.data, dtype=float)  # --> [0.5089, 0.4911, 0.4081, 0.4216, 0.4322, 0.4575]
    prediction = np.array(pred_cpu.data > threshold, dtype=float)  # --> [1., 0., 0., 0., 0., 0.] per batch
    target = target.to('cpu')
    target = np.array(target.data, dtype=float)

    acc = metrics.accuracy_score(y_true=target, y_pred=prediction)
    f1 = metrics.f1_score(y_true=target, y_pred=prediction, zero_division = 0, average = 'samples')
    return acc, f1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model_sentiment(net, loss_function, training_generator, validation_generator, tiny, device, dir):
    if tiny:
        print(f'training on tiny dataset')

    print('training on device:', device)
    net.to(device)
    net = net.float()
    print(net)
    # Getting number of parameters
    print(f'The model has {count_parameters(net):,} trainable parameters')
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #loss_function = nn.MSELoss()
    loss_function = loss_function

    # Loop over epochs
    loss_training = []
    loss_valid = []
    max_epochs = 50
    best_loss = 100 # initial random high number
    best_state_dict = {}

    with open(join(dir, 'net.txt'), 'w') as f:
        f.write('TRAINING {}\n{}\n{}\n\n{}\n{}\n{}\n{}\n\n'.format('='*20,
        datetime.now(), str(net), optimizer, loss_function, max_epochs, tiny))

    with open(join(dir,'metrics.csv'), 'w') as f:
        f.write('epoch,loss_training,loss_valid\n')

    for epoch in range(max_epochs):
        # Training
        net.train()
        running_loss = []
        for batch, labels in tqdm(training_generator):
            # Transfer to GPU
            labels = labels.to(device)
            a, t, v = batch
            a = a.to(device)
            t = t.to(device)
            v = v.to(device)

            labels_reshaped = labels.unsqueeze(1)  # reshaping to match input

            # Model computations
            net.zero_grad()
            outputs = net.forward(a, t, v)
            #print('outputs', outputs.shape)
            loss = loss_function(outputs, labels_reshaped)
            loss.backward()  # Compute gradients
            optimizer.step()  # Does the update
            running_loss.append(loss.item())

        loss_training.append(mean(running_loss))
        print(f'Epoch: {epoch}. Loss training: {loss_training[epoch]}')

        # Validation
        with torch.set_grad_enabled(False):
            net.eval()
            running_loss = []
            for batch, labels in tqdm(validation_generator):
                # Transfer to GPU
                labels = labels.to(device)
                a, t, v = batch
                a = a.to(device)
                t = t.to(device)
                v = v.to(device)

                labels_reshaped = labels.unsqueeze(1)

                # Model computations
                outputs = net(a, t, v)
                loss = loss_function(outputs, labels_reshaped)

                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = net.state_dict()
                    torch.save(best_state_dict, join(dir, model_name))
                running_loss.append(loss.item())

            loss_valid.append(mean(running_loss))

            with open(join(dir,'metrics.csv'), 'a') as f:
                f.write(f'{epoch},{loss_training[epoch]},{loss_valid[epoch]}\n')
            print(f'Epoch: {epoch}. Loss validation: {loss}')

    with open(join(dir, 'net.txt'), 'a') as f:
        f.write('{}\n{}'.format(datetime.now(), '='*20))


def train_model_sentiment_binary(net, loss_function, training_generator, validation_generator, tiny, device, dir):
    if tiny:
        print(f'training on tiny dataset')

    print('training on device:', device)
    net.to(device)
    net = net.float()
    print(net)
    # Getting number of parameters
    print(f'The model has {count_parameters(net):,} trainable parameters')
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #loss_function = nn.CELoss()
    loss_function = loss_function.to(device)

    # Loop over epochs
    loss_training = []
    loss_valid = []
    max_epochs = 50
    best_loss = 100 # initial random high number
    best_state_dict = {}

    with open(join(dir, 'net.txt'), 'w') as f:
        f.write('TRAINING {}\n{}\n{}\n\n{}\n{}\n{}\n{}\n\n'.format('='*20,
        datetime.now(), str(net), optimizer, loss_function, max_epochs, tiny))

    with open(join(dir,'metrics.csv'), 'w') as f:
        f.write('epoch,loss_training,loss_valid\n')

    for epoch in range(max_epochs):
        # Training
        net.train()
        running_loss = []
        for batch, labels in tqdm(training_generator):
            # Transfer to GPU
            _, labels = labels.max(dim=1)
            labels = labels.to(device)
            a, t, v = batch
            a = a.to(device)
            t = t.to(device)
            v = v.to(device)

            # Model computations
            net.zero_grad()
            outputs = net.forward(a, t, v)
            #print('outputs', outputs.shape)
            loss = loss_function(outputs, labels)
            loss.backward()  # Compute gradients
            optimizer.step()  # Does the update
            running_loss.append(loss.item())

        loss_training.append(mean(running_loss))
        print(f'Epoch: {epoch}. Loss training: {loss_training[epoch]}')

        # Validation
        with torch.set_grad_enabled(False):
            net.eval()
            running_loss = []
            for batch, labels in tqdm(validation_generator):
                # Transfer to GPU
                _, labels = labels.max(dim=1)
                labels = labels.to(device)
                a, t, v = batch
                a = a.to(device)
                t = t.to(device)
                v = v.to(device)

                # Model computations
                outputs = net(a, t, v)
                loss = loss_function(outputs, labels)

                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = net.state_dict()
                    torch.save(best_state_dict, join(dir, model_name))
                running_loss.append(loss.item())

            loss_valid.append(mean(running_loss))

            with open(join(dir,'metrics.csv'), 'a') as f:
                f.write(f'{epoch},{loss_training[epoch]},{loss_valid[epoch]}\n')
            print(f'Epoch: {epoch}. Loss validation: {loss}')

    with open(join(dir, 'net.txt'), 'a') as f:
        f.write('{}\n{}'.format(datetime.now(), '='*20))


def train_model_emotion(net, loss_function, training_generator, validation_generator, device):
    global save, tiny
    if tiny:
        print('training on tiny dataset')
    print('training on device:', device)

    net.to(device)
    net = net.float()
    print(net)
    # Getting number of parameters
    print(f'The model has {count_parameters(net):,} trainable parameters')
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = loss_function

    loss_training, loss_valid = [], []
    acc_training, acc_valid = [], []
    f1_training, f1_valid = [], []
    max_epochs = 50
    best_loss = 100 # initial random high number
    best_state_dict = {}
    threshold = 0.5  # > than 0.5 is considered 1.0


    with open(join(dir, 'net.txt'), 'w') as f:
        f.write('{}\n{}\n{}\n\n{}\n{}\n{}\n{}\n\n'.format('='*20,
        datetime.now(), str(net), optimizer, loss_function, max_epochs, tiny))

    with open(join(dir,'metrics.csv'), 'w') as f:
        f.write('epoch,loss_training,loss_valid,acc_training,acc_valid,f1_training,f1_valid\n')

    for epoch in range(max_epochs):
        # Training
        net.train()
        running_loss = []
        running_acc = []
        running_f1 = []
        for batch, labels in tqdm(training_generator):
            # Transfer to GPU
            if task == 'sentiment_binary':
                _, labels = labels.max(dim=1)
            labels = labels.to(device)
            a, t, v = batch
            a = a.to(device)
            t = t.to(device)
            v = v.to(device)

            # Model computations
            net.zero_grad()

            outputs = net.forward(a, t, v)

            loss = loss_function(outputs, labels)
            loss.backward()  # Compute gradients
            optimizer.step()  # Does the update

            if task == 'emotion':
                acc, f1 = calculate_metrics_emotion(outputs, labels, threshold=threshold)
                running_acc.append(acc)
                running_f1.append(f1)
            running_loss.append(loss.item())

        loss_training.append(mean(running_loss))
        acc_training.append(mean(running_acc))
        f1_training.append(mean(running_f1))
        print(f'Epoch: {epoch}. Loss training: {loss_training[epoch]}')

        # Validation
        with torch.set_grad_enabled(False):
            running_loss = []
            running_acc = []
            running_f1 = []
            net.eval()
            for batch, labels in tqdm(validation_generator):
                # Transfer to GPU
                labels = labels.to(device)
                a, t, v = batch
                a = a.to(device)
                t = t.to(device)
                v = v.to(device)

                # Model computations
                outputs = net(a, t, v)
                loss = loss_function(outputs, labels)

                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = net.state_dict()

                acc, f1 = calculate_metrics(outputs, labels, threshold=threshold)
                running_acc.append(acc)
                running_f1.append(f1)
                running_loss.append(loss.item())

            acc_valid.append(mean(running_acc))
            f1_valid.append(mean(running_f1))
            loss_valid.append(mean(running_loss))

            print(f'Epoch: {epoch}. Loss validation: {loss_valid[epoch]}')

            with open(join(dir,'metrics.csv'), 'a') as f:
                f.write(f'{epoch},{loss_training[epoch]},{loss_valid[epoch]},\
                                  {acc_training[epoch]},{acc_valid[epoch]},\
                                  {f1_training[epoch]},{f1_valid[epoch]}\n')

    with open(join(dir, 'net.txt'), 'a') as f:
        f.write('END TRAINING: {}\n{}'.format(datetime.now(), '='*20))


def train_model_multimodal(net, loss_function, training_generator, validation_generator, device):

    global save, dir, tiny

    if tiny:
        print('training on tiny dataset')
    print('training on device:', device)

    net.to(device)
    net = net.float()
    print(net)
    # Getting number of parameters
    print(f'The model has {count_parameters(net):,} trainable parameters')
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = loss_function.to(device)

    loss_training, loss_valid = [], []
    acc_training, acc_valid = [], []
    f1_training, f1_valid = [], []

    max_epochs = 2
    best_loss = 100  # initial random high number
    threshold = 0.5  # greater than 0.5 is considered 1.0 for emotion classification

    if save:
        with open(join(dir, 'net.txt'), 'w') as f:  # saving model's parameters
            f.write(f'{"="*20}\n{datetime.now()}\n{str(net)}\n\noptimizer: {optimizer}\
            \nloss function: {loss_function}\nmax epochs: {max_epochs}\ntiny: {tiny}\n\n')
        with open(join(dir,'metrics.csv'), 'w') as f:  # metrics csv
            f.write('epoch,loss_training,loss_valid,acc_training,acc_valid,f1_training,f1_valid\n')

    for epoch in range(max_epochs):
        # Training
        net.train()
        running_loss = []
        running_acc = []
        running_f1 = []
        for batch, labels in tqdm(training_generator):
            if task == 'sentiment':
                labels = labels.unsqueeze(1)

            if task == 'sentiment_binary':
                _, labels = labels.max(dim=1)

            # Transfer to GPU
            labels = labels.to(device)
            a, t, v = batch
            a = a.to(device)
            t = t.to(device)
            v = v.to(device)

            # Model computations
            net.zero_grad()
            outputs = net.forward(a, t, v)

            loss = loss_function(outputs, labels)
            loss.backward()  # Compute gradients
            optimizer.step()  # Does the update

            if task == 'emotion':
                acc, f1 = calculate_metrics_emotion(outputs, labels, threshold=threshold)
                running_acc.append(acc)
                running_f1.append(f1)
            running_loss.append(loss.item())

        loss_training.append(mean(running_loss))

        if task == 'emotion':
            acc_training.append(mean(running_acc))
            f1_training.append(mean(running_f1))

        print(f'Epoch: {epoch}. Loss training: {loss_training[epoch]}')

        # Validation
        with torch.set_grad_enabled(False):
            running_loss = []
            running_acc = []
            running_f1 = []
            net.eval()
            for batch, labels in tqdm(validation_generator):
                # Transfer to GPU
                if task == 'sentiment_binary':
                    _, labels = labels.max(dim=1)

                if task == 'sentiment':
                    labels = labels.unsqueeze(1)

                labels = labels.to(device)
                a, t, v = batch
                a = a.to(device)
                t = t.to(device)
                v = v.to(device)

                # Model computations
                outputs = net(a, t, v)
                loss = loss_function(outputs, labels)

                if save:
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(net.state_dict(), dir)

                if task == 'emotion':
                    acc, f1 = calculate_metrics_emotion(outputs, labels, threshold=threshold)
                    running_acc.append(acc)
                    running_f1.append(f1)
                running_loss.append(loss.item())

            if task == 'emotion':
                acc_valid.append(mean(running_acc))
                f1_valid.append(mean(running_f1))
            loss_valid.append(mean(running_loss))

            print(f'Epoch: {epoch}. Loss validation: {loss_valid[epoch]}')

            if save:
                with open(join(dir,'metrics.csv'), 'a') as f:
                    if task == 'emotion':
                        f.write(f'{epoch},{loss_training[epoch]},{loss_valid[epoch]},\
                                          {acc_training[epoch]},{acc_valid[epoch]},\
                                          {f1_training[epoch]},{f1_valid[epoch]}\n')
                    else:
                        f.write(f'{epoch},{loss_training[epoch]},{loss_valid[epoch]}\n')

    if save:
        with open(join(dir, 'net.txt'), 'a') as f:
            f.write(f'END TRAINING: {datetime.now()} {"="*20}')


if __name__ == '__main__':
    print('started training', datetime.now())
    torch.manual_seed(19)
    np.random.seed(19)

    # IMPORTING PARAMETERS
    params_model = parameters_multimodal.params_CNN
    params_dataset = parameters_multimodal.params_dataset

    # SAVE MODEL
    save = False
    if save:
        model_name = str(datetime.now())[:19]+'.pth'
        #model_name = 'MCNN_BS_ccc_.pth'  # model will be saved with this name
        path = '/group/project/cstr1/mscslp/2019-20/s1872685_daniel_mora/model_multimodal/'
        dir = join(path, model_name[:-4])
        if os.path.exists(dir):
            if input(f'folder {dir} already exists.. continue? [yes/no]') != 'yes':
                sys.exit()
        else:
            os.mkdir(dir)
    else:
        dir = None

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:4" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(4)
        print("device:", torch.cuda.current_device())

    # DATASET
    training_generator, validation_generator, _ = \
        get_data_generators(batch_size = params_model['batch_size'], **params_dataset)
    tiny = params_dataset['tiny']
    task = params_dataset['task']

    # MODEL
    if params_dataset['task'] == 'sentiment':
        net = SentConvNet(device = device, **params_model)
        loss_function = nn.MSELoss()

    elif params_dataset['task'] == 'sentiment_binary':
        net = SentBiConvNet(device = device, **params_model)
        loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor([16576, 6683]), reduction='mean')
        # 11476+5100 positive + neutral | 6683 negative

    elif params_dataset['task'] == 'emotion':
        net = EmoConvNet(device = device, **params_model)
        loss_function = nn.BCELoss()

    # TRAINING
    train_model_multimodal(net, loss_function, training_generator, validation_generator, device)

    print('finished training', datetime.now())
