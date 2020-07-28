#!/usr/bin/env python3
import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset_multimodal import get_data_generators
import parameters_multimodal


class SentConvNet(nn.Module):
    def __init__(self, batch_size, emb_dims, in_chan, out_chan, kernel, padding, max_pool, stride, dropout, device):
        super(SentConvNet, self).__init__()
        self.batch_size = batch_size
        self.emb_dim_a = emb_dims[0]
        self.emb_dim_t = emb_dims[1]
        self.emb_dim_v = emb_dims[2]
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.device = device
        self.stride = stride
        # batch 5, in 1, out 10, kernel 5, maxpool 2, padding 2
        self.conv1d = nn.Conv1d(self.in_chan, self.out_chan, kernel_size=kernel, padding=padding)
        # 5, 1, 88 --> 5, out_chan, emb_dim-(kernel-1)+(padding*2) = 5, 10, 88-(5-1)+(2*2) = 5, 10, 88
        self.batchnorm1d = nn.BatchNorm1d(self.out_chan)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # maxpool slices by its value
        self.maxpool1d = nn.MaxPool1d(max_pool)  # 5, 10, 88 --> 5, 10, (88//2)
        dim_a = (self.emb_dim_a - (kernel-1) + (padding*2)) // max_pool
        dim_t = (self.emb_dim_t - (kernel-1) + (padding*2)) // max_pool
        dim_v = (self.emb_dim_v - (kernel-1) + (padding*2)) // max_pool

        # SENTIMENT DENSE LAYERS
        self.fc1_a = nn.Linear(self.out_chan*dim_a, dim_a//2)
        self.fc1_t = nn.Linear(self.out_chan*dim_t, dim_t//2)
        self.fc1_v = nn.Linear(self.out_chan*dim_v, dim_v//2)
        self.fc2_a = nn.Linear(dim_a//2, 1)
        self.fc2_t = nn.Linear(dim_t//2, 1)
        self.fc2_v = nn.Linear(dim_v//2, 1)
        self.ff = nn.Linear(3,1)

    def extract_features_sentiment(self, sequence, fc1, fc2):
        x = sequence.unsqueeze_(1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.batchnorm1d(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.dropout(x)
        x = fc1(x.view(self.batch_size, -1))
        x = self.relu(x)
        x = fc2(x.view(self.batch_size, -1))
        x = self.relu(x)
        return x

    def forward(self, a, t, v):
        hid_a = self.extract_features_sentiment(a, self.fc1_a, self.fc2_a)
        hid_t = self.extract_features_sentiment(t, self.fc1_t, self.fc2_t)
        hid_v = self.extract_features_sentiment(v, self.fc1_v, self.fc2_v)
        h = torch.cat((hid_a, hid_t, hid_v), dim=1)
        h = self.ff(h)
        return h


class SentBiConvNet(nn.Module):
    def __init__(self, batch_size, emb_dims, in_chan, out_chan, kernel, padding, max_pool, stride, dropout, device):
        super(SentBiConvNet, self).__init__()
        self.batch_size = batch_size
        self.emb_dim_a = emb_dims[0]
        self.emb_dim_t = emb_dims[1]
        self.emb_dim_v = emb_dims[2]
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.device = device
        self.stride = stride
        # batch 5, in 1, out 10, kernel 5, maxpool 2, padding 2
        self.conv1d = nn.Conv1d(self.in_chan, self.out_chan, kernel_size=kernel, padding=padding)
        # 5, 1, 88 --> 5, out_chan, emb_dim-(kernel-1)+(padding*2) = 5, 10, 88-(5-1)+(2*2) = 5, 10, 88
        self.batchnorm1d = nn.BatchNorm1d(self.out_chan)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        # maxpool slices by its value
        self.maxpool1d = nn.MaxPool1d(max_pool)  # 5, 10, 88 --> 5, 10, (88//2)
        dim_a = (self.emb_dim_a - (kernel-1) + (padding*2)) // max_pool
        dim_t = (self.emb_dim_t - (kernel-1) + (padding*2)) // max_pool
        dim_v = (self.emb_dim_v - (kernel-1) + (padding*2)) // max_pool

        # SENTIMENT DENSE LAYERS
        self.fc1_a = nn.Linear(self.out_chan*dim_a, dim_a//2)
        self.fc1_t = nn.Linear(self.out_chan*dim_t, dim_t//2)
        self.fc1_v = nn.Linear(self.out_chan*dim_v, dim_v//2)
        self.fc2_a = nn.Linear(dim_a//2, 1)
        self.fc2_t = nn.Linear(dim_t//2, 1)
        self.fc2_v = nn.Linear(dim_v//2, 1)
        self.ff = nn.Linear(3,2)

    def extract_features_sentiment(self, sequence, fc1, fc2):
        x = sequence.unsqueeze_(1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.batchnorm1d(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.dropout(x)
        x = fc1(x.view(self.batch_size, -1))
        x = self.relu(x)
        x = fc2(x.view(self.batch_size, -1))
        x = self.relu(x)
        return x

    def forward(self, a, t, v):
        hid_a = self.extract_features_sentiment(a, self.fc1_a, self.fc2_a)
        hid_t = self.extract_features_sentiment(t, self.fc1_t, self.fc2_t)
        hid_v = self.extract_features_sentiment(v, self.fc1_v, self.fc2_v)
        h = torch.cat((hid_a, hid_t, hid_v), dim=1)
        h = self.ff(h)
        h = self.softmax(h)
        return h


class EmoConvNet(nn.Module):
    def __init__(self, batch_size, emb_dims, in_chan, out_chan, kernel, padding, max_pool, stride, dropout, device):
        super(EmoConvNet, self).__init__()
        self.batch_size = batch_size
        self.emb_dim_a = emb_dims[0]
        self.emb_dim_t = emb_dims[1]
        self.emb_dim_v = emb_dims[2]
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.device = device
        self.stride = stride
        self.threshold = torch.FloatTensor([0.5])
        self.not_emo = torch.zeros(self.batch_size, 6)
        self.emo = torch.ones(self.batch_size, 6)

        # batch 5, in 1, out 10, kernel 5, maxpool 2, padding 2
        self.conv1d = nn.Conv1d(self.in_chan, self.out_chan, kernel_size=kernel, padding=padding)
        # 5, 1, 88 --> 5, out_chan, emb_dim-(kernel-1)+(padding*2) = 5, 10, 88-(5-1)+(2*2) = 5, 10, 88
        self.batchnorm1d = nn.BatchNorm1d(self.out_chan)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # maxpool slices by its value
        self.maxpool1d = nn.MaxPool1d(max_pool)  # 5, 10, 88 --> 5, 10, (88//2)
        dim_a = (self.emb_dim_a - (kernel-1) + (padding*2)) // max_pool
        dim_t = (self.emb_dim_t - (kernel-1) + (padding*2)) // max_pool
        dim_v = (self.emb_dim_v - (kernel-1) + (padding*2)) // max_pool

        # EMOTION DENSE LAYERS
        self.fc1_a = nn.Linear(self.out_chan*dim_a, dim_a//2)
        self.fc1_t = nn.Linear(self.out_chan*dim_t, dim_t//2)
        self.fc1_v = nn.Linear(self.out_chan*dim_v, dim_v//2)
        self.fc2_a = nn.Linear(dim_a//2, 6)
        self.fc2_t = nn.Linear(dim_t//2, 6)
        self.fc2_v = nn.Linear(dim_v//2, 6)

    def extract_features_emotion(self, sequence, fc1, fc2):
        x = sequence.unsqueeze_(1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.batchnorm1d(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.dropout(x)
        x = fc1(x.view(self.batch_size, -1))
        x = self.relu(x)
        x = fc2(x.view(self.batch_size, -1))
        x = self.sigmoid(x)
        return x

    def forward(self, a, t, v):
        hid_a = self.extract_features_emotion(a, self.fc1_a, self.fc2_a)
        hid_t = self.extract_features_emotion(t, self.fc1_t, self.fc2_t)
        hid_v = self.extract_features_emotion(v, self.fc1_v, self.fc2_v)
        h = torch.stack((hid_a, hid_t, hid_v), dim=0)
        h = torch.mean(h, dim=0)
        return h


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    # IMPORTING PARAMETERS
    params_model = parameters_multimodal.params_CNN
    params_dataset = parameters_multimodal.params_dataset

    # MODEL
    if params_dataset['task'] == 'sentiment':
        net = SentConvNet(device = device, **params_model)
    if params_dataset['task'] == 'sentiment_binary':
        net = SentBiConvNet(device = device, **params_model)
    if params_dataset['task'] == 'emotion':
        net = EmoConvNet(device = device, **params_model)
    print(net)

    print(params_model)
    print(params_dataset)
    batch_size = params_model['batch_size']
    training_generator, validation_generator , _ = \
    get_data_generators(batch_size = batch_size, **params_dataset)
    data_iter = iter(training_generator)
    features, labels = data_iter.next()
    print('features', len(features))
    print('features[0] acoustic', features[0].shape)
    print('dtype', features[0].type())
    print('features[1] text', features[1].shape)
    print('dtype', features[1].type())
    print('features[2] visual', features[2].shape)
    print('dtype', features[2].type())
    print('labels', labels.shape)
    print('dtype', labels.type())
    print(labels)
