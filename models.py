#!/usr/bin/env python3
import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset_multimodal import get_data_generators
import parameters


class UniCNN_c(nn.Module):
    def __init__(self, batch_size, emb_dim, in_chan, out_chan, task):
        super(UniCNN_c, self).__init__()
        self.batch_size = batch_size
        self.task = task
        self.emb_dim = emb_dim
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1d = nn.Conv1d(self.in_chan, self.out_chan, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.fc_sr = nn.Linear(self.out_chan, 1)
        self.fc_sb2 = nn.Linear(self.out_chan, 2)
        self.fc_sb3 = nn.Linear(self.out_chan, 3)
        self.fc_em = nn.Linear(self.out_chan, 6)

    def forward(self, x):
        x = self.bn1(x)
        x.unsqueeze_(2)
        x = self.relu(self.conv1d(x))
        if self.task == 'sentiment':
            x = self.fc_sr(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_binary':
            x = self.fc_sb2(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_trinary':
            x = self.fc_sb3(x.view(self.batch_size, -1))
        elif self.task == 'emotion':
            x = self.fc_em(x.view(self.batch_size, -1))
            x = self.sigmoid(x)
        return x


class UniCNN_w(nn.Module):
    def __init__(self, batch_size, emb_dim, in_chan, out_chan, conv1, max_pool, dropout, max_len, task):
        super(UniCNN_w, self).__init__()
        self.batch_size = batch_size
        self.task = task
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1d_1 = nn.Conv1d(self.in_chan, self.out_chan, kernel_size=conv1['kernel'], padding=conv1['padding'], dilation=conv1['dilation'], stride=conv1['stride'])
        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.maxpool = nn.MaxPool1d(max_pool)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        dim1 = (max_len + 2 * conv1['padding'] - conv1['dilation'] * (conv1['kernel'] - 1) - 1) // conv1['stride'] + 1
        dim1 = dim1 // max_pool
        self.fc_collapse = nn.Linear(dim1, 1)
        self.fc_sr = nn.Linear(self.out_chan, 1)
        self.fc_sb2 = nn.Linear(self.out_chan, 2)
        self.fc_sb3 = nn.Linear(self.out_chan, 3)
        self.fc_em = nn.Linear(self.out_chan, 6)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn1(x)
        x = self.relu(self.conv1d_1(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.fc_collapse(x)
        if self.task == 'sentiment':
            x = self.fc_sr(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_binary':
            x = self.fc_sb2(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_trinary':
            x = self.fc_sb3(x.view(self.batch_size, -1))
        elif self.task == 'emotion':
            x = self.fc_em(x.view(self.batch_size, -1))
            x = self.sigmoid(x)
        return x


class UniCNNDeep_w(nn.Module):
    def __init__(self, batch_size, emb_dim, in_chan, out_chan, conv1, conv2, conv3, max_pool, dropout, max_len, task):
        super(UniCNNDeep_w, self).__init__()
        self.batch_size = batch_size
        self.task = task
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1d_1 = nn.Conv1d(self.in_chan, self.out_chan, kernel_size=conv1['kernel'], padding=conv1['padding'], dilation=conv1['dilation'], stride=conv1['stride'])
        self.conv1d_2 = nn.Conv1d(self.out_chan, self.out_chan, kernel_size=conv2['kernel'], padding=conv2['padding'], dilation=conv2['dilation'], stride=conv2['stride'])
        self.conv1d_3 = nn.Conv1d(self.out_chan, self.in_chan, kernel_size=conv3['kernel'], padding=conv3['padding'], dilation=conv3['dilation'], stride=conv3['stride'])
        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.maxpool = nn.MaxPool1d(max_pool)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        conv1_dim = (max_len + 2 * conv1['padding'] - conv1['dilation'] * (conv1['kernel'] - 1) - 1) // conv1['stride'] + 1
        conv1_dim = conv1_dim // max_pool
        conv2_dim = (conv1_dim + 2 * conv2['padding'] - conv2['dilation'] * (conv2['kernel'] - 1) - 1) // conv2['stride'] + 1
        conv2_dim = conv2_dim // max_pool
        conv3_dim = (conv2_dim + 2 * conv3['padding'] - conv3['dilation'] * (conv3['kernel'] - 1) - 1) // conv3['stride'] + 1
        conv3_dim = conv3_dim // max_pool
        self.fc_collapse = nn.Linear(conv3_dim, 1)
        self.fc_sr = nn.Linear(self.in_chan, 1)
        self.fc_sb2 = nn.Linear(self.out_chan, 2)
        self.fc_sb3 = nn.Linear(self.out_chan, 3)
        self.fc_em = nn.Linear(self.out_chan, 6)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn1(x)

        x = self.relu(self.conv1d_1(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.relu(self.conv1d_2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.relu(self.conv1d_3(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.fc_collapse(x)
        if self.task == 'sentiment':
            x = self.fc_sr(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_binary':
            x = self.fc_sb2(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_trinary':
            x = self.fc_sb3(x.view(self.batch_size, -1))
        elif self.task == 'emotion':
            x = self.fc_em(x.view(self.batch_size, -1))
            x = self.sigmoid(x)
        return x


class UniCNNSuperDeep_w(nn.Module):
    def __init__(self, batch_size, emb_dim, in_chan, out_chan, conv1, conv2, conv3, conv4, conv5, conv6, max_pool, dropout, max_len, task):
        super(UniCNNSuperDeep_w, self).__init__()
        self.batch_size = batch_size
        self.task = task
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1d_1 = nn.Conv1d(self.in_chan, self.out_chan, kernel_size=conv1['kernel'], padding=conv1['padding'], dilation=conv1['dilation'], stride=conv1['stride'])
        self.conv1d_2 = nn.Conv1d(self.out_chan, self.out_chan, kernel_size=conv2['kernel'], padding=conv2['padding'], dilation=conv2['dilation'], stride=conv2['stride'])
        self.conv1d_3 = nn.Conv1d(self.out_chan, self.out_chan, kernel_size=conv3['kernel'], padding=conv3['padding'], dilation=conv3['dilation'], stride=conv3['stride'])
        self.conv1d_4 = nn.Conv1d(self.out_chan, self.out_chan, kernel_size=conv4['kernel'], padding=conv4['padding'], dilation=conv4['dilation'], stride=conv4['stride'])
        self.conv1d_5 = nn.Conv1d(self.out_chan, self.out_chan, kernel_size=conv5['kernel'], padding=conv5['padding'], dilation=conv5['dilation'], stride=conv5['stride'])
        self.conv1d_6 = nn.Conv1d(self.out_chan, self.in_chan, kernel_size=conv6['kernel'], padding=conv6['padding'], dilation=conv6['dilation'], stride=conv6['stride'])

        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.maxpool = nn.MaxPool1d(max_pool)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        conv1_dim = (max_len + 2 * conv1['padding'] - conv1['dilation'] * (conv1['kernel'] - 1) - 1) // conv1['stride'] + 1
        conv2_dim = (conv1_dim + 2 * conv2['padding'] - conv2['dilation'] * (conv2['kernel'] - 1) - 1) // conv2['stride'] + 1
        conv3_dim = (conv2_dim + 2 * conv3['padding'] - conv3['dilation'] * (conv3['kernel'] - 1) - 1) // conv3['stride'] + 1
        conv4_dim = (conv3_dim + 2 * conv4['padding'] - conv4['dilation'] * (conv4['kernel'] - 1) - 1) // conv4['stride'] + 1
        conv5_dim = (conv4_dim + 2 * conv5['padding'] - conv5['dilation'] * (conv5['kernel'] - 1) - 1) // conv5['stride'] + 1
        conv6_dim = (conv5_dim + 2 * conv6['padding'] - conv6['dilation'] * (conv6['kernel'] - 1) - 1) // conv6['stride'] + 1

        self.fc_collapse = nn.Linear(conv6_dim//max_pool, 1)
        self.fc_sr = nn.Linear(self.in_chan, 1)
        self.fc_sb2 = nn.Linear(self.in_chan, 2)
        self.fc_sb3 = nn.Linear(self.in_chan, 3)
        self.fc_em = nn.Linear(self.in_chan, 6)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn1(x)

        x = self.relu(self.conv1d_1(x))
        x = self.dropout(x)
        x = self.relu(self.conv1d_2(x))
        x = self.dropout(x)
        x = self.relu(self.conv1d_3(x))
        x = self.dropout(x)
        x = self.relu(self.conv1d_4(x))
        x = self.dropout(x)
        x = self.relu(self.conv1d_5(x))
        x = self.dropout(x)
        x = self.relu(self.conv1d_6(x))

        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.fc_collapse(x)

        if self.task == 'sentiment':
            x = self.fc_sr(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_binary':
            x = self.fc_sb2(x.view(self.batch_size, -1))
        elif self.task == 'sentiment_trinary':
            x = self.fc_sb3(x.view(self.batch_size, -1))
        elif self.task == 'emotion':
            x = self.fc_em(x.view(self.batch_size, -1))
            x = self.sigmoid(x)
        return x



class MultiSRConvNet(nn.Module):
    def __init__(self, batch_size, emb_dims, in_chan_a, in_chan_t, in_chan_v, out_chan_a, out_chan_t, out_chan_v, kernel, padding, max_pool, stride, dropout):
        super(MultiSRConvNet, self).__init__()
        self.batch_size = batch_size
        self.emb_dim_a = emb_dims[0]
        self.emb_dim_t = emb_dims[1]
        self.emb_dim_v = emb_dims[2]
        print(emb_dims[0],emb_dims[1],emb_dims[2])
        self.in_chan_a = in_chan_a
        self.out_chan_a = out_chan_a
        self.in_chan_t = in_chan_t
        self.out_chan_t = out_chan_t
        self.in_chan_v = in_chan_v
        self.out_chan_v = out_chan_v

        self.stride = stride
        self.conv1d_a = nn.Conv1d(self.in_chan_a, self.out_chan_a, kernel_size=kernel, padding=padding)
        self.conv1d_t = nn.Conv1d(self.in_chan_t, self.out_chan_t, kernel_size=kernel, padding=padding)
        self.conv1d_v = nn.Conv1d(self.in_chan_v, self.out_chan_v, kernel_size=kernel, padding=padding)

¡        self.bn1_a = nn.BatchNorm1d(self.in_chan_a)
        self.bn1_t = nn.BatchNorm1d(self.in_chan_t)
        self.bn1_v = nn.BatchNorm1d(self.in_chan_v)

        self.stride = stride
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        dim_a = (self.emb_dim_a - (kernel-1) + (padding*2))
        dim_t = (self.emb_dim_t - (kernel-1) + (padding*2))
        dim_v = (self.emb_dim_v - (kernel-1) + (padding*2))

        # SENTIMENT DENSE LAYERS
        self.fc1_a = nn.Linear(self.out_chan_a, 1)
        self.fc1_t = nn.Linear(self.out_chan_t, 1)
        self.fc1_v = nn.Linear(self.out_chan_v, 1)
        self.ff = nn.Linear(3,1)

    def extract_features_sentiment(self, sequence, fc1, bn1, conv1d, name):
        x = sequence.unsqueeze_(2)
        x = bn1(x)
        x = self.relu(conv1d(x))
        x = self.dropout(x)
        x = fc1(x.view(self.batch_size, -1))
        return x

    def forward(self, a, t, v):
        hid_a = self.extract_features_sentiment(a, self.fc1_a, self.bn1_a, self.conv1d_a, 'audio')
        hid_t = self.extract_features_sentiment(t, self.fc1_t, self.bn1_t, self.conv1d_t, 'text')
        hid_v = self.extract_features_sentiment(v, self.fc1_v, self.bn1_v, self.conv1d_v, 'visual')
        h = torch.cat((hid_a, hid_t, hid_v), dim=1)
        h = self.ff(h)
        return h


class MultiSRConvNetA2W(nn.Module):
    def __init__(self, batch_size, emb_dims, in_chan_a, in_chan_t, in_chan_v, out_chan_a, out_chan_t, out_chan_v, kernel, padding, max_pool, stride, dropout, max_len):
        super(MultiSRConvNetA2W, self).__init__()
        self.batch_size = batch_size
        self.emb_dim_a = emb_dims[0]
        self.emb_dim_t = emb_dims[1]
        self.emb_dim_v = emb_dims[2]
        print(emb_dims[0],emb_dims[1],emb_dims[2])
        self.in_chan_a = in_chan_a
        self.out_chan_a = out_chan_a
        self.in_chan_t = in_chan_t
        self.out_chan_t = out_chan_t
        self.in_chan_v = in_chan_v
        self.out_chan_v = out_chan_v

        self.stride = stride
        self.conv1d_a = nn.Conv1d(self.in_chan_a, self.out_chan_a, kernel_size=kernel, padding=padding)
        self.conv1d_t = nn.Conv1d(self.in_chan_t, self.out_chan_t, kernel_size=kernel, padding=padding)
        self.conv1d_v = nn.Conv1d(self.in_chan_v, self.out_chan_v, kernel_size=kernel, padding=padding)

        self.bn1_a = nn.BatchNorm1d(self.in_chan_a)
        self.bn1_t = nn.BatchNorm1d(self.in_chan_t)
        self.bn1_v = nn.BatchNorm1d(self.in_chan_v)

        self.relu = nn.ReLU()
        dim = (max_len - (kernel-1) + (padding*2))

        # SENTIMENT DENSE LAYERS
        self.fc1_a = nn.Linear(self.out_chan_a*dim, 1)
        self.fc1_t = nn.Linear(self.out_chan_t*dim, 1)
        self.fc1_v = nn.Linear(self.out_chan_v*dim, 1)
        self.ff = nn.Linear(3,1)

    def extract_features_sentiment(self, sequence, fc1, bn1, conv1d, name):
        x = sequence
        x = x.permute(0,2,1)
        x = bn1(x)
        x = self.relu(conv1d(x))
        x = fc1(x.view(self.batch_size, -1))
        return x

    def forward(self, a, t, v):
        hid_a = self.extract_features_sentiment(a, self.fc1_a, self.bn1_a, self.conv1d_a, 'audio')
        hid_t = self.extract_features_sentiment(t, self.fc1_t, self.bn1_t, self.conv1d_t, 'text')
        hid_v = self.extract_features_sentiment(v, self.fc1_v, self.bn1_v, self.conv1d_v, 'visual')
        h = torch.cat((hid_a, hid_t, hid_v), dim=1)
        h = self.ff(h)
        return h


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
