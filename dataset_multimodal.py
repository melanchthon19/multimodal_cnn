#!/usr/bin/env python3
import sys
import os
from os.path import join
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import h5py
import pickle
from functools import partial
from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(self):
        self.polarities = []

        self.happiness = []
        self.sadness = []
        self.anger = []
        self.surprise = []
        self.disgust = []
        self.fear = []

        self.features = []
        self.name_features = []
        self.data = (self.polarities, self.features)
        self.labels = [self.polarities,
                      self.happiness,
                      self.sadness,
                      self.anger,
                      self.surprise,
                      self.disgust,
                      self.fear]


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, features, selected_features, scalers, tiny):
        'Initialization'
        self.labels = labels
        if tiny:
            self.list_IDs = list_IDs[:256]
        else:
            self.list_IDs = list_IDs
        self.features = features
        self.selected_features = selected_features
        self.scalers = scalers

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        PATH_H5 = '/group/project/cstr1/mscslp/2019-20/s1872685_daniel_mora/data_h5/'
        multimodal = []
        for feat in self.features:  # acoustic, text, visual
            dset = h5py.File(join(PATH_H5, feat), 'r')
            X = dset.get(ID)
            #print(self.selected_features)
            #print(self.selected_features[feat[:-3]])
            if self.selected_features[feat[:-3]] != ['all']:
                X = pd.DataFrame(X['features'])
                if feat[-4] != 'c':
                    X = np.array(X.loc[ : , self.selected_features[feat[:-3]]]).squeeze()
                else:
                    X = np.array(X.loc[self.selected_features[feat[:-3]], : ]).squeeze()
                X = torch.FloatTensor(X)
            else:
                X = torch.FloatTensor(X['features'])

            # normalizing
            try:  # when features collapsed (timesteps == 1)
                self.scalers[feat[:-3]].transform(X.reshape(1,-1))
            except ValueError:  # when features with timesteps > 1
                self.scalers[feat[:-3]].transform(X)

            multimodal.append(X)
            dset.close()
        # multimodal --> [acoustic, text, visual]
        y = torch.FloatTensor(np.array(self.labels[ID]))

        return multimodal, y


class DatasetAligned2Word(torch.utils.data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, features, selected_features, n_words, tiny):
        'Initialization'
        self.labels = labels
        if tiny:
            self.list_IDs = list_IDs[:128]
        else:
            self.list_IDs = list_IDs
        self.features = features
        self.selected_features = selected_features
        self.n_words = n_words

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        PATH_H5 = '/group/project/cstr1/mscslp/2019-20/s1872685_daniel_mora/data_h5/'
        multimodal = []
        for feat in self.features:  # acoustic, text, visual
            dset = h5py.File(join(PATH_H5, feat), 'r')
            X = dset.get(ID)
            if self.selected_features[feat[:-3]] != ['all']:
                X = pd.DataFrame(X['features'])
                X = np.array(X.loc[ : ,self.selected_features[feat[:-3]]])
                X = torch.FloatTensor(X)
            else:
                X = torch.FloatTensor(X['features'])
            X = self.average(X, self.n_words[ID])

            multimodal.append(X)
            dset.close()
        # multimodal --> [acoustic, text, visual]

        y = torch.FloatTensor(np.array(self.labels[ID]))

        return multimodal, y

    def average(self, X, n):
        if X.shape[0] < n:
            return X
        n = int(n)
        chunk = X.shape[0] // n
        chunks = torch.split(X, chunk, 0)
        X = torch.stack([torch.mean(chunk, 0) for chunk in chunks])
        return X


class Dataset_Bert(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, n):
        'Initialization'
        self.data = data['Data'][n[0]:n[1]]
        self.labels = data['level'][n[0]:n[1]]
        self.ini = n[0]
        self.end = n[1]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        y = self.labels[index]
        return X, y


def collate_seq(batch):
    sequence = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], batch_first = True)
    labels = torch.stack([sample[1] for sample in batch], dim=0)
    return sequence, labels


def collate_multimodal(batch):
    acoustic = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], batch_first = True)
    text = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], batch_first = True)
    visual = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], batch_first = True)

    #labels = torch.stack([torch.FloatTensor(torch.from_numpy(np.array(sample[1])) for sample in batch)], dim=0)
    labels = torch.stack([sample[1] for sample in batch], dim=0)
    multimodal = (acoustic, text, visual)
    return multimodal, labels


def collate_seq_aligned(max_len, batch):
    padded = False
    for sample in batch:
        if sample[0][0].shape[0] > max_len:
            padded = True
            sample[0][0] = sample[0][0][:max_len]
    if not padded:
        to_pad = max_len - batch[0][0][0].shape[0]
        h = batch[0][0][0].shape[1]
        pad = torch.zeros(to_pad, h)
        batch[0][0][0] = torch.cat((batch[0][0][0], pad), 0)

    sequence = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], batch_first = True)
    labels = torch.stack([sample[1] for sample in batch], dim=0)
    return sequence, labels

def pad_to_max_len(X, max_len):
    #print('pad to max len')
    #print(X.shape)
    to_pad = max_len - X.shape[0]
    h = X.shape[1]
    pad = torch.zeros(to_pad, h)
    X = torch.cat((X, pad), 0)
    #print('returning')
    #print(X.shape)
    return X

def collate_seq_aligned_frame(max_len, batch):
    #print('collate')
    for sample in range(len(batch)):
        #print('entering', sample)
        #print(batch[sample][0][0].shape)
        if len(batch[sample][0][0].shape) == 1:
            batch[sample][0][0].unsqueeze_(0)
        batch[sample][0][0] = average_to_maxlen(batch[sample][0][0], max_len)
        if batch[sample][0][0].shape[0] < max_len:
            #print(batch[0][0][0].shape)
            batch[sample][0][0] = pad_to_max_len(batch[sample][0][0], max_len)
            #print(batch[0][0][0].shape)
    #for sample in range(len(batch)):
    #    print(batch[sample][0][0].shape)
    sequence = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], batch_first = True)
    labels = torch.stack([sample[1] for sample in batch], dim=0)
    return sequence, labels


def collate_multimodal_aligned(max_len, batch):
    for sample in range(len(batch)):
        for modality in range(len(batch[sample][0])):  # traversing through modalities
            if batch[sample][0][modality].shape[0] > max_len:
                batch[sample][0][modality] = batch[sample][0][modality][:max_len]
            else:
                to_pad = max_len - batch[sample][0][modality].shape[0]
                h = batch[sample][0][modality].shape[1]
                pad = torch.zeros(to_pad, h)
                batch[sample][0][modality] = torch.cat((batch[sample][0][modality], pad), 0)

    acoustic = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], batch_first = True)
    text = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], batch_first = True)
    visual = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], batch_first = True)

    labels = torch.stack([sample[1] for sample in batch], dim=0)
    multimodal = (acoustic, text, visual)
    return multimodal, labels


def average_to_maxlen(X, max_len):
    if X.shape[0] < max_len:
        return X
    chunks = torch.chunk(X, max_len, 0)  # torch.chunk splits the tensor in max_len chunks
    X = torch.stack([torch.mean(chunk, 0) for chunk in chunks])  # averaging each chunk
    # shape of X is <= max_len
    return X


def emotion_labels(df_set):
    emotion_labels = {
    video_id:[df_set.loc[video_id, 'happiness'],
              df_set.loc[video_id, 'sadness'],
              df_set.loc[video_id, 'anger'],
              df_set.loc[video_id, 'surprise'],
              df_set.loc[video_id, 'disgust'],
              df_set.loc[video_id, 'fear'],
              ] for video_id in df_set.index.values.tolist()}
    preprocess_emotion(emotion_labels)
    return emotion_labels


def preprocess_emotion(video_id):
    # change all values greater than 0.0 to 1.0
    for video in video_id:
        for index in range(6):
            if video_id[video][index] > 0.0: video_id[video][index] = 1.0


def sentiment_binary_labels(df_set):
    binary_labels = {
    #video_id: [0.0,1.0] if df_set.loc[video_id, 'polarity'] > 0.0 else [1.0,0.0] for video_id in df_set.index.values.tolist()
    video_id: 1.0 if df_set.loc[video_id, 'polarity'] > 0.0 else 0.0 for video_id in df_set.index.values.tolist()
    }
    # positive (> 0.0) and negative (<= 0.0) are balanced
    return binary_labels


def sentiment_trinary_labels(df_set):
    binary_labels = {
    video_id: 2.0 if df_set.loc[video_id, 'polarity'] > 0.0 \
            else 0.0 if df_set.loc[video_id, 'polarity'] < 0.0 \
            else 1.0 for video_id in df_set.index.values.tolist()
    }
    # positive (> 0.0) and negative (<= 0.0) are balanced
    return binary_labels


def drop_videos(set, to_drop):
    to_drop.dropna(inplace=True)
    for video_id in to_drop:
        set.drop(video_id, inplace=True)


def select_features(data, index_selec_feat):
    df = pd.DataFrame.from_records(data)
    df = df.loc[:,index_selec_feat]
    data = df.values.tolist()
    return data


def get_scalers(selected_features):
    print(selected_features)
    scalers = {}
    for features in selected_features:
        name_p_file = features
        scaler = StandardScaler()
        if features[-1] == 'c': name_p_file = features[:-2]
        path_data_train = f'../s1872685_daniel_mora/data_p/train_{name_p_file}_mosei_sklearn.p'
        data_train = pickle.load(open(path_data_train, 'rb'))
        print(len(data_train.features))
        print(len(data_train.features[0]))
        if selected_features[features] != ['all']:
            print('selecting features:', features)
            data_train.features = select_features(data_train.features, selected_features[features])
        scaler.fit(data_train.features)
        scalers[features] = scaler
    return scalers


def get_data_generators(batch_size, task, modality, tiny, aligned2word, balance_polarity, selected_features, max_len):

    features = [feat + '.h5' for feat in selected_features.keys()]
    # Reading Training Set
    df_train = pd.read_csv('data/df_train_MOSEI.tsv')
    df_train.set_index('Unnamed: 0', inplace = True)
    # Reading Testing Set
    df_test = pd.read_csv('data/df_test_MOSEI.tsv')
    df_test.set_index('Unnamed: 0', inplace = True)
    # Reading Validation Set
    df_valid = pd.read_csv('data/df_valid_MOSEI.tsv')
    df_valid.set_index('Unnamed: 0', inplace = True)

    print(df_train.shape)
    if balance_polarity:
        print('dropping video ids to balance dataset')
        to_drop = pd.read_csv('data/to_balance_mosei.csv')
        drop_videos(df_train, to_drop['train'])
        #drop_videos(df_valid, to_drop['valid'])
        #drop_videos(df_test, to_drop['test'])
    print(df_train.shape)

    # Parameters DataLoader
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2,
              'drop_last': True
              }

    if modality == 'unimodal':
        if aligned2word:
            #params['collate_fn'] = partial(collate_seq_aligned, max_len)
            #print('collate_fn: collate_seq_aligned')
            params['collate_fn'] = partial(collate_seq_aligned_frame, max_len)
            print('collate_fn: collate_seq_aligned_frame')
        else:
            params['collate_fn'] = collate_seq
            print('collate_fn: collate_seq')
    else:  # task == 'multimodal'
        if aligned2word:
            params['collate_fn'] = partial(collate_multimodal_aligned, max_len)
            print('collate_fn: collate_ multimodal_aligned')
        else:
            params['collate_fn'] = collate_multimodal
            print('collate_fn: collate_multimodal')

    print('parameters DataLoader')
    for k,v in params.items(): print('\t',k,v)
    # Datasets
    partition = {'train':df_train.index.values.tolist(),
                'validation':df_valid.index.values.tolist(),
                'test':df_test.index.values.tolist()}

    # Labels per task
    if task == 'emotion':
        labels_training = emotion_labels(df_train)
        labels_test = emotion_labels(df_test)
        labels_valid = emotion_labels(df_valid)

    if task == 'sentiment':
        labels_training = {video_id:df_train.loc[video_id, 'polarity'] for video_id in df_train.index.values.tolist()}
        labels_test = {video_id:df_test.loc[video_id, 'polarity'] for video_id in df_test.index.values.tolist()}
        labels_valid = {video_id:df_valid.loc[video_id, 'polarity'] for video_id in df_valid.index.values.tolist()}

    if task == 'sentiment_binary':
        labels_training = sentiment_binary_labels(df_train)
        labels_test = sentiment_binary_labels(df_test)
        labels_valid = sentiment_binary_labels(df_valid)

    if task == 'sentiment_trinary':
        labels_training = sentiment_trinary_labels(df_train)
        labels_test = sentiment_trinary_labels(df_test)
        labels_valid = sentiment_trinary_labels(df_valid)

    # Normalizing data
    scalers = get_scalers(selected_features)
    print('scalers:', scalers)

    # Generators

    training_set = Dataset(partition['train'], labels_training, features, selected_features, scalers, tiny)
    validation_set = Dataset(partition['validation'], labels_valid, features, selected_features, scalers, tiny)
    test_set = Dataset(partition['test'], labels_test, features, selected_features, scalers, tiny)

    print('training Dataset')
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    print('validation Dataset')
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    print('test Dataset')
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    return training_generator, validation_generator, test_generator


if __name__ == '__main__':

    batch_size = 5
    params_dataset = {
    #'features': ['egemaps_c', 'bertwsp_c', 'openface_c'],
    #'features': ['egemaps_c'],
    'task': 'sentiment',  # 'sentiment' | 'emotion' | 'sentiment_binary'
    'tiny': False,
    'balance_polarity': False,
    'aligned2word': False,
    'modality': 'unimodal',  # 'unimodal' | 'multimodal'
    'selected_features': {'egemaps_c':[0,1,4], 'bertwsp_c':[5,8,5], 'openface_c':[10,123,321]}
    }
    training_generator, validation_generator, test_generator = \
        get_data_generators(batch_size=batch_size, **params_dataset)
    print(f'data generators for features (with timeFrames or steps), batch_size = {batch_size}')
    data_iter = iter(training_generator)
    features, labels = data_iter.next()
    print('type features', type(features))
    print('type labels', type(labels))
    print('labels shape', labels.shape)
    print('features len', len(features))
    #print('features[0]', features[0])
    print('features[0]', features[0].shape)
    #print('features[1]', features[1])
    print('features[1]', features[1].shape)
    #print('features[2]', features[2])
    print('features[2]', features[2].shape)
    print('batch size', batch_size)
    for feat in features:
        memory = feat.element_size() * feat.nelement() + labels.element_size() * labels.nelement()
        print('memory (byte) allocated per batch', memory)
