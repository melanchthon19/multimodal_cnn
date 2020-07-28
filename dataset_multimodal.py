#!/usr/bin/env python3
import sys
import os
from os.path import join
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import h5py


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, features, tiny=False):
        'Initialization'
        self.labels = labels
        if tiny:
            self.list_IDs = list_IDs[:100]
        else:
            self.list_IDs = list_IDs
        self.features = features

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        PATH_H5 = '/data_h5/'
        multimodal = []
        for feat in self.features:  # acoustic, text, visual
            dset = h5py.File(join(PATH_H5, feat), 'r')
            X = dset.get(ID)
            X = torch.FloatTensor(X['features'])
            multimodal.append(X)
            dset.close()
        # multimodal --> [acoustic, text, visual]

        y = torch.FloatTensor(np.array(self.labels[ID]))

        return multimodal, y


def collate_multimodal(batch):  # not being used
    acoustic = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], batch_first = True)
    text = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], batch_first = True)
    visual = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], batch_first = True)

    labels = torch.stack([torch.FloatTensor(torch.from_numpy(np.array(sample[1])) for sample in batch)], dim=0)
    multimodal = (acoustic, text, visual)
    return multimodal, labels


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
    video_id: [0.0,1.0] if df_set.loc[video_id, 'polarity'] > 0.0 else [1.0,0.0] for video_id in df_set.index.values.tolist()
    }
    return binary_labels


def get_data_generators(batch_size, features, task, tiny=False):

    features = [feat + '.h5' for feat in features]
    # Reading Training Set
    df_train = pd.read_csv('data/df_train_MOSEI.tsv')
    df_train.set_index('Unnamed: 0', inplace = True)
    # Reading Testing Set
    df_test = pd.read_csv('data/df_test_MOSEI.tsv')
    df_test.set_index('Unnamed: 0', inplace = True)
    # Reading Validation Set
    df_valid = pd.read_csv('data/df_valid_MOSEI.tsv')
    df_valid.set_index('Unnamed: 0', inplace = True)

    # Parameters DataLoader
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2,
              #'collate_fn': collate_multimodal,
              'drop_last': True
              }

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

    # Generators
    print('training Dataset')
    training_set = Dataset(partition['train'], labels_training, features, tiny)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    print('validation Dataset')
    validation_set = Dataset(partition['validation'], labels_valid, features, tiny)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    print('test Dataset')
    test_set = Dataset(partition['test'], labels_test, features, tiny)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    return training_generator, validation_generator, test_generator


if __name__ == '__main__':

    batch_size = 5
    task = 'sentiment'
    tiny = False

    features = ['egemaps_c', 'bert_c','openface_c']

    training_generator, validation_generator, test_generator = \
        get_data_generators(batch_size=batch_size, features=features, task=task, tiny=tiny)
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
