#!/usr/bin/env python3
from datetime import datetime
import sys
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np
import re
import pickle
from matplotlib import pyplot as plt
from sklearn import svm, preprocessing, metrics, model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skmultilearn.problem_transform import BinaryRelevance
from tqdm import tqdm
from dataset_multimodal import Data


def collate(X):
    for i in range(len(X)):
        if X[i] > 0: X[i] = 1
        else: X[i] = 0
    return X


def preprocess_emotion(dataset):
    y = []
    for index in range(len(dataset.happiness)):
        y.append(np.array(collate([dataset.happiness[index], dataset.sadness[index], dataset.anger[index],
                           dataset.surprise[index], dataset.disgust[index], dataset.fear[index]])))
        #y.extend(np.array([1 if v > 0 else 0 for v in emotion]))
    return np.array(y)


def check_paths(PATH_TO_EXTRACTED_FEATURES, TRAIN_FILE, TEST_FILE):
    if not isdir(PATH_TO_EXTRACTED_FEATURES):
        raise FileNotFoundError('{} is not a directory'.format(PATH_TO_EXTRACTED_FEATURES))
    if not isfile(join(PATH_TO_EXTRACTED_FEATURES, TRAIN_FILE)):
        raise FileNotFoundError('{} is not a valid name file'.format(TRAIN_FILE))
    if not isfile(join(PATH_TO_EXTRACTED_FEATURES, TEST_FILE)):
        raise FileNotFoundError('{} is not a valid name file'.format(TEST_FILE))


features = sys.argv[1]  # UCNN_SR_egemaps_superdeep_w
task = features[5:7]
if task == 'SB':
    task = features[5:8]
#mode = sys.argv[2]  # [save|load|test]
#name_model = sys.argv[3]  # 'my_rf_model.m'

PATH_TO_EXTRACTED_FEATURES = '../s1872685_daniel_mora/extracted_features/'
TRAIN_FILE = 'train_'+features+'.p'
TEST_FILE = 'test_'+features+'.p'

print('checking files and paths')
check_paths(PATH_TO_EXTRACTED_FEATURES, TRAIN_FILE, TEST_FILE)

if task == 'SR':

    print('=============RANDOM FOREST REGRESSOR WITH EXTRACTED FEATURES=============')

    print('loading data:\n{}\t{}'.format(TRAIN_FILE, TEST_FILE))
    dataset_train = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TRAIN_FILE), 'rb'))
    print(dataset_train)
    X_train = dataset_train.features
    y_train = dataset_train.polarities

    dataset_test = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TEST_FILE), 'rb'))
    X_test = dataset_test.features
    y_test = dataset_test.polarities
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    print(len(dataset_train.name_features))

    with open('rf_cnn_features.log', 'a') as f:
        f.write('\n\n========== BEGGINING RUN SR EXTRACTED FEATURES=========\n\n')
        f.write('features from model: {}\n'.format(features))
        f.write('len x_train: {}\t len x_test: {}\n'.format(len(X_train), len(X_test)))

    print('loaded {} data points'.format(len(X_train)))

    print('normalizing data')
    X_train = preprocessing.normalize(X_train, norm = 'l2')
    X_test = preprocessing.normalize(X_test, norm = 'l2')

    randomforest = RandomForestRegressor(random_state = 19, n_jobs=16, n_estimators=1400,
    min_samples_split=5, min_samples_leaf=2, max_features='auto', max_depth=60, bootstrap=True)

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'parameters: {randomforest}\n')

    print('fitting model')
    randomforest.fit(X_train, y_train)

    print('predicting on test data')
    y_pred = randomforest.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    exv = metrics.explained_variance_score(y_test, y_pred)

    print('mean squared error', mse)
    print('r2 score', r2)
    print('mae', mae)
    print('exv', exv)

    #print('len predictions', len(y_pred))
    #df = pd.DataFrame(y_pred)
    #df.to_csv(join(features, f'{name_model[:-2]}_predictions.csv'))

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'mean squared error: {mse}\n')
        f.write(f'r2 score: {r2}\n')
        f.write(f'mean absolute error: {mae}\n')
        f.write(f'explained variance score: {exv}\n')
        f.write('\n\n========== END OF RUN =========\n\n')

############################
if task == 'SB2':

    print('=============RANDOM FOREST SENTIMENT BINARY WITH EXTRACTED FEATURES=============')

    print('loading data:\n{}\t{}'.format(TRAIN_FILE, TEST_FILE))
    dataset_train = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TRAIN_FILE), 'rb'))
    print(dataset_train)
    X_train = dataset_train.features
    y_train = dataset_train.polarities

    dataset_test = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TEST_FILE), 'rb'))
    X_test = dataset_test.features
    y_test = dataset_test.polarities
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    print(len(dataset_train.name_features))

    with open('rf_cnn_features.log', 'a') as f:
        f.write('\n\n========== BEGGINING RUN SB2 EXTRACTED FEATURES=========\n\n')
        f.write('features from model: {}\n'.format(features))
        f.write('len x_train: {}\t len x_test: {}\n'.format(len(X_train), len(X_test)))

    print('loaded {} data points'.format(len(X_train)))

    print('normalizing data')
    X_train = preprocessing.normalize(X_train, norm = 'l2')
    X_test = preprocessing.normalize(X_test, norm = 'l2')

    y_train = [1.0 if value > 0.0 else 0.0 for value in y_train]
    y_test = [1.0 if value > 0.0 else 0.0 for value in y_test]

    randomforest = RandomForestClassifier(random_state = 19, n_jobs=16, n_estimators=1200,
    min_samples_split=5, min_samples_leaf=4, max_features='auto', max_depth=40, bootstrap=True)

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'parameters: {randomforest}\n')

    print('fitting model')
    randomforest.fit(X_train, y_train)

    print('predicting on test data')
    y_pred = randomforest.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    bacc = metrics.balanced_accuracy_score(y_test, y_pred)
    acc_not_normalize = metrics.accuracy_score(y_test, y_pred, normalize=False)
    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    print('acc', acc)
    print('bacc', bacc)
    print('acc_not_normalize', acc_not_normalize)
    print('f1_macro', f1_macro)
    print('f1_micro', f1_micro)
    print('f1_weighted', f1_weighted)

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'accuracy: {acc}\n')
        f.write(f'balanced accuracy: {bacc}\n')
        f.write(f'accuracy not normalized: {acc_not_normalize}\n')
        f.write(f'f1 macro: {f1_macro}\n')
        f.write(f'f1 micro: {f1_micro}\n')
        f.write(f'f1 weighted: {f1_weighted}\n')
        f.write('\n\n========== END OF RUN =========\n\n')

############################
if task == 'SB3':

    print('=============RANDOM FOREST SENTIMENT TRINARY WITH EXTRACTED FEATURES=============')

    print('loading data:\n{}\t{}'.format(TRAIN_FILE, TEST_FILE))
    dataset_train = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TRAIN_FILE), 'rb'))
    print(dataset_train)
    X_train = dataset_train.features
    y_train = dataset_train.polarities

    dataset_test = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TEST_FILE), 'rb'))
    X_test = dataset_test.features
    y_test = dataset_test.polarities
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    print(len(dataset_train.name_features))

    y_train = [1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0 for value in y_train]
    y_test = [1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0 for value in y_test]

    with open('rf_cnn_features.log', 'a') as f:
        f.write('\n\n========== BEGGINING RUN SB3 EXTRACTED FEATURES=========\n\n')
        f.write('features from model: {}\n'.format(features))
        f.write('len x_train: {}\t len x_test: {}\n'.format(len(X_train), len(X_test)))

    print('normalizing data')
    X_train = preprocessing.normalize(X_train, norm = 'l2')
    X_test = preprocessing.normalize(X_test, norm = 'l2')

    randomforest = RandomForestClassifier(random_state = 19, n_jobs=16, n_estimators=1200,
    min_samples_split=5, min_samples_leaf=4, max_features='auto', max_depth=40, bootstrap=True)

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'parameters: {randomforest}\n')

    print('fitting model')
    randomforest.fit(X_train, y_train)

    y_pred = randomforest.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    bacc = metrics.balanced_accuracy_score(y_test, y_pred)
    acc_not_normalize = metrics.accuracy_score(y_test, y_pred, normalize=False)
    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)

    print('acc', acc)
    print('bacc', bacc)
    print('acc_not_normalize', acc_not_normalize)
    print('f1_macro', f1_macro)
    print('f1_micro', f1_micro)
    print('f1_weighted', f1_weighted)

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'accuracy: {acc}\n')
        f.write(f'balanced accuracy: {bacc}\n')
        f.write(f'accuracy not normalized: {acc_not_normalize}\n')
        f.write(f'f1 macro: {f1_macro}\n')
        f.write(f'f1 micro: {f1_micro}\n')
        f.write(f'f1 weighted: {f1_weighted}\n')
        f.write('\n\n========== END OF RUN =========\n\n')


    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = metrics.plot_confusion_matrix(randomforest, X_test, y_test,
                                     display_labels=['negative', 'neutral', 'positive'],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        print(f'cm_{features}_{title[0]}.pdf')
        plt.savefig(f'cm_figs_cnn_rf/cm_{features}_{title[0]}.pdf')

#########################
if task == 'EM':

    print('=============RANDOM FOREST EMOTION WITH EXTRACTED FEATURES=============')

    print('loading data:\n{}\t{}'.format(TRAIN_FILE, TEST_FILE))
    dataset_train = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TRAIN_FILE), 'rb'))
    print(dataset_train)
    X_train = np.array(dataset_train.features)
    y_train = preprocess_emotion(dataset_train)

    dataset_test = pickle.load(open(join(PATH_TO_EXTRACTED_FEATURES, TEST_FILE), 'rb'))
    X_test = np.array(dataset_test.features)
    y_test = preprocess_emotion(dataset_test)
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    print(len(dataset_train.name_features))
    print(dataset_train.name_features)

    with open('rf_cnn_features.log', 'a') as f:
        f.write('\n\n========== BEGGINING RUN EM EXTRACTED FEATURES=========\n\n')
        f.write('features from model: {}\n'.format(features))
        f.write('len x_train: {}\t len x_test: {}\n'.format(len(X_train), len(X_test)))

    print('normalizing data')
    X_train = preprocessing.normalize(X_train, norm = 'l2')
    X_test = preprocessing.normalize(X_test, norm = 'l2')

    print('building model')
    randomforest = BinaryRelevance(RandomForestClassifier(random_state = 19, n_jobs=16,
                   n_estimators = 300, max_features = 'sqrt', max_depth = 100,
                   min_samples_split = 5, min_samples_leaf = 2, bootstrap=True))

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'parameters: {randomforest}\n')

    print('fitting model')
    randomforest.fit(X_train, y_train)

    print('predicting on test data')
    y_pred = randomforest.predict(X_test)
    y_pred_array = y_pred.toarray()

    acc_fraction = metrics.accuracy_score(y_test, y_pred)
    acc_number = metrics.accuracy_score(y_test, y_pred, normalize=False)

    f1_micro = metrics.f1_score(y_test, y_pred, zero_division = 0, average = 'micro')
    f1_macro = metrics.f1_score(y_test, y_pred, zero_division = 0, average = 'macro')
    f1_samples = metrics.f1_score(y_test, y_pred, zero_division = 0, average = 'samples')
    f1_weighted = metrics.f1_score(y_test, y_pred, zero_division = 0, average = 'weighted')
    f1_none = metrics.f1_score(y_test, y_pred, zero_division = 0, average = None)
    # F1 = 2 * (precision * recall) / (precision + recall)
    roc_micro = metrics.roc_auc_score(y_test, y_pred_array, average = 'micro')
    roc_macro = metrics.roc_auc_score(y_test, y_pred_array, average = 'macro')
    roc_weighted = metrics.roc_auc_score(y_test, y_pred_array, average = 'weighted')
    roc_none = metrics.roc_auc_score(y_test, y_pred_array, average = None)

    cm_class = metrics.multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred, samplewise=False)
    cm_sample = metrics.multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred, samplewise=True)

    print('accuracy (fraction of samples correctly classified):', acc_fraction)
    print('accuracy (number of sampels correctly classified):', acc_number)

    print('f1 micro:', f1_micro)
    print('f1 macro:', f1_macro)
    print('f1 samples:', f1_samples)
    print('f1 weighted:', f1_weighted)
    print('f1 none:', f1_none)

    print('roc micro:', roc_micro)
    print('roc macro:', roc_macro)
    print('roc weighted:', roc_weighted)
    print('roc none:', roc_none)

    print('confusion matrix per class:\n', cm_class)
    print('shape cm_class:', cm_class.shape)
    print('confusion matrix per sample: (excerpt)\n', cm_sample[10])
    print('shape cm_sample:', cm_sample.shape)

    with open('rf_cnn_features.log', 'a') as f:
        f.write(f'accuracy (fraction of samples correctly classified): {acc_fraction}\n')
        f.write(f'accuracy (number of sampels correctly classified): {acc_number}\n')

        f.write(f'f1 micro: {f1_micro}\n')
        f.write(f'f1 macro: {f1_macro}\n')
        f.write(f'f1 samples: {f1_samples}\n')
        f.write(f'f1 weighted: {f1_weighted}\n')
        f.write(f'f1 none: (excerpt)\n{f1_none}\n')

        f.write(f'roc micro: {roc_micro}\n')
        f.write(f'roc macro: {roc_macro}\n')
        f.write(f'roc weighted: {roc_weighted}\n')
        f.write(f'roc none: (excerpt)\n{roc_none}\n')

        f.write(f'confusion matrix per class:\n{cm_class}\n')
        f.write(f'shape cm_class:{cm_class.shape}\n')
        f.write(f'confusion matrix per sample: (excerpt)\n{cm_sample[10]}\n')
        f.write(f'shape cm_sample: {cm_sample.shape}\n')
        f.write('\n\n========== END OF RUN =========\n\n')
