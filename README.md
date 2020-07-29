# Multimodal CNN
Convolutional Neural Network for Multimodal Sentiment Analysis and Emotion Recognition on CMU-MOSEI dataset.
(in progress)

Clone the repo.
Download the data (h5 files) from [here](https://drive.google.com/file/d/1NHYYeNDV9Fk_WZ-XzhWYQguIGbxM3JpD/view?usp=sharing) and unzip it. The folder `data_h5` should contain egemaps_c.h5, openface_c.h5, bert_c.h5.
Ready to go.

Specify hyperparameters in `parameters_multimodal.py` and run `./train_model_multimodal.py`

## Files

parameters_multimodal.py --> parameters of the dataset and model, and task ('sentiment', 'sentiment_binary' or 'emotion').

dataset_multimodal.py --> DataLoader class to load h5 files

convnet_multimodal.py --> CNN architectures: 
* SentConvNet for sentiment (regression) prediction
* EmoConvNet for emotion recognition
* SentBiConvNet for sentiment (binary) classification
