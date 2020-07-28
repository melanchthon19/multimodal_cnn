# multimodal_cnn
Convolutional Neural Network for Sentiment Analysis and Emotion Recognition on CMU-MOSEI dataset.

Download the repo.
Download the data (h5 files) from here https://drive.google.com/file/d/1NHYYeNDV9Fk_WZ-XzhWYQguIGbxM3JpD/view?usp=sharing.
Unzip data_h5.
Ready to go.

./train_model_multimodal.py

Files:

parameters_multimodal.py --> parameters of the dataset and model

dataset_multimodal.py --> DataLoader class to load h5 files

convnet_multimodal.py --> CNN architectures: 
* SentConvNet for sentiment (regression) prediction
* EmoConvNet for emotion recognition
* SentBiConvNet for sentiment (binary) classification
