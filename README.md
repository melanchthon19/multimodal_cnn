# Multimodal CNN
Convolutional Neural Network for Multimodal Sentiment Analysis and Emotion Recognition on CMU-MOSEI dataset.

Clone the repo.
Download the data (h5 files) from [here](https://drive.google.com/file/d/1NHYYeNDV9Fk_WZ-XzhWYQguIGbxM3JpD/view?usp=sharing) and unzip it. The folder `data_h5` should contain egemaps_c.h5, openface_c.h5, bert_c.h5.
Ready to go.

Specify hyperparameters in `parameters.py` and run `./train_model_multimodal.py`

## Files

parameters.py --> parameters of the dataset, model, and task ('sentiment', 'sentiment_binary', or 'emotion').

dataset_multimodal.py --> DataLoader class to load h5 files

models.py --> CNN architectures for sentiment regression prediction, binary classification, and emotion recognition.
