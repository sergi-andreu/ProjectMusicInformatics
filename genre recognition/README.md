## /genre recognition/

This folder consists on notebooks for predicting the musical genres (*classical*, *electronic*, *pop*, *rock*) from musical samples in the emotify dataset [the song samples](../data/emotifymusic/).

Here one can find *attempts* of tackling this problem using different approaches. The files are named as ***{Features used}_{Method Used}.ipynb***.

- ***Clicks_CNN*** uses the estimated beat times from librosa ([Clicks](../preprocessing/)) and uses "simple" Convolutional Neural Networks (CNN) models to predict the emotion labels, using *Tensorflow*.
- ***MFCCS_CNN*** uses the Mel Frequency Cepstral Coefficients of the song samples from [MFCCs](../preprocessing/MFCCs) and uses "simple" Convolutional Neural Networks (CNN) models to predict the genre of the song, using *Tensorflow*.
- ***MusiCNN-pool5_CNN.ipynb*** uses the [*pool5* features](../preprocessing/) (intermediate outputs from the *MusiCNN* network) of the Emotify song samples to predict the genre of the song samples, using *Tensorflow*.
- ***MusiCNN-taggram_CNN.ipynb*** uses the [*taggram* features](../preprocessing/) (final outputs from the *MusiCNN* network) of the Emotify song samples to predict the genre of the song samples, using *Tensorflow*.
- ***SimpleFeatures_CNN*** uses the [simple features](../preprocessing/) to predict the genres, using a "simple" CNN using *Tensorflow*.
- ***SimpleFeatures_kNN*** uses k nearest neighbors from *sklearn* to predict the genre of the song samples from the [simple features](../preprocessing/) of the song samples. The training was too computationally expensive.
- ***SimpleFeatures_SVC*** uses a Support Vector Classifier to predict the genre of the music samples, using [simple features](../preprocessing/).
