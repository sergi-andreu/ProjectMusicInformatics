## /emotion recognition/

This folder consists on notebooks for predicting the emotions from musical samples in the emotify dataset [the song samples](../data/emotifymusic/).

Here one can find *attempts* of tackling this problem using different approaches. The files are named as ***{Features used}_{Method Used}.ipynb***.

- ***MFCCS_CNN*** uses the Mel Frequency Cepstral Coefficients of the song samples from [MFCCs](../preprocessing/MFCCs) and uses "simple" Convolutional Neural Networks (CNN) models to predict the emotion labels, using *Tensorflow*.
- ***MusiCNN-pool5_CNN.ipynb*** uses the [*pool5* features](../preprocessing/) (intermediate outputs from the *MusiCNN* network) of the Emotify song samples to predict the emotion labels, using *Tensorflow*.
- ***SimpleFeatures_CNN*** uses the [simple features](../preprocessing/) to predict the emotion labels, using a "simple" CNN using *Tensorflow*.
- ***SimpleFeatures_nuSVR*** uses a nu-Support Vector Regression (*nuSVR*) from *sklearn* to predict the emotion labels from the [simple features](../preprocessing/) of the song samples. The training was too computationally expensive.
- ***SimpleFeatures_SVC_majorityVote*** uses a Support Vector Classifier to *classify* the music samples on emotion, using [simple features](../preprocessing/), with a majority vote.
- ***SimpleFeatures_SVC*** the same as before, but without a majority vote.
