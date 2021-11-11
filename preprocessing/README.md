## /preprocessing/

This folder consists on notebooks for extracting the labels from the annotations, and different features from [the song samples](../data/emotifymusic/), as well as some of the extracted features. Most of the files with the extracted results are not saved in the github, even if listed here:

- ***build_Labels*** transforms the annotations of the songs into labels. The results are saved in ***labels.csv***
- ***build_WeightedLabels*** is a failed attempt. No more information is given about this file. We like secrecy.

- ***extract_clicks*** extracts the beat times with librosa. The results are saved at ***clicks/***.
- ***extract_info*** extracts information of the song samples (duration, ...). The results are saved at ***../data/***
- ***extract_MFCCs*** extracts the Mel Frequency Cepstral Coefficients of the song samples. The results are saved at ***MFCCs***
- ***extract_MusiCNN*** extracts features using MusiCNN. The results are saved at ***MusiCNNFeatures***
- ***extract_SimpleFeatures*** extracts simple features. The results are saved at ***SimpleFeatures***
- ***extract_STFT*** extracts the Short-time Fourier transform of the audio samples. 

The folders SimpleFeatures and MusicnnFeatures contain the simple features (extracted zero crossings, spectral centroid, spectral variance, static tempo estimate) and extracted musicnn features respectivey. Data files of too large size are not uploaded.

### Information


The data may not be saved in the appropiate subfolder when running the *extract_* notebooks. Once the notebook is ran, the user may have to move manually the extracted data to the correspondent subfolder, or slighly modify the line of code saving the numpy arrays.

The relative paths of the features have been modified. The user may encounter problems when loading data from this folder because of this. We recommend comming back to this folder, checking where the arrays are, and change the relative path when needed.
