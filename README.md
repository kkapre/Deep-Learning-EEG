# Deep-Learning-EEG
Neural networks to decode movements from EEG data. Final Project for C147 at UCLA.

Designed to be used in Google Colab with a GPU. Uses tensorflow-keras

Uses labeled EEG data from http://www.bbci.de/competition/iv/. Data is 4s of 22 channel EEG recordings from 9 total subjects who are imagining 1 of 4 movements

Compared convolutional neural network (CNN), recurrent neural network (RNN), and hybrid convolutional recurrent networks (CRNNs).

Achieved best performance with a CNN adapted to EEG data. CNN was designed with temporal filters to allow it to learn the key information contained in the EEG frequency power spectrum and spatial filters to combine data from multiple electrodes.  This network can decode the chosen movement with approximately 70% accuracy using only 1.2 seconds of data, suggesting possibility for brain-computer interface applications with improvements. 


Note:
This repository uses Git LFS which must be downloaded and installed to use properly
See: https://git-lfs.github.com/

Can be done on Linux with

'sudo apt-get install git-lfs
git lfs install'
