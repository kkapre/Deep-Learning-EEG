# Deep-Learning-EEG
Neural networks to decode movements from EEG data. Final Project for C147 at UCLA.

Designed to be used in Google Colab with a GPU.Uses tensorflow-keras

Applied neural network architectures to decode imagined movements using labeled EEG data from http://www.bbci.de/competition/iv/

Compared convolutional neural network (CNN), recurrent neural network (RNN), and hybrid convolutional recurrent networks (CRNNs).

Achieved best performance with a CNN adapted to EEG data. CNN was designed with temporal filters to allow it to learn the key information contained in the EEG frequency power spectrum and spatial filters to combine data from multiple electrodes.  Can predict with approximately 70% accuracy with only 1.2 seconds of data, suggesting possibility for brain-computer interface applications with improvements. 
