# Deep-Learning-EEG
Neural networks to decode movements from EEG data. Based off Final Project for C147 at UCLA. Results are further discussed in `DL_EEG_Paper.pdf`

Uses labeled EEG data from http://www.bbci.de/competition/iv/. Data is 4s of 22 channel EEG recordings from 9 total subjects who are imagining 1 of 4 movements

Compared convolutional neural network (CNN), recurrent neural network (RNN), and hybrid convolutional recurrent networks (CRNNs).

Achieved best performance with a CNN adapted to EEG data. This CNN was designed with temporal filters to allow it to learn the key information contained in the EEG frequency power spectrum and spatial filters to combine data from multiple electrodes.  This network can decode the chosen movement with approximately 70% accuracy using only 1.2 seconds of data, suggesting possibility for brain-computer interface applications with improvements. 

## Usage
The notebook is designed to be run in Google Colab which can be done simply at https://colab.research.google.com/github/kkapre/Deep-Learning-EEG/blob/main/DL_EEG_Decoding.ipynb. Running the cells will request Google Drive access and will create a new folder in the drive where this repository will be cloned. 

To run locally, Git-LFS must be downloaded and installed before cloning. On Windows see https://git-lfs.github.com/

On Linux

```
sudo apt-get install git-lfs
git lfs install
```

Clone the repository with 
```
git clone https://github.com/kkapre/Deep-Learning-EEG.git
```

Then run `DL_EEG_Decoding.ipynb`, changing any code involving Google Drive. 
