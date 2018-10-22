# Mini_PJ2-Classifier
Boston University
EC601 Mini-Project 2 

Copyright Alex Jeffrey Lin 

TensorFlow Mini-Project
================
Use TensorFlow library (Such as Keras) and model [Such as Convolutional Neural Network(CNN)] to recognize between Cat and Dog. The entire program is running on python 3 notebook on Google Colaboratory. The selected dataset is "Dogs vs. Cats Redux: Kernels Edition" from Kaggle, the link is following: 
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

Preparation
============
before running the program, you should obtain the json file to access Kaggle API first. Next step is to import the dataset from kaggle into the google colab by the following codes:  
```bash
#upload your json file
from google.colab import files
files.upload()

# Next, install the Kaggle API client.
!pip install -q kaggle

# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json

#download the dataset
!kaggle competitions download -c dogs-vs-cats-redux-kernels-edition

#unzip the downloaded dataset
!unzip test.zip
!unzip train.zip
```

