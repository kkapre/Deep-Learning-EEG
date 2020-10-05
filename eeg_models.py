# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 01:34:53 2020

@author: kkapr
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape,Input,LSTM,Dense,Dropout,Flatten,Conv2D,BatchNormalization,MaxPooling2D
from tensorflow.keras.regularizers import l2



def RNN(num_classes=4,input_dim=(1000,22),dropout=0.5,lstm_units=100,reg=0.001):
    inputs = Input(shape=input_dim)
    lstm1 = LSTM(units=lstm_units,return_sequences = True,dropout=dropout,kernel_regularizer=l2(reg))(inputs)
    flat1 = Flatten()(lstm1)
    dense1 = Dense(num_classes,activation='softmax',kernel_regularizer=l2(reg))(flat1)

    model = Model(inputs=inputs,outputs=dense1)
    return model


def Image_CNN(num_classes=4,input_dim=(1000,22),num_filter=[50,100],dropout=0.5,filter_size=[3,3],reg=0.001):
    inputs = Input(shape=input_dim)
    rshape= Reshape(input_dim+(1,))(inputs)

    block1 = Conv2D(num_filter[0],(filter_size[0],filter_size[0]),padding='valid',activation='relu',kernel_regularizer=l2(reg))(rshape)
    block1= MaxPooling2D(2,2)(block1)
    block1 = Dropout(dropout)(block1)
    block1 = BatchNormalization()(block1)

    block2 = Conv2D(num_filter[1],(filter_size[1],filter_size[1]),activation='relu',kernel_regularizer=l2(reg))(block1)
    block2 = MaxPooling2D(2,2)(block2)
    block2 = Dropout(dropout)(block2)
    block2= BatchNormalization()(block2)


    flat1 = Flatten()(block2)
    dense1 = Dense(num_classes,activation='softmax',kernel_regularizer=l2(reg))(flat1)

    model = Model(inputs=inputs,outputs=dense1)
    return model
  

def CRNN(num_classes=4,input_dim=(1000,22),num_filter=100,dropout=0.5,filter_size=22,reg=0.001):
    inputs = Input(shape=input_dim)
    rshape= Reshape(input_dim+(1,))(inputs)
    block1 = Conv2D(num_filter,(1,filter_size),activation='relu',kernel_regularizer=l2(reg))(rshape)

    block1 = Dropout(dropout)(block1)
    block1 = BatchNormalization()(block1)
    
    squeezed = tuple([x for x in block1.shape.as_list() if x != 1 and x is not None])
    rshape2 = Reshape(squeezed) (block1)

    block2 = LSTM(units=100,return_sequences = True,dropout=dropout,kernel_regularizer=l2(reg))(rshape2)


    flat1 = Flatten()(block2)
    dense1 = Dense(num_classes,activation='softmax',kernel_regularizer=l2(reg))(flat1)

    model = Model(inputs=inputs,outputs=dense1)
    return model


def EEG_CNN(num_classes=4,input_dim=(1000,22),num_filter=[50,100],dropout=0.5,filter_size=[22,75],reg=0.001):
    inputs = Input(shape=input_dim)
    rshape= Reshape(input_dim+(1,))(inputs)

    block1 = Conv2D(num_filter[0],(1,filter_size[0]),padding='valid',activation='linear',kernel_regularizer=l2(reg))(rshape)
    block1 = Conv2D(num_filter[1],(filter_size[1],1),padding='valid',activation='relu',kernel_regularizer=l2(reg))(block1)
    
    poolsize = np.ceil((input_dim[0]/1000)*70)
    strides = np.ceil((input_dim[1]/22)*12)

    block1= MaxPooling2D(pool_size=(poolsize,1),strides=(strides,1))(block1) #adjust pool size if time bins < 1000
    block1 = Dropout(dropout)(block1)
    block1 = BatchNormalization()(block1)

    flat1 = Flatten()(block1)
    dense1 = Dense(num_classes,activation='softmax',kernel_regularizer=l2(reg))(flat1)

    model = Model(inputs=inputs,outputs=dense1)
    return model



  


