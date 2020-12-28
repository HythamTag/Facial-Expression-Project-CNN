#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten,

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import cv2
import pandas as pd 
#from tensorflow.python.keras.callbacks import TensorBoard

width = 48
height = 48
num_classes = 7
batch_size = 256
epochs = 1000
patience = 50

input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7


# In[2]:


#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

Emotion_Data_Set_Path = "Data/Face_Expression_Detection_Model/fer2013.csv"


# In[3]:


DF_fer2013 = pd.read_csv(Emotion_Data_Set_Path)


# In[4]:


train_set = DF_fer2013[DF_fer2013.Usage == 'Training'] .reset_index()
test_set  = DF_fer2013[DF_fer2013.Usage == 'PublicTest'].reset_index()


train_data_list=train_set.pixels.tolist()
test_data_list =test_set.pixels.tolist()
train_label= train_set.emotion.tolist()
test_label = test_set.emotion.tolist()

train_label = to_categorical(train_set.emotion)
test_label = to_categorical(test_set.emotion)


# In[5]:


def process_string(set):
    list= []
    for line in set:
        list.append(line.split())
    return list
        
    


# In[6]:


train_data = process_string(train_data_list)
test_data= process_string(test_data_list)


# In[7]:


x_train = np.array(train_data,'float32')
y_train = np.array(train_label,'float32')
x_test  = np.array(test_data,'float32')
y_test  = np.array(test_label,'float32')

x_train = x_train.reshape(x_train.shape[0], width, height, 1)
x_test = x_test.reshape(x_test.shape[0], width, height,1)


# In[8]:


x_train= x_train/255.0
x_test= x_test/255.0


# In[9]:


#------------------------------#------------------------------
print("X_Train Shape: {}".format(x_train.shape))
print("Y_Train Shape: {}".format(y_train.shape))
print("X_Test Shape: {}".format(x_test.shape))
print("X_Test Shape: {}".format(y_test.shape))

print("X_Train dtype: {}".format(x_train.dtype))
print("Y_Train dtype: {}".format(y_train.dtype))
print("X_Test dtype: {}".format(x_test.dtype))
print("X_Test dtype: {}".format(y_test.dtype))

print("X_Train ndim: {}".format(x_train.ndim))
print("Y_Train ndim: {}".format(y_train.ndim))
print("X_Test ndim: {}".format(x_test.ndim))
print("X_Test ndim: {}".format(y_test.ndim))

print("X_Train size: {}".format(x_train.size))
print("Y_Train size: {}".format(y_train.size))
print("X_Test size: {}".format(x_test.size))
print("X_Test size: {}".format(y_test.size))

print("X_Train dtype.name: {}".format(x_train.dtype.name))
print("Y_Train dtype.name: {}".format(y_train.dtype.name))
print("X_Test dtype.name: {}".format(x_test.dtype.name))
print("X_Test dtype.name: {}".format(y_test.dtype.name))

print("number of images: ",len(x_test)+len(x_train))
print("instance length train: ",x_train.shape)
print("instance length test: ",x_test.shape)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#------------------------------#------------------------------


# In[13]:


def CNN_Model_v1(input_shape,num_classes):
    #construct CNN structure
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate = 1 - 0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate = 1 - 0.2))

    model.add(Dense(num_classes, activation='softmax'))
    return model


# In[14]:


mc = ModelCheckpoint(
    filepath='Trained Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True)

es = EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=50,
    verbose=1,
    mode='max')
rlrop = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

csvl = CSVLogger(
    filename='tmp/training.log',
    separator=',',
    append=False)
ts = TensorBoard(log_dir='tmp')

callbacks = [mc, es, rlrop, csvl, ts]


# In[ ]:



gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

#------------------------------
model = CNN_Model_v1(input_shape,num_classes)
model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)

model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test))


history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_generator.n//train_generator.batch_size,
                                epochs=epochs,
                                validation_data = validation_generator,
                                validation_steps = validation_generator.n//validation_generator.batch_size,
                                callbacks=callbacks
                                )
# In[ ]:


#model.fit_generator(train_generator, steps_per_epoch=len(x_train) / batch_size, epochs=epochs ,callbacks=callbacks , verbose=1, validation_data=val_data)
#model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs,validation_data=val_data,callbacks=callbacks )


# In[ ]:




