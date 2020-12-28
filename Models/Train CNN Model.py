from keras.layers import Input
from keras import layers
from keras.regularizers import l2
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
import tensorflow.compat.v1 as tf
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import h5py
import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
# from keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import TensorBoard
import cv2
import matplotlib.pyplot as plt
from Models.CNN_MODEL import CNN_MODEL2

batch_size = 32
num_epochs = 10000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50



# mc = ModelCheckpoint(
#     filepath='../Trained Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
#     monitor='val_loss',
#     verbose=0,
#     save_best_only=True,
#     save_weights_only=False,
#     mode='max',
#     period=5)

mc = ModelCheckpoint(
    filepath='../Trained Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True)

es = EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=50,
    verbose=1,
    mode='max')
# rlrop = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.1,
#     patience=patience/4,
#     verbose=0,
#     mode='auto',
#     min_delta=0.0001,
#     cooldown=4,
#     min_lr=10e-7)

rlrop = ReduceLROnPlateau('val_loss', factor=0.1,
                          patience=int(patience / 4), verbose=1)

csvl = CSVLogger(
    filename='../tmp/training.log',
    separator=',',
    append=False)
ts = TensorBoard(
    log_dir='../tmp')
# histogram_freq=0,
# write_graph=True,
# write_images=False,
# embeddings_freq=100,
# embeddings_layer_names=None, # this list of embedding layers...
# embeddings_metadata=None)      # with this metadata associated with them.)

callbacks = [mc, es, rlrop, csvl]

# data generator
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split) * num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


def Emotion_Data_set(Emotion_Detection_CSV_Path):
    width, height = 48, 48
    emotions = []
    data = pd.read_csv(Emotion_Detection_CSV_Path)
    image_strings = data['pixels'].tolist()
    image_size = (64, 64)

    for image_string in image_strings:
        emotion = [int(pixle) for pixle in image_string.split()]
        emotion = np.asarray(emotion)
        emotion = emotion.reshape(width, height)
        # emotion = cv2.resize(emotion.astype('uint8'), image_size)
        emotions.append(emotion.astype('float32'))
    emotions = np.asarray(emotions)
    emotions = np.expand_dims(emotions, -1)
    emotions = emotions / 255.0

    emotion_label = pd.get_dummies(data['emotion']).as_matrix()

    # emm = data.groupby('emotion').reset_index()
    # print("emm : {}".format(emm.shape))
    # print("faces : {}".format(emotions.shape))
    # print("emotions : {}".format(emotions.shape))
    # print("emotions : {}".format(type(emotions)))

    return emotions, emotion_label


Emotion_Detection_CSV_Path = "../Data/Face_Expression_Detection_Model/fer2013.csv"
Emotion_Data_Set, EMotion_Label = Emotion_Data_set(Emotion_Detection_CSV_Path)

# model = simple_CNN(input_shape, num_classes)
model = CNN_MODEL2(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

train_data, val_data = split_data(Emotion_Data_Set, EMotion_Label, validation_split)
train_faces, train_emotions = train_data

H = model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)

# plot the training + testing loss and accuracy
# plt.style.use("ggplot")
# # plt.figure()
# # plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
# # plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
# # plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
# # plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
# # plt.title("Training Loss and Accuracy")
# # plt.xlabel("Epoch #")
# # plt.ylabel("Loss/Accuracy")
# # plt.legend()
# # plt.show()
