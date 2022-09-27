import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten

import tensorflow_addons as tfa
import cv2
import os 
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import accuracy_score
import random 
import math

import time
import pandas as pd

# User defined libs
import parameters
import dataset
import models

AUTOTUNE = tf.data.AUTOTUNE

            
train_dict, test_dict = dataset.create_dict()
# N_CLASSES = len(train_dict)


train_dataset = tf.data.Dataset.from_generator(train_skin_data, 
                                               output_types=(tf.float64, tf.float32), 
                                               output_shapes=(tf.TensorShape([parameters.N_RES, parameters.N_RES, 3]), tf.TensorShape([parameters.N_CLASSES])))

test_dataset = tf.data.Dataset.from_generator(test_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([parameters.N_RES, parameters.N_RES, 3]), tf.TensorShape([parameters.N_CLASSES])))


# split_size = int(len(train_images) * 0.2)
# split_train_dataset = train_dataset.skip(split_size)
# split_val_dataset = train_dataset.take(split_size)

# split_train_dataset = split_train_dataset.shuffle(128).batch(32).prefetch(AUTOTUNE)
# split_val_dataset = split_val_dataset.shuffle(128).batch(32).prefetch(AUTOTUNE)

model, hist = models.run_expriment('efficient', train_dataset, optimizer='adam', batch_size=32, trainable=True, epochs=25)


# evaluation
model.save(f'{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_effection.h5')

# import pandas as pd
hist_df = pd.DataFrame(hist.history)
with open(f'{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_effection_history.csv', mode='w') as f:
    hist_df.to_csv(f)

