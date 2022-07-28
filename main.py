import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten

from tqdm.keras import TqdmCallback


import tensorflow_addons as tfa
import cv2
import os 
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import accuracy_score
import random 
import math

import time
import pandas as pd


import models 
import dataset_generator 
import parameters


import warnings 
warnings.filterwarnings(action='ignore')

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

AUTOTUNE = tf.data.AUTOTUNE

            
# train_dict, test_dict = dataset.create_dict()
# N_CLASSES = len(train_dict)

            # Open a strategy scope.
            # def create_model(model_name, optimizer='adam', trainable=False, mc=False
            
train_images, train_labels = dataset_generator.create_train_list() 
            
for skf_num in [5, 10]:
        skf = StratifiedKFold(n_splits=skf_num)
        
        kfold = 0 
        for train_idx, valid_idx in skf.split(train_images, train_labels):
            
            strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:0', '/device:GPU:1', '/device:GPU:2'],
                                                      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            with strategy.scope():
                
                train_dataset = dataset_generator.create_dataset(train_images[train_idx], train_labels[train_idx], aug=False) 
                valid_dataset = dataset_generator.create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
                

                train_dataset = train_dataset.batch(parameters.num_batch, drop_remainder=True).prefetch(AUTOTUNE)
                valid_dataset = valid_dataset.batch(parameters.num_batch, drop_remainder=True).prefetch(AUTOTUNE)
                
                model = models.create_model('resnet', 
                                            optimizer='sgd', 
                                            num_classes=parameters.num_classes, 
                                            trainable=True, 
                                            num_trainable=-2,
                                            batch_size=parameters.num_batch,
                                            train_length=len(train_images[train_idx]))

                

                # model, hist = run_expriment('efficient', train_dataset, valid_dataset, class_weights=None, optimizer='sgd', trainable=False, batch_size=N_BATCH, mc=False, epochs=50)
                
                filepath = os.path.join(f'../../models/child_classification_infection/check_point_efficient_{time.strftime("%Y%m%d-%H%M%S")}.h5')
                

                checkpoints = [tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                                  monitor='val_accuracy', 
                                                                  verbose=0, 
                                                                  save_best_only=True,
                                                                  save_weights_only=False, 
                                                                  mode='max', 
                                                                  freq='epoch'), 
                               tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                                  monitor='val_loss', 
                                                                  verbose=0, 
                                                                  save_best_only=True,
                                                                  save_weights_only=False, 
                                                                  mode='min', 
                                                                  freq='epoch')]
                
                hist = model.fit(train_dataset, 
                                validation_data=valid_dataset,
                                epochs = 100,
                                verbose = 1,
                                callbacks=[checkpoints])  


            # evaluation
            model.save(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_infection_kfold_{skf_num}_{kfold}.h5')

            # import pandas as pd
            hist_df = pd.DataFrame(hist.history)
            with open(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_infection_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                hist_df.to_csv(f)

            kfold += 1

