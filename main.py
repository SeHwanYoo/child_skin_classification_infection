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
import argparse


import models 
import dataset_generator 
import parameters


import warnings 
warnings.filterwarnings(action='ignore')

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

AUTOTUNE = tf.data.AUTOTUNE

            
if __name__ == '__main__':
            
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--model_name', required=False, default='efficient')
    parser.add_argument('--gpus', required=False, default=2)
    # parser.add_argument('--min_num', required=False, default=min_num)
    # parser.add_argument('--max_num', required=False, default=max_num)
    parser.add_argument('--part', required=False, default='head')
    parser.add_argument('--epochs', required=False, default=500)
    
    args = parser.parse_args()
    
    gpus = []
    for gpu in range(args.gpus):
        gpus.append(f'/GPU:{gpu}')
    
    train_images, train_labels = dataset_generator.create_train_list() 
                
    for skf_num in [5, 10]:
            skf = StratifiedKFold(n_splits=skf_num)
            
            kfold = 0 
            for train_idx, valid_idx in skf.split(train_images, train_labels):
                
                
                mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpus)
                
                with mirrored_strategy.scope():
                    
                    train_dataset = dataset_generator.create_dataset(train_images[train_idx], train_labels[train_idx]) 
                    valid_dataset = dataset_generator.create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
                    
                    train_dataset = train_dataset.map(dataset_generator.aug1, num_parallel_calls=AUTOTUNE).batch(parameters.num_batch, drop_remainder=True).prefetch(AUTOTUNE)
                    valid_dataset = valid_dataset.map(dataset_generator.aug1, num_parallel_calls=AUTOTUNE).batch(parameters.num_batch, drop_remainder=True).prefetch(AUTOTUNE)
                    
                    model = models.create_model(args.model_name, 
                                                optimizer='sgd', 
                                                num_classes=parameters.num_classes, 
                                                trainable=True, 
                                                num_trainable=-2,
                                                batch_size=parameters.num_batch,
                                                train_length=len(train_images[train_idx]))

                    

                    # model, hist = run_expriment('efficient', train_dataset, valid_dataset, class_weights=None, optimizer='sgd', trainable=False, batch_size=N_BATCH, mc=False, epochs=50)
                    
                    filepath = os.path.join(f'../../models/child_classification_infection/check_point_efficient_{time.strftime("%Y%m%d-%H%M%S")}.h5')
                    

                    checkpoints1 = [tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                                    monitor='val_accuracy', 
                                                                    verbose=0, 
                                                                    save_best_only=True,
                                                                    save_weights_only=False, 
                                                                    mode='max', 
                                                                    freq='epoch')]
                    
                     
                    checkpoints2 = [tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                                    monitor='val_loss', 
                                                                    verbose=0, 
                                                                    save_best_only=True,
                                                                    save_weights_only=False, 
                                                                    mode='min', 
                                                                    freq='epoch')]
                    
                    hist = model.fit(train_dataset, 
                                    validation_data=valid_dataset,
                                    epochs = args.epochs,
                                    # verbose = 1,
                                    callbacks=[checkpoints1, checkpoints2])  


                # evaluation
                model.save(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_{args.model_name}_infection_kfold_{skf_num}_{kfold}.h5')

                # import pandas as pd
                hist_df = pd.DataFrame(hist.history)
                with open(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_{args.model_name}_infection_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                    hist_df.to_csv(f)

                kfold += 1

