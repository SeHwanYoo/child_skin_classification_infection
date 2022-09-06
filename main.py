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
from sklearn.utils import class_weight 

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
from keras import backend as K
            
if __name__ == '__main__':
            
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--model_name', required=False, default='efficient')
    parser.add_argument('--gpus', required=False, default=1)
    parser.add_argument('--min_num', required=False, type=int, default=parameters.min_num)
    parser.add_argument('--max_num', required=False, type=int, default=parameters.max_num)
    parser.add_argument('--part', required=False, default='head')
    parser.add_argument('--epochs', required=False, type=int, default=500)
    parser.add_argument('--optim', required=False, default='adam')
    parser.add_argument('--batch_size', required=False, type=int, default=parameters.num_batch)
    parser.add_argument('--trainable', required=False, default=True)
    
    args = parser.parse_args()
    
    gpus = []
    for gpu in range(int(args.gpus)):
        gpus.append(f'/GPU:{gpu}')
        
    all_dict, count_all_dict = dataset_generator.create_part_all_dict(parameters.dataset_path, args.min_num, args.max_num, part=args.part)
    num_classes = len(all_dict)
        
    train_images, train_labels = dataset_generator.create_train_list(all_dict, part=args.part) 
    
    print(f'{len(train_images)} images were founded')
    sum_count_all_dict = sum([ll for ll in count_all_dict.values()])
    print(f'Now {sum_count_all_dict} images were founded')

    # initial_bias = dataset_generator.create_initial_bias(train_labels)
    
    # print(f'initial_bias : {initial_bias}')
    
    # class_weights = dataset_generator.create_class_weight(train_labels)
    # class_weights = {0 : 0.1, 1 : 0.9}
    # print(f'class weight : {class_weights} weights applied!')
                
    for skf_num in [5, 10]:
            skf = StratifiedKFold(n_splits=skf_num)
            
            kfold = 0 
            for train_idx, valid_idx in skf.split(train_images, train_labels):
                
                # mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                # with tf.device(f'/device:CPU:{args.gpus}'):
                    train_dataset = dataset_generator.create_dataset(train_images[train_idx], train_labels[train_idx]) 
                    valid_dataset = dataset_generator.create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
                    
                    train_dataset = train_dataset.map(dataset_generator.aug1, num_parallel_calls=AUTOTUNE).batch(args.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
                    valid_dataset = valid_dataset.map(dataset_generator.aug1, num_parallel_calls=AUTOTUNE).batch(args.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

                    model = models.create_model(args.model_name, 
                                                optimizer=args.optim,
                                                num_classes=parameters.num_classes, 
                                                trainable=args.trainable, 
                                                num_trainable=-2,
                                                batch_size=args.batch_size,
                                                train_length=len(train_images[train_idx]),
                                                # output_bias=initial_bias
                                                )

                    
                    min_filepath = os.path.join(f'../../models/child_classification_infection/min_check_point_efficient_{time.strftime("%Y%m%d-%H%M%S")}.h5')
                    max_filepath = os.path.join(f'../../models/child_classification_infection/max_check_point_efficient_{time.strftime("%Y%m%d-%H%M%S")}.h5')
                    

                    checkpoints1 = [tf.keras.callbacks.ModelCheckpoint(max_filepath, 
                                                                       monitor='val_accuracy', 
                                                                       verbose=0, 
                                                                       save_best_only=True,
                                                                       save_weights_only=False, 
                                                                       mode='max', 
                                                                       freq='epoch')]
                    
                    checkpoints2 = [tf.keras.callbacks.ModelCheckpoint(min_filepath, 
                                                                       monitor='val_loss', 
                                                                       verbose=0, 
                                                                       save_best_only=True,
                                                                       save_weights_only=False, 
                                                                       mode='min', 
                                                                       freq='epoch')]
                    
                    checkpoint3 = [tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                                           patience=4,
                                                           min_delta=0)]
                    

                    hist = model.fit(train_dataset, 
                                    validation_data=valid_dataset,
                                    epochs = args.epochs,
                                    # verbose = 1,
                                    # class_weight=class_weights, 
                                    shuffle=True, 
                                    callbacks=[checkpoints1, checkpoints2, checkpoint3])  
                    
                    
                    model.save(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_{args.model_name}_infection_kfold_{skf_num}_{kfold}.h5')
                    
                    hist_df = pd.DataFrame(hist.history)
                    with open(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_{args.model_name}_infection_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                        hist_df.to_csv(f)
                        
                    kfold += 1


                # evaluation
                # model.save(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_{args.model_name}_infection_kfold_{skf_num}_{kfold}.h5')
                # model.save(f'{time.strftime("%Y%m%d-%H%M%S")}_{args.model_name}_infection_kfold_{skf_num}_{kfold}.h5')

                
                # hist_df = pd.DataFrame(hist.history)
                # with open(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_{args.model_name}_infection_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                #     hist_df.to_csv(f)

                # kfold += 1

