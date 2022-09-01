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
from keras import backend as K
            
if __name__ == '__main__':
            
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--model_name', required=False, default='efficient')
    parser.add_argument('--gpus', required=False, default=2)
    # parser.add_argument('--min_num', required=False, default=min_num)
    # parser.add_argument('--max_num', required=False, default=max_num)
    parser.add_argument('--part', required=False, default='head')
    parser.add_argument('--epochs', required=False, default=10)
    parser.add_argument('--optim', required=False, default='adam')
    
    args = parser.parse_args()
    
    gpus = []
    for gpu in range(int(args.gpus)):
        gpus.append(f'/GPU:{gpu}')
        
    len_inf_list = len(parameters.infection_list)
    len_non_inf_list = len(parameters.class_list_without_infection)
    
    rand_idx_list = [] 
    
    for _ in range(math.ceil(len_non_inf_list / len_inf_list)):
        num_len_inf = 0
        non_inf_list = [] 
        while True:
            rand_idx = np.random.randint(len_non_inf_list)
            
            if rand_idx not in rand_idx_list:
                rand_idx_list.append(rand_idx)
                num_len_inf += 1
                
                non_inf_list.append(parameters.class_list_without_infection[rand_idx])
                
                
            if (num_len_inf >= len_inf_list) or (len(rand_idx_list) >= len_non_inf_list): 
                break
            
        
        inf_images, inf_labels = dataset_generator.create_train_list_by_folders(parameters.infection_list, part=args.part) 
        
        print( )
        print('Non Inf List')
        print(non_inf_list) 
        print( )
        
        non_inf_images, non_inf_labels = dataset_generator.create_train_list_by_folders(non_inf_list, part=args.part) 
        
        train_images = np.concatenate([inf_images, non_inf_images], axis=0) 
        train_labels = np.concatenate([inf_labels, non_inf_labels], axis=0) 

        train = list(zip(train_images, train_labels))
        random.shuffle(train) # shuffle
        train_images, train_labels = zip(*train)
        
        # print(f'{len(train_images)} images were founded')
        
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpus)
        
        with mirrored_strategy.scope():
            train_dataset = dataset_generator.create_dataset(train_images, train_labels) 
            train_dataset = train_dataset.map(dataset_generator.aug1, num_parallel_calls=AUTOTUNE).batch(parameters.num_batch, drop_remainder=True).prefetch(AUTOTUNE)

            model = models.create_model(args.model_name, 
                                        optimizer=args.optim,
                                        num_classes=parameters.num_classes, 
                                        trainable=True, 
                                        num_trainable=-2,
                                        batch_size=parameters.num_batch,
                                        train_length=len(train_images),
                                        )
            
            hist = model.fit(train_dataset, 
                            # validation_data=valid_dataset,
                            # validation_split=0.3, 
                            epochs = args.epochs,
                            # verbose = 1,
                            # class_weight=class_weights, 
                            shuffle=True, 
                            # callbacks=[checkpoints1, checkpoints2]
                            )  
            
            inf_test_images, inf_test_labels = dataset_generator.create_test_list_by_folders(parameters.infection_list, part=args.part) 
            non_inf_test_images, non_inf_test_labels = dataset_generator.create_test_list_by_folders(non_inf_list, part=args.part) 
            
            test_images = np.concatenate([inf_test_images, inf_test_labels], axis=0) 
            test_labels = np.concatenate([non_inf_test_images, non_inf_test_labels], axis=0) 
            
            test = list(zip(test_images, test_labels))
            random.shuffle(test) # shuffle
            test_images, test_labels = zip(*test)
            
            test_dataset = dataset_generator.create_dataset(test_images, test_labels) 
            test_dataset = test_dataset.map(dataset_generator.aug1, num_parallel_calls=AUTOTUNE).batch(parameters.num_batch, drop_remainder=True).prefetch(AUTOTUNE)
            
            loss, acc = model.evaluate(test_dataset)
            
            print(f'----------------------------> loss : {loss} acc : {acc}')

