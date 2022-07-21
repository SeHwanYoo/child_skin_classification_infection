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


import warnings 
warnings.filterwarnings(action='ignore')

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

AUTOTUNE = tf.data.AUTOTUNE

# N_BEF_RES = 32
num_res = 32
num_batch = 32 
num_classes = 1

# PATH = 'C:/Users/user/Desktop/Child Skin Disease'
# PATH = '/data/snuh/datasets/Child Skin Disease'
base_path = '../../datasets/Child Skin Disease'
dataset_path = os.path.join(base_path, 'Total_Dataset')
# effect positi
infection_list = ['Abscess',
                'Cellulitis',
                'Chicken pox (varicella)',
                'Cutaneous larva migrans',
                'Eczema herpeticum',
                'Folliculitis',
                'Furuncle',
                'Green nail syndrome',
                'Herpes simplex infection',
                'Herpes zoster',
                'Impetigo',
                'Molluscum contagiosum',
                'Paronychia',
                'Staphylococcal scalded skin syndrome',
                'Tinea capitis',
                'Tinea corporis',
                'Tinea cruris',
                'Tinea faciale', 
                'Tinea manus',
                'Tinea pedis',
                'Verruca plana',
                'Viral exanthem',
                'Wart']

    # if class_weights is None:
    #     hist = model.fit(train_dataset, 
    #                     validation_data=val_dataset,
    #                     epochs = epochs,
    #                     verbose = 0,
    #                     callbacks=[sv])    
    # else:
    #     hist = model.fit(train_dataset.repeat(),
    #                     validation_data = val_dataset,
    #                     epochs = epochs,
    #                     class_weight=class_weights, 
    #                     verbose = 0,
    #                     callbacks=[sv])
    # return model, hist

# def run_expriment(model_name, train_dataset, val_dataset, class_weights=None, optimizer='adam', trainable=False, batch_size=32, mc=False, epochs=100): 

#     strategy = tf.distribute.MirroredStrategy()

#     with strategy.scope():
    
#         if model_name == 'efficient':
#             base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
#             base_model.trainable = trainable
            
#             inputs = keras.Input(shape=(N_RES, N_RES, 3))
#             x = base_model(inputs)
#             x = keras.layers.GlobalAveragePooling2D()(x) 
#             x = get_dropout(x, mc)
#             # x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
#             x = keras.layers.Dense(1, activation='sigmoid')(x)
#             model = tf.keras.Model(inputs=inputs, outputs=x)
            
#         # VGG16 
#         else:
#             base_model = keras.applications.VGG16(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
#             base_model.trainable = True
            
#             inputs = keras.Input(shape=(N_RES, N_RES, 3))
#             x = base_model(inputs)
#             x = keras.layers.Flatten(name = "avg_pool")(x) 
#             x = keras.layers.Dense(512, activation='relu')(x)
#             x = get_dropout(x, mc)
#             x = keras.layers.Dense(256, activation='relu')(x)
#             x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
#             model = tf.keras.Model(inputs=inputs, outputs=x)
            

#         sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'{PATH}/models/{model_name}_mc-{mc}_bs-{batch_size}_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
#                                                 monitor='val_accuracy', 
#                                                 verbose=0, 
#                                                 save_best_only=True,
#                                                 save_weights_only=False, 
#                                                 mode='max', 
#                                                 save_freq='epoch'), 
#             tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
#                                             patience = 4, 
#                                             mode='auto',
#                                             min_delta = 0.01)]

        
#         LR = 0.001
#         # steps_per_epoch = len(train_images) // batch_size
#         # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)
        
#         if optimizer == 'adam':
#             optimizer = tf.keras.optimizers.Adam(LR)
#         else:
#             optimizer = tf.keras.optimizers.SGD(LR)
        
#         model.compile(loss='binary_crossentropy', 
#                     optimizer = optimizer,
#                     metrics=['accuracy'])


#     if class_weights is None:
#         hist = model.fit(train_dataset, 
#                         validation_data=val_dataset,
#                         epochs = epochs,
#                         verbose = 0,
#                         callbacks=[sv])    
#     else:
#         hist = model.fit(train_dataset.repeat(),
#                         validation_data = val_dataset,
#                         epochs = epochs,
#                         class_weight=class_weights, 
#                         verbose = 0,
#                         callbacks=[sv])
    
    
#     return model, hist

# def create_dataset(images, labels, d_type='train', aug=False):
    
#     if d_type == 'test':
#         return tf.data.Dataset.from_generator(test_skin_data, 
#                                               output_types=(tf.float64, tf.float32), 
#                                               output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([1])),
#                                               args=[images, labels])
        
#     else:
#         return tf.data.Dataset.from_generator(train_skin_data, 
#                                               output_types=(tf.float64, tf.float32), 
#                                               output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([1])),
#                                               args=[images, labels, aug])
        
            
# for i in range(7, 9): 
#     # files = [val for val in list(train_dict.keys())]
#     files = os.listdir(os.path.join(dataset, f'H{i}'))
    
#     for f in files:
        
#         imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
#         key = 0 
#         if f in infection_list:
#             key = 1
        
#         if key in test_dict:
#             test_dict[key] = test_dict[key] + len(imgs) 
#         else:
#             test_dict[key] = len(imgs) 
            


if __name__ == '__main__':
    
    # all_dict, count_all_dict = dataset_generator.create_all_dict(dataset_path)
    # num_classes = len(all_dict)
    
    # print(f'number of classes : {num_classes}')

    train_images, train_labels = dataset_generator.create_train_list()

# for skf_num in range(3, 11):
    for skf_num in [5, 10]:
        skf = StratifiedKFold(n_splits=skf_num)
        kfold = 0 
        for train_idx, valid_idx in skf.split(train_images, train_labels):

            strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:0', '/device:GPU:1', '/device:GPU:2'])
            options = tf.data.Options()
            # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

            # Open a strategy scope.
            # def create_model(model_name, optimizer='adam', trainable=False, mc=False
            with strategy.scope():
                model = models.create_model('mobilenet', 
                                            optimizer='sgd', 
                                            num_classes=num_classes, 
                                            trainable=True, 
                                            num_trainable=-2)

            train_dataset = dataset_generator.create_dataset(train_images[train_idx], train_labels[train_idx], aug=False) 
            valid_dataset = dataset_generator.create_dataset(train_images[valid_idx], train_labels[valid_idx]) 
            
            train_dataset = train_dataset.with_options(options)
            valid_dataset = valid_dataset.with_options(options)

            train_dataset = train_dataset.batch(num_batch, drop_remainder=True).prefetch(AUTOTUNE)
            valid_dataset = valid_dataset.batch(num_batch, drop_remainder=True).prefetch(AUTOTUNE)

            # model, hist = run_expriment('efficient', train_dataset, valid_dataset, class_weights=None, optimizer='sgd', trainable=False, batch_size=N_BATCH, mc=False, epochs=50)

            sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'../../models/child_classification_infection/check_point_efficient_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
                                                    monitor='val_accuracy', verbose=0, 
                                                    save_best_only=True,save_weights_only=False, mode='max', 
                                                    freq='epoch'), 
            # tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
            #                                 patience = 4, 
            #                                 mode='auto',
            #                                 min_delta = 0.01)
            # 
            ]

            hist = model.fit(train_dataset, 
                            validation_data=valid_dataset,
                            epochs = 100,
                            verbose = 1,
                            callbacks=[sv])  


            # evaluation
            model.save(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_infection_kfold_{skf_num}_{kfold}.h5')

            # import pandas as pd
            hist_df = pd.DataFrame(hist.history)
            with open(f'../../models/child_classification_infection/{time.strftime("%Y%m%d-%H%M%S")}_efficientb4_infection_kfold_{skf_num}_{kfold}.csv', mode='w') as f:
                hist_df.to_csv(f)

            kfold += 1

