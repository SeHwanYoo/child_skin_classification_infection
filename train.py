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

import warnings 
warnings.filterwarnings(action='ignore')

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

AUTOTUNE = tf.data.AUTOTUNE

N_BEF_RES = 256
N_RES = 256 
N_BATCH = 32 

N_CLASSES = 2

# PATH = 'C:/Users/user/Desktop/Child Skin Disease'
# PATH = '/data/snuh/datasets/Child Skin Disease'
PATH = '../../datasets/Child Skin Disease'
dataset = os.path.join(PATH, 'Total_Dataset')

# effect positive
effection = ['Abscess',
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

def cutmix(images, labels):
    # imgs = []; labs = []
    # for i in range(N_BATCH):
    APPLY = tf.cast(tf.random.uniform(()) >= 0.5, tf.int32)
    idx = tf.random.uniform((), 0, len(train_images), tf.int32)
    
    # random_img = 0 
    # random_lbl = 0 
    
    random_img = cv2.imread(train_images[idx], cv2.COLOR_BGR2YCR_CB)
    random_img = cv2.resize(random_img, (N_BEF_RES, N_BEF_RES))
    random_img = cv2.normalize(random_img, None, 0, 255, cv2.NORM_MINMAX)
    
    random_idx = train_images[idx].split('/')[-2]
    
    random_key = 0 
    if random_idx in effection:
        random_key = 1
        
    random_lbl = tf.keras.utils.to_categorical(random_key, N_CLASSES)

    W = N_RES
    H = N_RES
    lam = tf.random.uniform(())
    cut_ratio = tf.math.sqrt(1.-lam)
    cut_w = tf.cast(W * cut_ratio, tf.int32) * APPLY
    cut_h = tf.cast(H * cut_ratio, tf.int32) * APPLY

    cx = tf.random.uniform((), int(W/8), int(7/8*W), tf.int32)
    cy = tf.random.uniform((), int(H/8), int(7/8*H), tf.int32)

    xmin = tf.clip_by_value(cx - cut_w//2, 0, W)
    ymin = tf.clip_by_value(cy - cut_h//2, 0, H)
    xmax = tf.clip_by_value(cx + cut_w//2, 0, W)
    ymax = tf.clip_by_value(cy + cut_h//2, 0, H)

    mid_left = images[ymin:ymax, :xmin, :]
    # mid_mid = images[idx, ymin:ymax, xmin:xmax, :]
    mid_mid = random_img[ymin:ymax, xmin:xmax, :]
    mid_right = images[ymin:ymax, xmax:, :]
    middle = tf.concat([mid_left, mid_mid, mid_right], axis=1)
    top = images[:ymin, :, :]
    bottom = images[ymax:, :, :]
    new_img = tf.concat([top, middle, bottom], axis=0)
    # imgs.append(new_img)

    cut_w_mod = xmax - xmin
    cut_h_mod = ymax - ymin
    alpha = tf.cast((cut_w_mod*cut_h_mod)/(W*H), tf.float32)
    # label1 = labels[i]
    label1 = labels
    # label2 = labels[idx]
    label2 = random_lbl
    new_label = ((1-alpha)*label1 + alpha*label2)
    # labs.append(new_label)
        
    # new_imgs = tf.reshape(tf.stack(imgs), [-1, N_RES, N_RES, 3])
    # new_labs = tf.reshape(tf.stack(labs), [-1, N_CLASSES])

    return new_img, new_label

# def train_skin_data(files):
def train_skin_data(images, labels, aug):
    
    # for file in files:
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # print(f'=================>{f}')
        # idx = f.split('\\')[2]
        # idx = f.split('/')[-2]
        
        # key = 0 
        # if idx in effection:
        #     key = 1 
        # lbl = tf.keras.utils.to_categorical(key, N_CLASSES)

        yield (img, lbl)    

        # if aug:
        #     pass
        
        # if lower than base num, should apply data augmentation
        # if base_num <= int(train_dict[idx]):
        # if key == 0:
            
        #     # saturated
        #     saturated_img = tf.image.adjust_saturation(img, 3)
        #     yield (saturated_img, lbl)
            
        #     # flip 
        #     random_flip_img = tf.image.random_flip_left_right(img)
        #     yield (random_flip_img, lbl) 
            
        #     # Btight 
        #     random_bright_img = tf.image.random_brightness(img, max_delta=0.5)
        #     yield (random_bright_img, lbl) 
    
        #     # rotation 90 
        #     rotated_img = tf.image.rot90(img)        
        #     yield (rotated_img, lbl) 
            
        #     # rotation 180
        #     rotated_img = tf.image.rot90(img, k=2)        
        #     yield (rotated_img, lbl) 
            
        #     # rotation 270 
        #     rotated_img = tf.image.rot90(img, k=3)        
        #     yield (rotated_img, lbl) 
            
        #     # curmix 
        #     cutmixed_img, cutmixed_lbl = cutmix(img, lbl)
        #     yield (cutmixed_img, cutmixed_lbl)
            
            
def test_skin_data(files):
    
    for file in files:
    
        f = file.decode('utf-8')
        
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_BEF_RES, N_BEF_RES))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        idx = f.split('/')[-2]
        
        key = 0 
        if idx in effection:
            key = 1 
            
        lbl = tf.keras.utils.to_categorical(key, N_CLASSES)

        yield (img, lbl)    
        
        
def get_dropout(input_tensor, p=0.3, mc=False):
    if mc: 
        layer = Dropout(p, name='top_dropout')
        return layer(input_tensor, training=True)
    else:
        return Dropout(p, name='top_dropout')(input_tensor, training=False)
    

def create_class_weight(train_dict):
    total = np.sum(list(train_dict.values()))
    class_weight = dict()
    # for idx, key in label_to_index.items():
    #     class_weight[key] = train_dict[idx] / total
    class_weight[0] = train_dict[0] / total
    class_weight[1] = train_dict[1] / total

    return class_weight

def create_model(model_name, optimizer='adam', trainable=False, num_trainable=100, mc=False): 

    # strategy = tf.distribute.MirroredStrategy()

    # with strategy.scope():
    
    if model_name == 'efficient':
        base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        # base_model.trainable = trainable
        
        base_model.trainable = trainable
        
        if trainable:
            for layer in base_model.layers[:num_trainable]:
                layer.trainable = False
        
        inputs = keras.Input(shape=(N_RES, N_RES, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        x = get_dropout(x, mc)
        # x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    # VGG16 
    else:
        base_model = keras.applications.VGG16(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model.trainable = True
        
        inputs = keras.Input(shape=(N_RES, N_RES, 3))
        x = base_model(inputs)
        x = keras.layers.Flatten(name = "avg_pool")(x) 
        x = keras.layers.Dense(512, activation='relu')(x)
        x = get_dropout(x, mc)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)

    LR = 0.001
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(LR)
    else:
        optimizer = tf.keras.optimizers.SGD(LR)
        
    model.compile(loss='binary_crossentropy', 
                optimizer=optimizer,
                metrics=['accuracy'])

    return model
        

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

def run_expriment(model_name, train_dataset, val_dataset, class_weights=None, optimizer='adam', trainable=False, batch_size=32, mc=False, epochs=100): 

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
    
        if model_name == 'efficient':
            base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
            base_model.trainable = trainable
            
            inputs = keras.Input(shape=(N_RES, N_RES, 3))
            x = base_model(inputs)
            x = keras.layers.GlobalAveragePooling2D()(x) 
            x = get_dropout(x, mc)
            # x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
            x = keras.layers.Dense(1, activation='sigmoid')(x)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            
        # VGG16 
        else:
            base_model = keras.applications.VGG16(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
            base_model.trainable = True
            
            inputs = keras.Input(shape=(N_RES, N_RES, 3))
            x = base_model(inputs)
            x = keras.layers.Flatten(name = "avg_pool")(x) 
            x = keras.layers.Dense(512, activation='relu')(x)
            x = get_dropout(x, mc)
            x = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            

        sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'{PATH}/models/{model_name}_mc-{mc}_bs-{batch_size}_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
                                                monitor='val_accuracy', 
                                                verbose=0, 
                                                save_best_only=True,
                                                save_weights_only=False, 
                                                mode='max', 
                                                save_freq='epoch'), 
            tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                            patience = 4, 
                                            mode='auto',
                                            min_delta = 0.01)]

        
        LR = 0.001
        # steps_per_epoch = len(train_images) // batch_size
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)
        
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(LR)
        else:
            optimizer = tf.keras.optimizers.SGD(LR)
        
        model.compile(loss='binary_crossentropy', 
                    optimizer = optimizer,
                    metrics=['accuracy'])


    if class_weights is None:
        hist = model.fit(train_dataset, 
                        validation_data=val_dataset,
                        epochs = epochs,
                        verbose = 0,
                        callbacks=[sv])    
    else:
        hist = model.fit(train_dataset.repeat(),
                        validation_data = val_dataset,
                        epochs = epochs,
                        class_weight=class_weights, 
                        verbose = 0,
                        callbacks=[sv])
    
    
    return model, hist

def create_dataset(images, labels, d_type='train', aug=False):
    
    if d_type == 'test':
        return tf.data.Dataset.from_generator(test_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([1])),
                                              args=[images, labels])
        
    else:
        return tf.data.Dataset.from_generator(train_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([N_RES, N_RES, 3]), tf.TensorShape([1])),
                                              args=[images, labels, aug])
        
        
train_dict = {}
test_dict = {} 

for i in range(6):
    files = os.listdir(os.path.join(dataset, f'H{i}'))
    
    for f in files: 
               
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        key = 0 
        if f in effection:
            key = 1
            
        if key in train_dict:
            train_dict[key] = train_dict[key] + len(imgs)
        else:
            train_dict[key] = len(imgs)

# add JeonNam unv 
files = os.listdir(os.path.join(dataset, 'H9'))

for f in files: 
               
    imgs = glob(os.path.join(dataset, 'H9', f) + '/*.jpg')

    key = 0 
    if f in effection:
        key = 1

    if key in train_dict:
        train_dict[key] = train_dict[key]+ len(imgs)
    else:
        train_dict[key] = len(imgs)

            
for i in range(7, 9): 
    # files = [val for val in list(train_dict.keys())]
    files = os.listdir(os.path.join(dataset, f'H{i}'))
    
    for f in files:
        
        imgs = glob(os.path.join(dataset, f'H{i}', f) + '/*.jpg')
        
        key = 0 
        if f in effection:
            key = 1
        
        if key in test_dict:
            test_dict[key] = test_dict[key] + len(imgs) 
        else:
            test_dict[key] = len(imgs) 
            

train_images = [] 
test_images = []

for i in range(6):
    # for key in train_dict.keys():
    
    img = glob(dataset + f'/H{str(i)}/*/*.jpg')
    train_images.extend(img) 

# add JeonNam unv
img = glob(dataset + '/H9/*/*.jpg')
train_images.extend(img) 
        
for i in range(7, 9):
    # for key in train_dict.keys():
    img = glob(dataset + f'/H{str(i)}/*/*.jpg')
    test_images.extend(img) 
        
        
random.shuffle(train_images)
random.shuffle(test_images)

train_labels = [] 
test_labels = [] 

for img in train_images: 
    lbl = img.split('/')[-2]
    # lbl
    if lbl in effection:
        lbl = 1 
    else:
        lbl = 0 

    train_labels.append(lbl) 


train_images = np.reshape(train_images, [-1, 1])
train_labels = np.reshape(train_labels, [-1, 1])


# for skf_num in range(3, 11):
for skf_num in [5, 10]:
    skf = StratifiedKFold(n_splits=skf_num)
    kfold = 0 
    for train_idx, valid_idx in skf.split(train_images, train_labels):

        strategy = tf.distribute.MirroredStrategy()
        # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        # def create_model(model_name, optimizer='adam', trainable=False, mc=False
        with strategy.scope():
            model = create_model('efficient', 
                                 optimizer='sgd', 
                                 trainable=True, 
                                 num_trainable=-2)

        train_dataset = create_dataset(train_images[train_idx], train_labels[train_idx], aug=False) 
        valid_dataset = create_dataset(train_images[valid_idx], train_labels[valid_idx]) 

        train_dataset = train_dataset.batch(N_BATCH, drop_remainder=True).prefetch(AUTOTUNE)
        valid_dataset = valid_dataset.batch(N_BATCH, drop_remainder=True).prefetch(AUTOTUNE)

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

