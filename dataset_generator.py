import cv2 
import tensorflow as tf
import tensorflow_addons as tfa
import os 
from glob import glob 
import random
import numpy as np

import main
import parameters

from imblearn.over_sampling import *
from imblearn.keras import BalancedBatchGenerator

# def random_oversampling(imgs, lbls): 
    
#     print(f'imgs--------------------->{imgs.shape}')
    
#     x_samp, y_samp = RandomOverSampler(imgs, lbls)
    
    
#     print(f'x_samp--------------------->{x_samp.shape}')
    
    # return x_samp, y_samp

def create_initial_bias(labels):
    non, inf = np.bincount(labels[:, 0])
    
    return np.log([inf / non])
    

def create_class_weight(labels):
    non, inf = np.bincount(labels[:, 0])
    
    total = non + inf 
    
    weight_for_0 = (1 / non) * (total / 2.0)
    weight_for_1 = (1 / inf) * (total / 2.0)
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    return class_weight
    

def aug1(img, lbl):
    # shape augmentation
    img = tf.image.random_crop(img, [parameters.num_res, parameters.num_res, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.flip_up_down(img)
    
    # color augmentation    
    img = tf.image.random_brightness(img, 0.2) 
    img = tf.image.random_contrast(img, 0.2, 0.5) 
    img = tf.image.random_hue(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    
    return img, lbl


def aug(img, label):
    def flip(x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return x
    
    def rotate(x):
        x = tf.cond(tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5,
                   lambda: tfa.image.rotate(x,
                                       tf.random.uniform(shape=[], minval=0.0, maxval=360.0, dtype=tf.float32),
                                       interpolation='BILINEAR'),
                   lambda: x)
        return x
    
    def translation(x):
        dx = tf.random.uniform(shape=[], minval=-10.0, maxval=10.0, dtype=tf.float32)
        dy = tf.random.uniform(shape=[], minval=-10.0, maxval=10.0, dtype=tf.float32)
        x = tf.cond(tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5,
                    lambda: tfa.image.transform(x,
                                                [0, 0, dx, 0, 0, dy, 0, 0],
                                                interpolation='BILINEAR'),
                    lambda: x)
        return x
    
    img = flip(img)
    img = rotate(img)
    img = translation(img)
           
    return img, label

def create_train_list(dataset_path=None, part='head'):
    train_images = [] 
    # test_images = []
    
    if dataset_path is None:
        dataset_path = parameters.dataset_path

    for i in range(7):
        # for key in train_dict.keys():
        
        img = glob(dataset_path + f'/H{str(i)}/*/{part}/*.jpg')
        train_images.extend(img) 

    # add JeonNam unv
    img = glob(dataset_path + f'/H9/*/{part}/*.jpg')
    train_images.extend(img) 
            
    random.shuffle(train_images)

    train_labels = [] 
    for img in train_images: 
        lbl = img.split('/')[-3]
        
        if lbl.lower().replace(' ', '') in parameters.name_dict1:
            lbl = parameters.name_dict1[lbl.lower().replace(' ', '')]
        
        if lbl in parameters.infection_list:
            lbl = 1 
        else:
            lbl = 0 

        train_labels.append(lbl) 
        
    
    print(f'Non-infection found : {train_labels.count(0)}, Infection found : {train_labels.count(1)}')

    train_images = np.reshape(train_images, [-1, 1])
    train_labels = np.reshape(train_labels, [-1, 1])
    
    return train_images, train_labels


def create_test_list(dataset_path=None, part='head'):
    
    if dataset_path is None:
        dataset_path = parameters.dataset_path

    test_images = []

    for i in [7, 8]:
        # for key in train_dict.keys():
        img = glob(dataset_path + f'/H{str(i)}/*/{part}/*.jpg')
        test_images.extend(img) 

    # add JeonNam unv
    img = glob(dataset_path + f'/H9/*/{part}/*.jpg')
    test_images.extend(img) 
            
    random.shuffle(test_images)

    test_labels = [] 
    for img in test_images: 
        # lbl = img.split('/')[-3]
        lbl = img.split('\\')[-3]
        
        if lbl.lower().replace(' ', '') in parameters.name_dict1:
            lbl = parameters.name_dict1[lbl.lower().replace(' ', '')]
        
        # lbl
        if lbl in parameters.infection_list:
            lbl = 1 
        else:
            lbl = 0 

        test_labels.append(lbl) 


    print(f'Non-infection found : {test_labels.count(0)}, Infection found : {test_labels.count(1)}')

    test_images = np.reshape(test_images, [-1, 1])
    test_labels = np.reshape(test_labels, [-1, 1])
    
    return test_images, test_labels

def create_imbalanced_dataset(images, labels, d_type='train'):
    imgs = [] 
    lbls = [] 
    print(f'images--------------------->{images}')
    
    print(f'images len--------------------->{len(images)}')
    
    for img, lbl in zip(images, labels): 
        
        try:
            img = tf.io.read_file(img[0]) 
            img = tf.io.decode_image(img, dtype=tf.float64)
        except Exception as e:
            print(e)
            continue
            
        img = tf.image.resize(img, [parameters.num_res, parameters.num_res])
        
        imgs.append(img)
        lbls.append(lbl)
        
    imgs = np.reshape(imgs, [-1, parameters.num_res * parameters.num_res * 3])
    lbls = np.reshape(lbls, [-1, 1])
    
    if d_type == 'test':
        return tf.data.Dataset.from_tensor_slices((imgs, lbls)).batch(parameters.num_batch, drop_remainder=True)
    else:
        print(f'imgs--------------------->{imgs.shape}')
        sm = SMOTE(random_state=42)
        # x_samp, y_samp = RandomOverSampler(imgs, lbls)
        x_samp, y_samp = sm.fit_resample(imgs, lbls)
        print(f'x_samp--------------------->{x_samp.shape}')
        
        x_samp = np.reshape(x_samp, [-1, parameters.num_res, parameters.num_res, 3])
        # return BalancedBatchGenerator(imgs, lbls, sampler=SMOTE(), batch_size=parameters.num_batch, random_state=42)
        return tf.data.Dataset.from_tensor_slices((x_samp, y_samp)).batch(parameters.num_batch, drop_remainder=True)
        


def create_dataset(images, labels, d_type='train'):
    
    if d_type == 'test':
        return tf.data.Dataset.from_generator(
            test_skin_data, 
            output_types=(tf.float64, tf.float32), 
            output_shapes=(tf.TensorShape([parameters.num_res, parameters.num_res, 3]), tf.TensorShape([1])),
            args=[images, labels])
        
    else:
        return tf.data.Dataset.from_generator(
            train_skin_data, 
            output_types=(tf.float64, tf.float32), 
            output_shapes=(tf.TensorShape([parameters.num_res, parameters.num_res, 3]), tf.TensorShape([1])),
            args=[images, labels])

# def train_skin_data(files):
def train_skin_data(images, labels):
    
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img_path = img
        
        try:
            img = tf.io.read_file(img) 
            img = tf.io.decode_image(img, dtype=tf.float64)
        except:
            print(f'{img_path} is crushed')
            continue
            
        img = tf.image.resize(img, [parameters.num_res, parameters.num_res])

        yield (img, lbl)    

            
def test_skin_data(images, labels,):
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img_path = img
        
        try:
            img = tf.io.read_file(img) 
            img = tf.io.decode_image(img, dtype=tf.float64)
        except:
            print(f'{img_path} is crushed')
            continue
            
        img = tf.image.resize(img, [parameters.num_res, parameters.num_res])
        
        yield (img, lbl)    