import cv2 
import tensorflow as tf
import tensorflow_addons as tfa
import os 
from glob import glob 
import random
import numpy as np

import main
import parameters

def aug1(img, lbl):
    # shape augmentation
    img = tf.image.random_crop(img, [main.num_res, main.num_res, 3])
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

def create_train_list(part='head'):
    train_images = [] 
    test_images = []

    for i in range(6):
        # for key in train_dict.keys():
        
        img = glob(parameters.dataset_path + f'/H{str(i)}/*/{part}/*.jpg')
        train_images.extend(img) 

    # add JeonNam unv
    img = glob(parameters.dataset_path + f'/H9/*/{part}/*.jpg')
    train_images.extend(img) 
            
    random.shuffle(train_images)

    train_labels = [] 
    for img in train_images: 
        lbl = img.split('/')[-2].lower().replace(' ', '')
        
        if lbl in parameters.name_dict:
            lbl = parameters.name_dict[lbl][0]
        
        # lbl
        if lbl in parameters.infection_list:
            lbl = 1 
        else:
            lbl = 0 

        train_labels.append(lbl) 


    train_images = np.reshape(train_images, [-1, 1])
    train_labels = np.reshape(train_labels, [-1, 1])
    
    return train_images, train_labels


def create_test_list(part='head'):
    test_images = []

    for i in range(6):
        # for key in train_dict.keys():
        
        img = glob(parameters.dataset_path + f'/H{str(i)}/*/{part}/*.jpg')
        test_images.extend(img) 

    # add JeonNam unv
    img = glob(parameters.dataset_path + f'/H9/*/{part}/*.jpg')
    test_images.extend(img) 
            
    random.shuffle(test_images)

    test_labels = [] 
    for img in test_images: 
        lbl = img.split('/')[-2].lower().replace(' ', '')
        
        if lbl in parameters.name_dict:
            lbl = parameters.name_dict[lbl][0]
        
        # lbl
        if lbl in parameters.infection_list:
            lbl = 1 
        else:
            lbl = 0 

        test_labels.append(lbl) 


    test_images = np.reshape(test_images, [-1, 1])
    test_labels = np.reshape(test_labels, [-1, 1])
    
    return test_images, test_labels


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
def train_skin_data(images, labels, aug):
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (parameters.num_res, parameters.num_res))

        yield (img, lbl)    

            
def test_skin_data(images, labels,):
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (parameters.num_res, parameters.num_res))
        
        yield (img, lbl)    