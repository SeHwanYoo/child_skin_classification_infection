import cv2 
import tensorflow as tf
import os 
from glob import glob 
import random

import parameters

def create_dict():
    
    train_dict = {}
    test_dict = {} 
    
    for i in range(10):

        # 전남대
        if i >= 7 and i < 9:
            continue

        files = os.listdir(os.path.join(parameters.N_DATASET, f'H{i}'))
        
        for f in files: 
                
            imgs = glob(os.path.join(parameters.N_DATASET, f'H{i}', f) + '/*.jpg')
            # print(len(imgs))
            # print(os.path.join(N_DATASET, f'H{i}', f) + '/*.jpg')
            
            # if len(imgs) > min_num and len(imgs) <= max_num: 
            key = 0 
            if f in parameters.infection:
                key = 1
                
            if key in train_dict:
                train_dict[key] = train_dict[key] + len(imgs)
            else:
                train_dict[key] = len(imgs)
                
    for i in range(7, 9): 
    # files = [val for val in list(train_dict.keys())]
        files = os.listdir(os.path.join(parameters.N_DATASET, f'H{i}'))

        for f in files:

            imgs = glob(os.path.join(parameters.N_DATASET, f'H{i}', f) + '/*.jpg')

        key = 0 
        if f in parameters.infection:
            key = 1

        if key in test_dict:
            test_dict[key] = test_dict[key] + len(imgs) 
        else:
            test_dict[key] = len(imgs)             
                
                
    return train_dict, test_dict
    # return train_dct
    

def cutmix(images, labels):
    # imgs = []; labs = []
    # for i in range(N_BATCH):
    APPLY = tf.cast(tf.random.uniform(()) >= 0.5, tf.int32)
    idx = tf.random.uniform((), 0, len(train_images), tf.int32)
    
    # random_img = 0 
    # random_lbl = 0 
    
    random_img = cv2.imread(train_images[idx], cv2.COLOR_BGR2YCR_CB)
    random_img = cv2.resize(random_img, (parameters.N_RES, parameters.N_RES))
    random_img = cv2.normalize(random_img, None, 0, 255, cv2.NORM_MINMAX)
    
    random_idx = train_images[idx].split('\\')[2]
    
    random_key = 0 
    if random_idx in parameters.infection:
        random_key = 1
        
    random_lbl = tf.keras.utils.to_categorical(random_key, parameters.N_CLASSES)

    W = parameters.N_RES
    H = parameters.N_RES
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

def train_skin_data(files):
    
    files = [] 
    for i in range(6):
        # for key in train_dict.keys():
        img = glob(parameters.dataset + f'/H{str(i)}/*/*.jpg')
        files.extend(img) 

    # 전남대
    img = glob(parameters.dataset + f'/H9/*/*.jpg')
    files.extend(img) 
    
    random.shuffle(files)
    
    for file in files:
    
        f = file.decode('utf-8')
        
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (parameters.N_RES, parameters.N_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        idx = f.split('\\')[2]
        
        key = 0 
        if idx in parameters.infection:
            key = 1 
            
        lbl = tf.keras.utils.to_categorical(key, parameters.N_CLASSES)

        yield (img, lbl)    
        
        # if lower than base num, should apply data augmentation
        # if base_num <= int(train_dict[idx]):
        # if key == 0:
            
            # def aug(image, label):
            #     image = tf.image.random_crop(image, [RES, RES, 3])
            #     image = tf.image.random_flip_left_right(image)
            #     return image, label
            # grayscaled_img = tf.image.rgb_to_grayscale(img) 
            # yield (grayscaled_img, lbl)
            
            # saturated
            # saturated_img = tf.image.adjust_saturation(img, 3)
            # yield (saturated_img, lbl)
            
            
            # crop centre 
            # crop_centre_img = tf.image.central_crop(img, central_fraction=0.5)
            # yield (crop_centre_img, lbl)
            
            
            # flip 
            # random_flip_img = tf.image.random_flip_left_right(img)
            # yield (random_flip_img, lbl) 
            
            # # Btight 
            # random_bright_img = tf.image.random_brightness(img, max_delta=0.5)
            # # random_bright_img = tf.clip_by_value(random_bright_img, 0, 255)
            # # random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_img)
            # yield (random_bright_img, lbl) 
    
            # # rotation 90 
            # rotated_img = tf.image.rot90(img)        
            # yield (rotated_img, lbl) 
            
            # # rotation 180
            # rotated_img = tf.image.rot90(img, k=2)        
            # yield (rotated_img, lbl) 
            
            # # rotation 270 
            # rotated_img = tf.image.rot90(img, k=3)        
            # yield (rotated_img, lbl) 
            
            # # curmix 
            # cutmixed_img, cutmixed_lbl = cutmix(img, lbl)
            # yield (cutmixed_img, cutmixed_lbl)
            
            
def test_skin_data(files):
    
    files = [] 
    for i in range(7, 9):
        # for key in train_dict.keys():
        img = glob(parameters.dataset + f'/H{str(i)}/*/*.jpg')
        files.extend(img) 

    
    for file in files:
    
        f = file.decode('utf-8')
        
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (parameters.N_RES, parameters.N_RES))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # lbl = tf.keras.utils.to_categorical(label_to_index[f.split('\\')[1].split('/')[2]], len(train_dict))
        idx = f.split('\\')[2]
        
        key = 0 
        if idx in parameters.infection:
            key = 1 
            
        lbl = tf.keras.utils.to_categorical(key, parameters.N_CLASSES)

        yield (img, lbl)    
        


        
