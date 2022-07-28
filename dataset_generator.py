import cv2 
import tensorflow as tf
import os 
from glob import glob 
import random
import numpy as np

import main
import parameters

# def create_all_dict():
#     train_dict = {}
#     # test_dict = {} 

#     for i in range(6):
#         files = os.listdir(os.path.join(main.dataset_path, f'H{i}'))
        
#         for f in files: 
                
#             imgs = glob(os.path.join(main.dataset_path, f'H{i}', f) + '/*.jpg')
            
#             key = 0 
#             if f in main.infection_list:
#                 key = 1
                
#             if key in train_dict:
#                 train_dict[key] = train_dict[key] + len(imgs)
#             else:
#                 train_dict[key] = len(imgs)

#     # add JeonNam unv 
#     files = os.listdir(os.path.join(main.dataset_path, 'H9'))

#     for f in files: 
                
#         imgs = glob(os.path.join(main.dataset_path, 'H9', f) + '/*.jpg')

#         key = 0 
#         if f in main.infection_list:
#             key = 1

#         if key in train_dict:
#             train_dict[key] = train_dict[key]+ len(imgs)
#         else:
#             train_dict[key] = len(imgs)
            
#     return train_dict


def create_train_list():
    train_images = [] 
    test_images = []

    for i in range(6):
        # for key in train_dict.keys():
        
        img = glob(parameters.dataset_path + f'/H{str(i)}/*/*.jpg')
        train_images.extend(img) 

    # add JeonNam unv
    img = glob(parameters.dataset_path + '/H9/*/*.jpg')
    train_images.extend(img) 
            
    random.shuffle(train_images)

    train_labels = [] 
    for img in train_images: 
        lbl = img.split('/')[-2].lower().replace(' ', '')
        
        # if lbl in main.name_dict:
        #     lbl = main.name_dict[lbl]
        
        # if lbl not in main.class_list:
        #     print(f'WARNING! NO FOUND CALASS : {lbl}')
        
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


def create_dataset(images, labels, d_type='train', aug=False):
    
    if d_type == 'test':
        return tf.data.Dataset.from_generator(test_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([parameters.num_res, parameters.num_res, 3]), tf.TensorShape([1])),
                                              args=[images, labels])
        
    else:
        return tf.data.Dataset.from_generator(train_skin_data, 
                                              output_types=(tf.float64, tf.float32), 
                                              output_shapes=(tf.TensorShape([parameters.num_res, parameters.num_res, 3]), tf.TensorShape([1])),
                                              args=[images, labels, aug])
    # return train_dct
    

# def cutmix(images, labels):
#     # imgs = []; labs = []
#     # for i in range(N_BATCH):
#     APPLY = tf.cast(tf.random.uniform(()) >= 0.5, tf.int32)
#     idx = tf.random.uniform((), 0, len(train_images), tf.int32)
    
#     # random_img = 0 
#     # random_lbl = 0 
    
#     random_img = cv2.imread(train_images[idx], cv2.COLOR_BGR2YCR_CB)
#     random_img = cv2.resize(random_img, (parameters.N_RES, parameters.N_RES))
#     random_img = cv2.normalize(random_img, None, 0, 255, cv2.NORM_MINMAX)
    
#     random_idx = train_images[idx].split('\\')[2]
    
#     random_key = 0 
#     if random_idx in parameters.infection:
#         random_key = 1
        
#     random_lbl = tf.keras.utils.to_categorical(random_key, parameters.N_CLASSES)

#     W = parameters.N_RES
#     H = parameters.N_RES
#     lam = tf.random.uniform(())
#     cut_ratio = tf.math.sqrt(1.-lam)
#     cut_w = tf.cast(W * cut_ratio, tf.int32) * APPLY
#     cut_h = tf.cast(H * cut_ratio, tf.int32) * APPLY

#     cx = tf.random.uniform((), int(W/8), int(7/8*W), tf.int32)
#     cy = tf.random.uniform((), int(H/8), int(7/8*H), tf.int32)

#     xmin = tf.clip_by_value(cx - cut_w//2, 0, W)
#     ymin = tf.clip_by_value(cy - cut_h//2, 0, H)
#     xmax = tf.clip_by_value(cx + cut_w//2, 0, W)
#     ymax = tf.clip_by_value(cy + cut_h//2, 0, H)

#     mid_left = images[ymin:ymax, :xmin, :]
#     # mid_mid = images[idx, ymin:ymax, xmin:xmax, :]
#     mid_mid = random_img[ymin:ymax, xmin:xmax, :]
#     mid_right = images[ymin:ymax, xmax:, :]
#     middle = tf.concat([mid_left, mid_mid, mid_right], axis=1)
#     top = images[:ymin, :, :]
#     bottom = images[ymax:, :, :]
#     new_img = tf.concat([top, middle, bottom], axis=0)
#     # imgs.append(new_img)

#     cut_w_mod = xmax - xmin
#     cut_h_mod = ymax - ymin
#     alpha = tf.cast((cut_w_mod*cut_h_mod)/(W*H), tf.float32)
#     # label1 = labels[i]
#     label1 = labels
#     # label2 = labels[idx]
#     label2 = random_lbl
#     new_label = ((1-alpha)*label1 + alpha*label2)
#     # labs.append(new_label)
        
#     # new_imgs = tf.reshape(tf.stack(imgs), [-1, N_RES, N_RES, 3])
#     # new_labs = tf.reshape(tf.stack(labs), [-1, N_CLASSES])

#     return new_img, new_label

# def cutmix(images, labels):
#     # imgs = []; labs = []
#     # for i in range(N_BATCH):
#     APPLY = tf.cast(tf.random.uniform(()) >= 0.5, tf.int32)
#     idx = tf.random.uniform((), 0, len(train_images), tf.int32)
    
#     # random_img = 0 
#     # random_lbl = 0 
    
#     random_img = cv2.imread(train_images[idx], cv2.COLOR_BGR2YCR_CB)
#     random_img = cv2.resize(random_img, (N_BEF_RES, N_BEF_RES))
#     random_img = cv2.normalize(random_img, None, 0, 255, cv2.NORM_MINMAX)
    
#     random_idx = train_images[idx].split('/')[-2]
    
#     random_key = 0 
#     if random_idx in infection_list:
#         random_key = 1
        
#     random_lbl = tf.keras.utils.to_categorical(random_key, N_CLASSES)

#     W = N_RES
#     H = N_RES
#     lam = tf.random.uniform(())
#     cut_ratio = tf.math.sqrt(1.-lam)
#     cut_w = tf.cast(W * cut_ratio, tf.int32) * APPLY
#     cut_h = tf.cast(H * cut_ratio, tf.int32) * APPLY

#     cx = tf.random.uniform((), int(W/8), int(7/8*W), tf.int32)
#     cy = tf.random.uniform((), int(H/8), int(7/8*H), tf.int32)

#     xmin = tf.clip_by_value(cx - cut_w//2, 0, W)
#     ymin = tf.clip_by_value(cy - cut_h//2, 0, H)
#     xmax = tf.clip_by_value(cx + cut_w//2, 0, W)
#     ymax = tf.clip_by_value(cy + cut_h//2, 0, H)

#     mid_left = images[ymin:ymax, :xmin, :]
#     # mid_mid = images[idx, ymin:ymax, xmin:xmax, :]
#     mid_mid = random_img[ymin:ymax, xmin:xmax, :]
#     mid_right = images[ymin:ymax, xmax:, :]
#     middle = tf.concat([mid_left, mid_mid, mid_right], axis=1)
#     top = images[:ymin, :, :]
#     bottom = images[ymax:, :, :]
#     new_img = tf.concat([top, middle, bottom], axis=0)
#     # imgs.append(new_img)

#     cut_w_mod = xmax - xmin
#     cut_h_mod = ymax - ymin
#     alpha = tf.cast((cut_w_mod*cut_h_mod)/(W*H), tf.float32)
#     # label1 = labels[i]
#     label1 = labels
#     # label2 = labels[idx]
#     label2 = random_lbl
#     new_label = ((1-alpha)*label1 + alpha*label2)
#     # labs.append(new_label)
        
#     # new_imgs = tf.reshape(tf.stack(imgs), [-1, N_RES, N_RES, 3])
#     # new_labs = tf.reshape(tf.stack(labs), [-1, N_CLASSES])

#     return new_img, new_label

# def train_skin_data(files):
def train_skin_data(images, labels, aug):
    
    # for file in files:
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (parameters.num_res, parameters.num_res))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # print(f'=================>{f}')
        # idx = f.split('\\')[2]
        # idx = f.split('/')[-2]
        
        # key = 0 
        # if idx in infection:
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
            
            
def test_skin_data(images, labels,):
    
    for img, lbl in zip(images, labels):
    
        img = img.decode('utf-8')
        
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (parameters.num_res, parameters.num_res))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # key = 0 
        # if idx in infection_list:
        #     key = 1 
            
        # lbl = tf.keras.utils.to_categorical(key, N_CLASSES)

        yield (img, lbl)    