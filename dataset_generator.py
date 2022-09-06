import cv2 
import tensorflow as tf
import tensorflow_addons as tfa
import os 
from glob import glob 
import random
import numpy as np

import main
import parameters

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
    # img = tf.image.random_crop(img, [parameters.num_res, parameters.num_res, 3])
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

def create_part_all_dict(dataset_path, min_num, max_num, part='head'):
    all_dict = dict() 
    count_all_dict = dict() 
    
    kk = [ll for ll in range(7)]
    kk.append(9)

    # for i in range(10):
    for i in kk: 
        folders = os.listdir(os.path.join(dataset_path, f'H{i}'))
        
        for folder in folders:
            imgs = glob(f'{dataset_path}/H{i}/{folder}/{part}/*.jpg')
            
            # folder = folder.lower().replace(' ', '')

            # class 통합 관련 내용 변경
            if folder.lower().replace(' ', '') in parameters.name_dict1: 
                folder = parameters.name_dict1[folder.lower().replace(' ', '')]
            
            if folder not in count_all_dict:
                count_all_dict[folder] = len(imgs) 
            else:
                count_all_dict[folder] += len(imgs)

    new_count_dict = count_all_dict.copy()

    # 데이터 정제
    for key, val in count_all_dict.items():
        if val < min_num:
            del new_count_dict[key]

        if val > max_num:
            new_count_dict[key] = max_num
            

    idx_num = 0 
    for key, val in new_count_dict.items():
        # print(idx)
        all_dict[key] = idx_num 
        idx_num += 1 
        
    return all_dict, new_count_dict

def create_train_list(all_dict, dataset_path=None, part='head'):
    
    images = [] 
    kk = [ll for ll in range(7)]
    kk.append(9)
    
    if dataset_path is None:
        dataset_path = parameters.dataset_path

    for i in kk:
        # for key in train_dict.keys():
        img = glob(dataset_path + f'/H{str(i)}/*/{part}/*.jpg')
        images.extend(img) 
            
    random.shuffle(images)

    train_labels = [] 
    train_images = [] 
    for img in images: 
        
        lbl = get_label(img) 
        
        if lbl not in all_dict:
            continue
        
        if lbl in parameters.infection_list:
            lbl = 1 
        else:
            lbl = 0 

        train_labels.append(lbl) 
        train_images.append(img)
        
    print(f'Non-infection found : {train_labels.count(0)}, Infection found : {train_labels.count(1)}')

    train_images = np.reshape(train_images, [-1, 1])
    train_labels = np.reshape(train_labels, [-1, 1])
    
    return train_images, train_labels


def create_test_list(all_dict, dataset_path=None, part='head'):
    
    if dataset_path is None:
        dataset_path = parameters.dataset_path

    test_images = []

    for i in [7, 8]:
        # for key in train_dict.keys():
        img = glob(dataset_path + f'/H{i}/*/{part}/*.jpg')
        test_images.extend(img) 

    # add JeonNam unv
    # img = glob(dataset_path + f'/H9/*/{part}/*.jpg')
    # test_images.extend(img) 
            
    random.shuffle(test_images)

    test_labels = [] 
    for img in test_images: 
        # print(img)
        # lbl = get_label(img) 
        # lbl = img.split('\\')[3]
        
        # print(lbl)
        lbl = get_test_label(img) 
        
        if lbl not in all_dict:
            continue
        
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

def create_train_list_by_folders(folders, dataset_path=None, part='head'):
    train_images = [] 
    # test_images = []
    
    if dataset_path is None:
        dataset_path = parameters.dataset_path

    for i in range(7):
        # for key in train_dict.keys():
        for folder in folders:
            img = glob(dataset_path + f'/H{str(i)}/{folder}/{part}/*.jpg')
            train_images.extend(img) 

    # add JeonNam unv
    for folder in folder:
        img = glob(dataset_path + f'/H9/{folder}/{part}/*.jpg')
        train_images.extend(img) 
            
    random.shuffle(train_images)

    train_labels = [] 
    for img in train_images: 
        lbl = get_label(img) 
        
        if lbl in parameters.infection_list:
            lbl = 1 
        else:
            lbl = 0 

        train_labels.append(lbl) 
        
    
    print(f'Non-infection found : {train_labels.count(0)}, Infection found : {train_labels.count(1)}')

    train_images = np.reshape(train_images, [-1, 1])
    train_labels = np.reshape(train_labels, [-1, 1])
    
    return train_images, train_labels


def create_test_list_by_folder(folders, dataset_path=None, part='head'):
    
    if dataset_path is None:
        dataset_path = parameters.dataset_path

    test_images = []

    for i in [7, 8]:
        # for key in train_dict.keys():
        for folder in folders:
            img = glob(dataset_path + f'/H{str(i)}/{folder}/{part}/*.jpg')
            test_images.extend(img) 

    # add JeonNam unv
    img = glob(dataset_path + f'/H9/*/{part}/*.jpg')
    test_images.extend(img) 
            
    random.shuffle(test_images)

    test_labels = [] 
    for img in test_images: 
        lbl = get_label(img) 

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

def get_label(img):          
    lbl = img.split('\\')[-3] # E drive
    # lbl = img.split('/')[-3]
    
    if lbl.lower().replace(' ', '') in parameters.name_dict1:
        lbl = parameters.name_dict1[lbl.lower().replace(' ', '')]
            
    if lbl not in parameters.class_list:
        print(lbl)
        raise TypeError(f'Not Found {lbl}')

    return lbl


def get_test_label(img):          
    # print(f'----------------------------------------------->{img}')  
    lbl = img.split('\\')[-3]
    
    if lbl.lower().replace(' ', '') in parameters.name_dict1:
        lbl = parameters.name_dict1[lbl.lower().replace(' ', '')]
            
    if lbl not in parameters.class_list:
        print(lbl)
        raise TypeError(f'Not Found {lbl}')

    return lbl

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
        
def centre_crop(img, num_res): 
    height, width, _ = img.shape
    cent_h, cent_w = height // 2, width // 2
    return img[cent_h - (num_res // 2) : cent_h + (num_res // 2), cent_w - (num_res // 2) : cent_w + (num_res // 2)]

def rotate_img(img, degree=90):        
    if degree == 90:
        degrees = cv2.ROTATE_90_CLOCKWISE
    elif degree == 120:
        degrees = cv2.ROTATE_120
    elif degree == 180:
        degrees = cv2.ROTATE_180
    else:
        # 270
        degrees = cv2.ROTATE_90_COUNTERCLOCKWISE
            
    return cv2.rotate(img, degrees)

def flip_img(img,flip=0):        
    if flip == 90:
        flips = 0
    else:
        flips = 1
            
    return cv2.flip(img, flips)

# def train_skin_data(files):
def train_skin_data(images, labels):
    unique, count = np.unique(labels, return_counts=True)
    
    # print(f'-------------------------------------->{unique}')
    # print(f'-------------------------------------->{count}')
    
    max_dict_values = max(count)
    flip_list = [0, 1]
    degree_list = [90, 120, 180, 270]
    
    for img, lbl in zip(images, labels):
        img = img[0].decode('utf-8')
        
        img_path = img
        try:
            # img = tf.io.decode_image(img, dtype=tf.float64)
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (500, 500))
        except:
            print(f'{img_path} is crushed')
            continue
        
        img = centre_crop(img, parameters.num_res)
        
        # category_lbl = tf.keras.utils.to_categorical(lbl, num_classes)
        # category_lbl = np.reshape(category_lbl, [num_classes])

        yield (img, lbl) 

        count_idx = count[np.where(unique == lbl)[0][0]]
        
        for ii in range(max_dict_values // count_idx):
            if ii % 2 == 0:
                # pass 
                randint = np.random.randint(len(flip_list))
                f_img = flip_img(img, randint)
                
                yield (f_img, lbl) 
            else:
                randint = np.random.randint(len(degree_list))
                r_img = rotate_img(img, randint)
                
                yield (r_img, lbl) 

            
def test_skin_data(images, labels,):
    for img, lbl in zip(images, labels):
    
        img = img[0].decode('utf-8')
        img_path = img
        
        try:
            img = tf.io.read_file(img) 
            img = tf.io.decode_image(img, dtype=tf.float64)
        except:
            continue
            
        img = tf.image.resize(img, [parameters.num_res, parameters.num_res])
        
        yield (img, lbl)    