import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.applications.resnet50 import preprocess_input

import math 

import numpy as np
import main
import parameters as parms

# class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, max_lr, warmup_steps, decay_steps):
#         super(CustomSchedule, self).__init__()
#         self.max_lr = max_lr
#         self.warmup_steps = warmup_steps
#         self.decay_steps = decay_steps

#     def __call__(self, step):
#         lr = tf.cond(step < self.warmup_steps, 
#                     lambda: self.max_lr / self.warmup_steps * step, 
#                     lambda: 0.5 * (1+tf.math.cos(math.pi * (step - self.warmup_steps) / self.decay_steps))*self.max_lr)
#         return lr

# def get_dropout(input_tensor, p=0.3, mc=False):
#     if mc: 
#         layer = Dropout(p, name='top_dropout')
#         return layer(input_tensor, training=True)
#     else:
#         return Dropout(p, name='top_dropout')(input_tensor, training=False)
    

def create_class_weight(train_dict):
    total = np.sum(list(train_dict.values()))
    class_weight = dict()
    # for idx, key in label_to_index.items():
    #     class_weight[key] = train_dict[idx] / total
    class_weight[0] = train_dict[0] / total
    class_weight[1] = train_dict[1] / total

    return class_weight

def create_model(model_name, optimizer='adam', num_classes=2, trainable=False, num_trainable=100, mc=False, batch_size=32, train_length=100, epochs=100): 

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ])
    
    if model_name == 'efficient':
        base_model = keras.applications.EfficientNetB4(include_top=False, input_shape=(parms.num_res, parms.num_res, 3),  weights = 'imagenet')
        # base_model.trainable = trainable
        
        base_model.trainable = trainable
        
        if trainable:
            for layer in base_model.layers[:num_trainable]:
                layer.trainable = False
        
        inputs = keras.Input(shape=(parms.num_res, parms.num_res, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        # x = get_dropout(x, mc)
        x = Dropout(0.3)(x)
        # x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    elif model_name == 'resnet':
        if model_name == 'efficient':
            base_model = keras.applications.resnet50.ResNet50(include_top=False, 
                                                              input_shape=(parms.num_res, parms.num_res, 3),  
                                                              weights = 'imagenet')
        
        base_model.trainable = trainable
        
        if trainable:
            for layer in base_model.layers[:num_trainable]:
                layer.trainable = False
        
        inputs = keras.Input(shape=(parms.num_res, parms.num_res, 3))
        x = keras.applications.resnet50.preprocess_input(inputs)
        x = data_augmentation(x) 
        x = base_model(x)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        # x = get_dropout(x, mc)
        x = Dropout(0.3)(x)
        # x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    elif model_name == 'mobilenet':
        base_model = keras.applications.MobileNetV2(include_top=False, input_shape=(parms.num_res, parms.num_res, 3),  weights = 'imagenet')

        base_model.trainable = trainable
        if trainable:
            for layer in base_model.layers[:num_trainable]:
                layer.trainable = False

        inputs = keras.Input(shape=(parms.num_res, parms.num_res, 3))
        # x = keras.applications.mobilenetv2.pre
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        # x = get_dropout(x, mc)
        x = Dropout(0.3)(x)
        # x = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    # VGG16 
    else:
        base_model = keras.applications.VGG16(include_top=False, input_shape=(parms.num_res, parms.num_res, 3),  weights = 'imagenet')
        base_model.trainable = True
        
        inputs = keras.Input(shape=(parms.num_res, parms.num_res, 3))
        # x = keras.applications.vgg16.prepro
        x = base_model(inputs)
        x = keras.layers.Flatten(name = "avg_pool")(x) 
        x = keras.layers.Dense(512, activation='relu')(x)
        # x = get_dropout(x, mc)
        x = Dropout(0.3)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)

    # LR = 0.001
    LR = 0.001
    steps_per_epoch = (train_length / batch_size)
    # lr_schedule = CustomSchedule(LR, 3*steps_per_epoch, epochs * steps_per_epoch)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
    else:
        optimizer = tf.keras.optimizers.SGD(lr_schedule)
        
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model