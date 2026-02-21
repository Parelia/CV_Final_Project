###########################
# Author: Capuzzo Daniele #
###########################

import keras
from keras import layers, optimizers
import config

import tensorflow as tf
import numpy as np


# MODEL FACTORY (design pattern)
# function needed by KerasTuner to automatically select the best parameter
# values according to the validation loss
def make_builder():
    def model_builder(hp):
        model = keras.Sequential([
            layers.Input(config.input_shape),

            # normalize
            layers.Rescaling(1./255),

            # augmentation
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(hp.Float('rotation', 0.1, 0.3, step=0.1)),
            layers.RandomZoom(hp.Float('zoom', 0.05, 0.2, step=0.05)),

            # THESE BAD GUYS BROKE THE PIPELINE (FULL MODEL COLLAPSE)
            # this is a reminder to whoever comes after me: save yourselves
            # 4 hours of bug fixing. If you are better than me, try... be warned however
            # TOTAL HOURS COUNTER: 4
            # layers.RandomBrightness(factor=0.15),                    # was max_delta
            # layers.RandomContrast(factor=0.15),

            # block 1
            layers.Conv2D(hp.Int('f1', 16, 48, step=8), (3,3), padding='same', kernel_initializer='he_normal', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(hp.Float('drop_1', 0.0, 0.3, step=0.05)),

            # block 2
            layers.Conv2D(hp.Int('f2', 32, 80, step=8), (3,3), padding='same', kernel_initializer='he_normal', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(hp.Float('drop_2', 0.0, 0.3, step=0.05)),

            # block 3
            layers.Conv2D(hp.Int('f3', 64, 128, step=16), (3,3), padding='same', kernel_initializer='he_normal', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(hp.Float('drop_3', 0.1, 0.4, step=0.05)),

            # block 4
            layers.Conv2D(hp.Int('f4', 64, 128, step=16), (3,3), padding='same', kernel_initializer='he_normal', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(hp.Float('drop_4', 0.1, 0.4, step=0.05)),

            # head
            layers.GlobalAveragePooling2D(),
            layers.Dense(hp.Int('dense_units', 32, 96, step=16), activation='relu', kernel_initializer='he_normal'),
            layers.Dropout(hp.Float('drop_dense', 0.2, 0.5, step=0.05)),
            layers.Dense(1, activation='sigmoid') # IDE error
        ])

        model.compile(
            # here only IDE errors, according to Keras doc, it's fine
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG'),
                clipnorm=5.0
            ),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model

    return model_builder