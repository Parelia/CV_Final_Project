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

# wraps model builder to easily handle 
# custom initializations (i.e: bias)
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

            # block 1
            layers.SeparableConv2D(hp.Int('f1', 16, 48, step=8), (3,3), padding='same', use_bias=False),
            layers.GroupNormalization(groups=8),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(hp.Float('drop_1', 0.0, 0.3, step=0.05)),

            # block 2
            layers.SeparableConv2D(hp.Int('f2', 32, 80, step=8), (3,3), padding='same', use_bias=False),
            layers.GroupNormalization(groups=8),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(hp.Float('drop_2', 0.0, 0.3, step=0.05)),

            # block 3
            layers.SeparableConv2D(hp.Int('f3', 64, 128, step=16), (3,3), padding='same', use_bias=False),
            layers.GroupNormalization(groups=8),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(hp.Float('drop_3', 0.1, 0.4, step=0.05)),

            # head
            layers.GlobalAveragePooling2D(),
            layers.Dense(hp.Int('dense_units', 32, 96, step=16), activation='relu'),
            layers.Dropout(hp.Float('drop_dense', 0.2, 0.5, step=0.05)),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG'),
                clipnorm=5.0
            ),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model

    return model_builder