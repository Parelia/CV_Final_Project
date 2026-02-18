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
def make_builder(pos_counts, neg_counts):

    # precompute initial biases to speed up convergence
    # reduces model predicting all 0 or all 1 initially 
    initial_bias = np.log(pos_counts / neg_counts)
    output_bias = keras.initializers.Constant(initial_bias)

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

            # head
            layers.GlobalAveragePooling2D(),
            layers.Dense(hp.Int('dense_units', 32, 96, step=16), activation='relu', kernel_initializer='he_normal'),
            layers.Dropout(hp.Float('drop_dense', 0.2, 0.5, step=0.05)),
            layers.Dense(1, activation='sigmoid', bias_initializer=output_bias) # IDE error
        ])

        model.compile(
            # here only IDE errors, according to Keras doc, it's fine
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG'),
                clipnorm=5.0
            ),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model

    return model_builder


# def make_builder2(pos_counts, neg_counts):
#     # precompute initial biases to speed up convergence
#     # since it avoids all 0/1 initual guesses 
#     initial_bias = np.log(pos_counts / neg_counts)
#     output_bias = keras.initializers.Constant(initial_bias)

#     def model_builder(hp):
#         activaction = 'leaky_relu'
#         model = keras.Sequential()
#         model.add(layers.Input(config.input_shape))
        
#         # make sure any data passed in is properly normalized
#         model.add(layers.Rescaling(1./255))

#         # FURTHER DATA AUGMENTATION (AUTOMATICALLY ONLY ON TRAINING)
#         model.add(layers.RandomRotation(factor=0.2))                    # rotate
#         model.add(layers.RandomFlip(mode='horizontal_and_vertical'))    # flip
#         model.add(layers.RandomBrightness(factor=0.15))
#         model.add(layers.RandomContrast(factor=0.15))

#         # block 1
#         model.add(
#             layers.SeparableConv2D(
#                 filters=hp.Int('c1_filters', min_value=16, max_value=40, step=8),
#                 kernel_size=(3, 3),
#                 activation=activaction,
#                 padding='same',
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 64x64x16
#         model.add(layers.Dropout(hp.Float('drop_1', min_value=0.0, max_value=0.4, step=0.025)))


#         # block 2
#         model.add(
#             layers.SeparableConv2D(
#                 hp.Int('c2_filters', min_value=32, max_value=54, step=8),
#                 kernel_size=(3, 3),
#                 activation=activaction,
#                 padding='same'
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 32x32x32
#         model.add(layers.Dropout(hp.Float('drop_2', min_value=0.0, max_value=0.4, step=0.05)))

#         # block 3
#         model.add(
#             layers.SeparableConv2D(
#                 filters=hp.Int('c3_filters', min_value=48, max_value=72, step=8),
#                 kernel_size=(3, 3),
#                 activation=activaction,
#                 padding='same'
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 16x16x48
#         model.add(layers.Dropout(hp.Float('drop_3', min_value=0.0, max_value=0.4, step=0.025)))


#         # block 4
#         model.add(
#             layers.SeparableConv2D(
#                 filters=hp.Int('c4_filters', min_value=64, max_value=88, step=8),
#                 kernel_size=(3, 3),
#                 activation=activaction,
#                 padding='same'
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 8x8x64
#         model.add(layers.Dropout(hp.Float('drop_4', min_value=0.0, max_value=0.4, step=0.025)))

#         # Global Average Pooling
#         model.add(layers.GlobalAveragePooling2D())

#         # Dense Layer
#         model.add(
#             layers.Dense(
#                 units=hp.Int('dense_units', min_value=32, max_value=128, step=16),
#                 activation=activaction
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.Dropout(hp.Float('drop_dense', min_value=0.0, max_value=0.5, step=0.025)))

#         # Output Layer
#         model.add( layers.Dense(
#             1, 
#             activation='sigmoid', 
#             bias_initializer=output_bias
#         ))

#         # set NAdam as optimizer, and let tuner decide LR
#         # narrow window because we use also nesterov momemntum
#         lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG')
#         optimizer = keras.optimizers.Nadam(learning_rate=lr)

#         model.compile(
#             optimizer=optimizer,
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
#         return model

#     # return the function
#     return model_builder

# def make_builder(pos_counts, neg_counts):
#     # precompute initial biases to speed up convergence
#     # since it avoids all 0/1 initual guesses 
    
#     # OPTIONAL CUSTOM INITIALIZATION
#     initial_bias = np.log(pos_counts / neg_counts)
#     output_bias = keras.initializers.Constant(initial_bias)

#     def model_builder(hp):
#         hp_activation = hp.Choice('activation', values=['relu', 'leaky_relu', 'elu'])
#         model = keras.Sequential()

#         # block 1
#         model.add(layers.Input(config.input_shape))
#         model.add(
#             layers.SeparableConv2D(
#                 filters=hp.Int('c1_filters', min_value=16, max_value=40, step=8),
#                 kernel_size=(3, 3),
#                 activation=hp_activation,
#                 padding='same',
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 64x64x16
#         model.add(layers.Dropout(hp.Float('drop_1', min_value=0.0, max_value=0.4, step=0.025)))


#         # block 2
#         model.add(
#             layers.SeparableConv2D(
#                 hp.Int('c2_filters', min_value=32, max_value=54, step=8),
#                 kernel_size=(3, 3),
#                 activation=hp_activation,
#                 padding='same'
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 32x32x32
#         model.add(layers.Dropout(hp.Float('drop_2', min_value=0.0, max_value=0.4, step=0.05)))

#         # block 3
#         model.add(
#             layers.SeparableConv2D(
#                 filters=hp.Int('c3_filters', min_value=48, max_value=72, step=8),
#                 kernel_size=(3, 3),
#                 activation=hp_activation,
#                 padding='same'
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 16x16x48
#         model.add(layers.Dropout(hp.Float('drop_3', min_value=0.0, max_value=0.4, step=0.025)))


#         # block 4
#         model.add(
#             layers.SeparableConv2D(
#                 filters=hp.Int('c4_filters', min_value=64, max_value=88, step=8),
#                 kernel_size=(3, 3),
#                 activation=hp_activation,
#                 padding='same'
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 8x8x64
#         model.add(layers.Dropout(hp.Float('drop_4', min_value=0.0, max_value=0.4, step=0.025)))

#         # block 5
#         model.add(
#             layers.SeparableConv2D(
#                 filters=hp.Int('c5_filters', min_value=64, max_value=88, step=8),
#                 kernel_size=(3, 3),
#                 activation=hp_activation,
#                 padding='same'
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2))) # 8x8x64
#         model.add(layers.Dropout(hp.Float('drop_5', min_value=0.0, max_value=0.4, step=0.025)))

#         # Global Average Pooling
#         model.add(layers.GlobalAveragePooling2D())

#         # Dense Layer
#         model.add(
#             layers.Dense(
#                 units=hp.Int('dense_units', min_value=32, max_value=128, step=16),
#                 activation=hp_activation
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.Dropout(hp.Float('drop_dense', min_value=0.0, max_value=0.5, step=0.025)))

#         # Output Layer
#         model.add(layers.Dense(
#             1, 
#             activation='sigmoid', 
#             bias_initializer=output_bias
#         ))


#         optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
#         lrate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')

#         # defaults to adam
#         match optimizer_choice:
#             case 'sgd':
#                 optimizer = keras.optimizers.SGD(learning_rate=lrate, clipnorm=1.0)
#             case 'rmsprop':
#                 optimizer = keras.optimizers.RMSprop(learning_rate=lrate, clipnorm=1.0)
#             case 'adam':
#                 optimizer = keras.optimizers.Adam(learning_rate=lrate, clipnorm=1.0)
#             case _ :
#                 optimizer = keras.optimizers.Adam(learning_rate=lrate, clipnorm=1.0)

#         model.compile(
#             optimizer=optimizer,
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
#         return model

#     # return the function
#     return model_builder