###########################
# Author: Capuzzo Daniele #
###########################

import keras
from keras import layers, optimizers
import config

# function needed by KerasTuner to automatically select the best parameter
# values according to the validation loss
def model_builder(hp):
    hp_activation = hp.Choice('activation', values=['relu', 'leaky_relu', 'elu'])
    model = keras.Sequential()

    # block 1
    model.add(layers.Input(config.input_shape))
    model.add(
        layers.SeparableConv2D(
            filters=hp.Int('c1_filters', min_value=16, max_value=40, step=8),
            kernel_size=(3, 3),
            activation=hp_activation,
            padding='same',
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2))) # 64x64x16
    model.add(layers.Dropout(hp.Float('drop_1', min_value=0.0, max_value=0.4, step=0.025)))


    # block 2
    model.add(
        layers.SeparableConv2D(
            hp.Int('c2_filters', min_value=32, max_value=54, step=8),
            kernel_size=(3, 3),
            activation=hp_activation,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2))) # 32x32x32
    model.add(layers.Dropout(hp.Float('drop_2', min_value=0.0, max_value=0.4, step=0.05)))

    # block 3
    model.add(
        layers.SeparableConv2D(
            filters=hp.Int('c3_filters', min_value=48, max_value=72, step=8),
            kernel_size=(3, 3),
            activation=hp_activation,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2))) # 16x16x48
    model.add(layers.Dropout(hp.Float('drop_3', min_value=0.0, max_value=0.4, step=0.025)))


    # block 4
    model.add(
        layers.SeparableConv2D(
            filters=hp.Int('c4_filters', min_value=64, max_value=88, step=8),
            kernel_size=(3, 3),
            activation=hp_activation,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2))) # 8x8x64
    model.add(layers.Dropout(hp.Float('drop_4', min_value=0.0, max_value=0.4, step=0.025)))

    # block 5
    model.add(
        layers.SeparableConv2D(
            filters=hp.Int('c5_filters', min_value=64, max_value=88, step=8),
            kernel_size=(3, 3),
            activation=hp_activation,
            padding='same'
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2))) # 8x8x64
    model.add(layers.Dropout(hp.Float('drop_5', min_value=0.0, max_value=0.4, step=0.025)))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Dense Layer
    model.add(
        layers.Dense(
            units=hp.Int('dense_units', min_value=32, max_value=128, step=16),
            activation=hp_activation
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Float('drop_dense', min_value=0.0, max_value=0.5, step=0.025)))

    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))


    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    lrate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG')

    # defaults to adam
    optimizer = keras.optimizers.Adam()
    match optimizer_choice:
        case 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=lrate)
        case 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=lrate)
        case 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=lrate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# def model_builder(hp):
#     hp_activation = 'leaky_relu'

#     model = keras.Sequential()
#     model.add(layers.Input(config.input_shape))

#     # make sure any data passed in is properly normalized
#     model.add(layers.Rescaling(1./255))

#     # FURTHER DATA AUGMENTATION (AUTOMATICALLY ONLY ON TRAINING)
#     model.add(layers.RandomRotation(factor=0.2))                    # rotate
#     model.add(layers.RandomFlip(mode='horizontal_and_vertical'))    # flip

#     # optimize the number of blocks and dropout rate
#     num_blocks = hp.Int('num_conv_blocks', min_value=2, max_value=5)
#     conv_dropout = hp.Float('conv_dropout', 0.0, 0.3, step=0.1)

#     for i in range(num_blocks):
#         # i-th block
#         filters = 16 * (2**i)
#         model.add(
#             layers.SeparableConv2D(
#                 filters=filters,
#                 kernel_size=(3, 3),
#                 activation=hp_activation,
#                 padding='same',
#             )
#         )
#         model.add(layers.BatchNormalization())
#         model.add(layers.MaxPooling2D((2, 2)))
#         model.add(layers.Dropout(conv_dropout))


#     # Global Average Pooling
#     model.add(layers.GlobalAveragePooling2D())

#     # Dense Layer
#     model.add(
#         layers.Dense(
#             units=hp.Int('dense_units', min_value=32, max_value=128, step=16),
#             activation=hp_activation
#         )
#     )
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(hp.Float('dense_dropout', min_value=0.0, max_value=0.5, step=0.025)))

#     # Output Layer
#     model.add(layers.Dense(1, activation='sigmoid'))
#     lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG')

#     model.compile(
#         optimizer=keras.optimizers.Nadam(learning_rate=lr), # fake error, doc says that it can be optimizer instance
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
#     return model