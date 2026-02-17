###########################
# Author: Capuzzo Daniele #
###########################

#### IMPORTS #### 

# constants
import config

# dataset
import shutil
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import pandas as pd
import collections
# model
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import optimizers
import keras_tuner as kt
import json

# check GPU availability
from tensorflow.python.client import device_lib
tf.config.list_physical_devices('GPU')



#### DATASET ####
# [LOADING] dataset using os
print('LOADING the dataset from folder: ', config.directory)
filepath = []
label = []
folds = os.listdir(config.directory)
for fold in folds:
    f_path = os.path.join(config.directory, fold)
    imgs = os.listdir(f_path)
    for img in imgs:
        img_path = os.path.join(f_path, img)
        filepath.append(img_path)
        label.append(fold)

# Concatenate data paths with labels
file_path_series = pd.Series(filepath, name='filepath')
Label_path_series = pd.Series(label, name='label')
df_train = pd.concat([file_path_series, Label_path_series], axis=1)

# [SPLITTING] dataset into train/test using keras utils
# no need to reshape, resizing done by TensorFlow if necessary
print('SPLITTING the dataset into train/test')
train_set, test_set = keras.utils.image_dataset_from_directory(
    config.directory,
    validation_split=0.2,  # 80/20%
    subset="both",
    # shuffle=False,
    seed=config.SEED,
    image_size=(config.img_height, config.img_width),
    batch_size=config.batch_size,
    label_mode='binary'    # parasitized/uninfected
)
# getting the class names
classes = train_set.class_names
num_classes = len(classes)
print(f'[0={classes[0]}, 1={classes[1]}]')

# [STATISTICS] compute dataset statistics
# calculates how many images there are for each class
def num_img_per_class(dataset):
  class_counts = collections.Counter()
  for _, labels in dataset:
      class_indices = labels.numpy().squeeze().astype(int) 
      class_counts.update(map(int, class_indices))
  return class_counts

# getting the counts
print('COMPUTING DATASETS STATISTICS')
train_class_counts = num_img_per_class(train_set)
val_class_counts = num_img_per_class(test_set)
print('Training set:', train_class_counts)
print('Test set:', val_class_counts)

# [BALANCING] with class weights ####
# weight = num_samples / num_of_that_class
total_samples = sum(train_class_counts.values())
number_of_classes = len(train_class_counts)
class_weights = {
    class_id: total_samples / (number_of_classes * count)
    for class_id, count in train_class_counts.items()
}
# normalize the weights
min_weight = min(class_weights.values())
class_weights = {k: v / min_weight for k, v in class_weights.items()}
print("Computed Class Weights:", class_weights)

# [SPLITTING] to get a validation set
print('SPLITTING Full training set into training and validation')
train_size = int(train_set.cardinality().numpy() * 0.8)
train_reduced = train_set.take(train_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_set = train_set.skip(train_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)



#### MODEL SELECTION ####
# function needed by KerasTuner to automatically select the best parameter
# values according to the validation loss
def model_builder(hp):
    hp_activation = 'leaky_relu'

    model = keras.Sequential()
    model.add(layers.Input(config.input_shape))

    # make sure any data passed in is properly normalized
    model.add(layers.Rescaling(1./255))

    # FURTHER DATA AUGMENTATION (AUTOMATICALLY ONLY ON TRAINING)
    model.add(layers.RandomRotation(factor=0.2))                    # rotate
    model.add(layers.RandomFlip(mode='horizontal_and_vertical'))    # flip

    # optimize the number of blocks and dropout rate
    num_blocks = hp.Int('num_conv_blocks', min_value=2, max_value=5)
    conv_dropout = hp.Float('conv_dropout', 0.0, 0.3, step=0.1)

    for i in range(num_blocks):
        # i-th block
        filters = 16 * (2**i)
        model.add(
            layers.SeparableConv2D(
                filters=filters,
                kernel_size=(3, 3),
                activation=hp_activation,
                padding='same',
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(conv_dropout))


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
    model.add(layers.Dropout(hp.Float('dense_dropout', min_value=0.0, max_value=0.5, step=0.025)))

    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG')

    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=lr), # fake error, doc says that it can be optimizer instance
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# [SETUP] keras tuner
tuner = kt.Hyperband(
    hypermodel=model_builder,
    objective='val_accuracy',
    max_epochs=config.max_epochs,
    factor = config.factor,
    directory = config.tuned_models_dir,
    project_name=config.project_name,
    overwrite=True
)

# set early stopping to prevent overfitting, then run procedure
stop_early = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=config.early_stop_patience, 
    restore_best_weights=True
)

print('SEARCHING for the best model using KerasTuner and Hyperband algorithm')
tuner.search(
    train_reduced, 
    epochs=config.tuner_epochs, 
    validation_data=val_set, 
    callbacks=[stop_early],
    class_weight=class_weights
)
print('SEARCH OVER')


#### SAVING RESULTS ####
# getting the best hyperparaams
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
hp_dict = best_hps.values

# saving them
save_path = os.path.join(config.save_directory, 'best_config.json')
print(f"SAVING the best hyperparameters at: {save_path}")
with open(save_path, 'w') as f:
    json.dump(hp_dict, f, indent=4)

print(f"Successfully saved best hyperparameters to: {save_path}")