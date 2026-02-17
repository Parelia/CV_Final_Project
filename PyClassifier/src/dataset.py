###########################
# Author: Capuzzo Daniele #
###########################

import collections
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import config

def load_train_test_ds():
    train_set, test_set = keras.utils.image_dataset_from_directory(
        config.directory,
        validation_split=config.val_split,  # 80/20%
        subset="both",
        # shuffle=False,
        seed=config.SEED,
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        label_mode='binary'    # parasitized/uninfected
    )
    classes = train_set.class_names
    return train_set, test_set, classes

def num_img_per_class(dataset):
    class_counts = collections.Counter()
    for _, labels in dataset:
        class_indices = labels.numpy().squeeze().astype(int)  # (batch_size, 1) --> (batch_size,)
        class_counts.update(map(int, class_indices))
    return class_counts


def get_agumented_dataset(train_set):
    preprocessing = keras.Sequential([
        layers.RandomRotation(factor=0.2),
        layers.RandomFlip(mode='horizontal_and_vertical'),
        layers.RandomBrightness(factor=0.15),
        layers.RandomContrast(factor=0.15),
        layers.GaussianNoise(stddev=0.05),
        #  layers.Rescaling(1./255), # data standardization NOT HERE because this would apply only to augmented
    ])
    return train_set.map(lambda x, y: (preprocessing(x), y))
    
