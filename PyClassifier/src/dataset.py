###########################
# Author: Capuzzo Daniele #
###########################

import collections
import tensorflow as tf
import keras
import config
import os

def augment(image, label):
    # different by those done by the model
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.image.random_hue(image, max_delta=0.05)
    return image, label


# def get_datasets():
#     train_ds, val_test_ds = keras.utils.image_dataset_from_directory(
#         config.directory,
#         validation_split=config.val_split,
#         shuffle=True,                       # ensure classes are mixed
#         seed=config.SEED,
#         subset="both",
#         image_size=(config.img_height, config.img_width),
#         batch_size=config.batch_size,
#         label_mode='binary'                 # parasitized/uninfected
#     )
#     class_names = train_ds.class_names

#     # split the other dataset to get validation and test
#     val_batches = tf.data.experimental.cardinality(val_test_ds)
#     test_batches = val_batches // 2
#     val_ds = val_test_ds.take(test_batches)
#     test_ds = val_test_ds.skip(test_batches)

#     # augment the trainig set
#     aug_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
#     train_ds = train_ds.concatenate(aug_ds)

#     # shuffle AFTER concatenation
#     train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=tf.data.AUTOTUNE)
#     val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
#     test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
#     print("Datasets ready: Train augmented, Val, Test")

#     return train_ds, val_ds, test_ds, class_names


def get_datasets():

    # loads the train-val dataset from the training subfolder
    # already split here
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(config.directory, "Train"),
        validation_split=config.val_split,
        shuffle=True,                       # ensure classes are mixed
        seed=config.SEED,
        subset="both",
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        label_mode='binary'                 # parasitized/uninfected
    )

    # loads the test dataset from the test subfolder
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(config.directory, "Test"),
        shuffle=False,                      # no need to shuffle the test set
        seed=config.SEED,
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        label_mode='binary'                 # parasitized/uninfected
    )

    class_names = train_ds.class_names

    # augment the trainig set
    aug_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.concatenate(aug_ds)

    # shuffle AFTER concatenation
    train_ds = train_ds.cache().shuffle(len(train_ds) * config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # do not cache the test set
    # test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Datasets ready: Train augmented, Val, Test")

    return train_ds, val_ds, test_ds, class_names

def num_img_per_class(dataset):
    class_counts = collections.Counter()
    for _, labels in dataset:
        class_indices = labels.numpy().flatten().astype(int)  # (batch_size, 1) --> (batch_size,)
        class_counts.update(map(int, class_indices))
    return class_counts
    
