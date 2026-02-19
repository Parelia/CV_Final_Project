###########################
# Author: Capuzzo Daniele #
###########################

import collections
import tensorflow as tf
import keras
import config
import os

# def augment(image, label):
#     # different by those done by the model
#     image = tf.image.random_brightness(image, max_delta=0.15)
#     image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
#     image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
#     image = tf.image.random_hue(image, max_delta=0.05)
#     return image, label

def make_ds(dataset):
    return dataset.shuffle(buffer_size=1000).repeat()

def get_unbalanced_ds(batch_size=None):
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(config.directory, "Train"),
        shuffle=True,
        seed=config.SEED,
        image_size=(config.img_height, config.img_width),
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(config.directory, "Validation"),
        shuffle=True,
        seed=config.SEED,
        image_size=(config.img_height, config.img_width),
        batch_size=batch_size,
        label_mode='binary'
    )
    print(
        f'Dataset information:\n  - Classes: {train_ds.class_names}\n  - Training samples: {len(train_ds)}\n  - Validation samples: {len(val_ds)}'
    )

    return train_ds, val_ds

def prepare_and_balance(train_ds_unbatched, val_ds_unbatched):
    
    # get the parasitized and uninfected cells and cache them
    parasitized_ds = train_ds_unbatched.filter(lambda x, y: tf.equal(y[0], 0.0)).cache()
    uninfected_ds  = train_ds_unbatched.filter(lambda x, y: tf.equal(y[0], 1.0)).cache()
    
    # getting the counts
    para_count = parasitized_ds.reduce(0, lambda x, _: x + 1).numpy()
    uni_count = uninfected_ds.reduce(0, lambda x, _: x + 1).numpy()

    # rebalance the training set with Oversampling
    print('Rebalancing training set with Oversampling')
    parasitized_ds = make_ds(parasitized_ds)
    uninfected_ds  = make_ds(uninfected_ds)

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        [parasitized_ds, uninfected_ds],
        weights=[0.5, 0.5]
    )

    # no shuffle, no repeat on the combined dataset
    train_ds = balanced_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_unbatched.batch(config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, para_count, uni_count


def merge_and_balance(train_ds_unbatched, val_ds_unbatched):
    merged = train_ds_unbatched.concatenate(val_ds_unbatched)

    parasitized_ds = merged.filter(lambda x, y: tf.equal(y[0], 0.0)).cache()
    uninfected_ds  = merged.filter(lambda x, y: tf.equal(y[0], 1.0)).cache()

    parasitized_ds = make_ds(parasitized_ds)
    uninfected_ds  = make_ds(uninfected_ds)

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        [parasitized_ds, uninfected_ds], weights=[0.5, 0.5]
    )

    return balanced_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

def get_test_ds():
    # loads the test dataset from the test subfolder
    # this is loaded already batched
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(config.directory, "Test"),
        shuffle=False,                      # no need to shuffle the test set
        seed=config.SEED,
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        label_mode='binary'                 # parasitized/uninfected
    )
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    return test_ds

def num_img_per_class(dataset):
    class_counts = collections.Counter()
    for _, labels in dataset:
        class_indices = labels.numpy().flatten().astype(int)  # (batch_size, 1) --> (batch_size,)
        class_counts.update(map(int, class_indices))
    return class_counts
    
