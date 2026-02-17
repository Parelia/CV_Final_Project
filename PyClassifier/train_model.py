###########################
# Author: Capuzzo Daniele #
###########################

#### IMPORTS #### 
# dataset
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import pandas as pd
import collections

import config
import json

# model
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import optimizers
import keras_tuner as kt

# graphs/stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

# check GPU availability
from tensorflow.python.client import device_lib
tf.config.list_physical_devices('GPU')


def load_train_test_ds():
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
    classes = train_set.class_names
    return train_set, test_set, classes


def num_img_per_class(dataset):
    class_counts = collections.Counter()
    for _, labels in dataset:
        class_indices = labels.numpy().squeeze().astype(int)  # (batch_size, 1) → (batch_size,)
        class_counts.update(map(int, class_indices))
    return class_counts

def plot_class_distribution(train_class_counts, val_class_counts):
    # bar diagram of training and validation classes distribution
    labels_map = {0: 'Parasitized', 1: 'Uninfected'}

    # Prepare data
    x_labels = [labels_map[i] for i in sorted(labels_map.keys())]
    x_pos = range(len(x_labels))
    train_vals = [train_class_counts[i] for i in sorted(labels_map.keys())]
    val_vals = [val_class_counts[i] for i in sorted(labels_map.keys())]

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot bars
    bar_width = 0.35
    bars1 = plt.bar([i - bar_width/2 for i in x_pos], train_vals, width=bar_width, label='Training', color='skyblue')
    bars2 = plt.bar([i + bar_width/2 for i in x_pos], val_vals, width=bar_width, label='Test', color='orange')

    # Add counts on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height}', ha='center', va='bottom', fontsize=9)

    # final touches
    plt.xticks(ticks=x_pos, labels=x_labels)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Training vs Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

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


def main():

    #### DATASETS #########################
    print('Loading the training/test sets')
    train_set, test_set, classes = load_train_test_ds()

    print('Computing datasets statistics')
    train_class_counts = num_img_per_class(train_set)
    val_class_counts = num_img_per_class(test_set)
    print('Training set:', train_class_counts)
    print('Test set:', val_class_counts)

    plot_class_distribution(train_class_counts, val_class_counts)

    # [DATA AUGM.] to enlarge the dataset
    preprocessing = keras.Sequential([
        layers.RandomRotation(factor=0.2),
        layers.RandomFlip(mode='horizontal_and_vertical'),
        layers.RandomBrightness(factor=0.15),
        layers.RandomContrast(factor=0.15),
        layers.GaussianNoise(stddev=0.05),
        #  layers.Rescaling(1./255), # data standardization NOT HERE because this would apply only to augmented
    ])
    augmented_dataset = train_set.map(lambda x, y: (preprocessing(x), y))

    # concatenate the two datasets to form a big one
    train_set = train_set.concatenate(augmented_dataset)
    print('Augmented train set size:', augmented_dataset.cardinality().numpy()*config.batch_size)
    print('Merged train set size:', train_set.cardinality().numpy()*config.batch_size)
    train_set = train_set.map(lambda x, y: (x/255, y)) # standardize here


    #### MODEL SELECTION ####################
    # split train into train_reduced/val
    print('SPLITTING Full training set into training and validation')
    train_size = int(train_set.cardinality().numpy() * 0.8)
    train_reduced = train_set.take(train_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_set = train_set.skip(train_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # [SETUP] keras tuner
    tuner = kt.Hyperband(
        hypermodel=model_builder,
        objective='val_accuracy',
        max_epochs=6,
        factor = 3,
        directory = 'tuned_models',
        project_name='malaria_det'
    )
    # set early stopping to prevent overfitting, then run procedure
    print('SEARCHING for the best model using KerasTuner and Hyperband algorithm')
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    tuner.search(train_reduced, 
        epochs=100, 
        validation_data=val_set, 
        callbacks=[stop_early]
    )

    # [SAVING] the obtained hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    hp_dict = best_hps.values

    # saving them
    if not os.path.exists(config.save_directory):
        print(f"Directory {config.save_directory} not found. Creating it...")
        os.makedirs(config.save_directory, exist_ok=True)
    save_path = os.path.join(config.save_directory, 'best_config.json')
    print(f"SAVING the best hyperparameters at: {save_path}")
    with open(save_path, 'w') as f:
        json.dump(hp_dict, f, indent=4)

    print(f"Successfully saved best hyperparameters to: {save_path}")



    #### TRAIN THE BEST MODEL ####################
    # [LOADING] the hyperparameters of the best model (avoids re-optimizing)
    save_path = os.path.join(config.save_directory, 'best_config.json')
    with open(save_path, 'r') as f:
        loaded_hp_dict = json.load(f)
    hp = kt.HyperParameters()
    for key, value in loaded_hp_dict.items():
        hp.Fixed(key, value) # treat it as if it was an optimizer choice
    # load the fresh model
    best_model = model_builder(hp)
    best_model.summary()

    # [PRE-TRAINING]
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=config.patience_train, 
        restore_best_weights=True
    )
    history = best_model.fit(
        train_reduced,
        epochs=config.epochs_train,
        validation_data=val_set,
        callbacks=[stop_early]
    )
    plot_learning_curves(history)

    # [TRAINING]
    # first determine the optimal number of epochs
    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1 # due to indexing from 0
    print(f"Best epoch found: {best_epoch}")

    # create fresh model and retrain it on the full train set
    best_model = model_builder(hp)
    history = best_model.fit(
        train_set,
        epochs=best_epoch
    )

    save_path = os.path.join(config.save_directory, 'malaria_det.keras')
    if not os.path.exists(config.save_directory):
        print(f"Directory {config.save_directory} not found. Creating it...")
        os.makedirs(config.save_directory, exist_ok=True)
    print(f'SAVING the best model at {save_path}')
    keras.saving.save_model(best_model, save_path)

    if (config.evaluate) :
        # testing the model
        test_set = test_set.map(lambda x, y: (x/255, y))
        test_set = test_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        loss, accuracy = best_model.evaluate(test_set, verbose=0)
        print(f'Best model test loss: {loss:.4f}')
        print(f'Best model test accuracy: {accuracy:.4f}')


        # full classification report with precision, recall, F1-score
        y_true = []
        y_probs = []
        for x_batch, y_batch in test_set:
            preds = best_model.predict(x_batch, verbose=0).ravel()
            y_probs.extend(preds)
            y_true.extend(y_batch.numpy())

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = (y_probs > 0.5).astype(int) # uninfected or parasitized
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=['Uninfected', 'Parasitized']))

        # ROC curve and AUC-ROC parameter
        print(f"AUC-ROC: {roc_auc_score(y_true, y_probs):.4f}")
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_probs):.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Uninfected', 'Parasitized'],
                    yticklabels=['Uninfected', 'Parasitized'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()


if __name__ == "__main__":
    main()