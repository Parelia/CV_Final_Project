###########################
# Author: Capuzzo Daniele #
###########################

#### IMPORTS #### 
import os
import json
import numpy as np
import tensorflow as tf
import keras
import keras_tuner as kt

# my modules
import config
import dataset as ds 
import visualization as vis
import model

# set logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# check GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def tune_model(train_reduced, val_set):
    # [SETUP] keras tuner
    tuner = kt.Hyperband(
        hypermodel=model.model_builder,
        objective='val_accuracy',
        max_epochs=config.max_epochs,
        factor = config.factor,
        directory = config.tuned_models_dir,
        project_name=config.project_name
    )
    # set early stopping to prevent overfitting, then run procedure
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.early_stop_patience)
    tuner.search(
        train_reduced, 
        epochs=config.tuner_epochs, 
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


def main():

    #### DATASETS #########################
    print('LOADING AND PROCESSING training/test sets')
    train_set, test_set, classes = ds.load_train_test_ds()

    print('Computing datasets statistics')
    train_class_counts = ds.num_img_per_class(train_set)
    val_class_counts = ds.num_img_per_class(test_set)
    print('Training set:', train_class_counts)
    print('Test set:', val_class_counts)

    print('Class distribution before augmentation')
    vis.plot_class_distribution(train_class_counts, val_class_counts)

    print('Performing data augmentation')
    augmented_dataset = ds.get_agumented_dataset(train_set)
    train_set = train_set.concatenate(augmented_dataset)
    train_set = train_set.map(lambda x, y: (x/255, y)) # standardize here
    print('Merged train set size:', train_set.cardinality().numpy()*config.batch_size)

    # split train into train_reduced/val
    print('splitting Full training set into training and validation')
    train_size = int(train_set.cardinality().numpy() * 0.8)
    train_reduced = train_set.take(train_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_set = train_set.skip(train_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)


    #### MODEL SELECTION ####################
    # performed only if the user requires it
    response = input('Perform hyperparameter tuning [SLOW]? [y/n] ')
    if (response=='y' or response=='Y'):
        print('Running the tuning procedure')
        tune_model(train_reduced, val_set)

    
    #### TRAIN THE BEST MODEL ####################
    # [LOADING] the hyperparameters of the best model (avoids re-optimizing)
    print('Training the model with the best_config.json hparams')
    save_path = os.path.join(config.save_directory, 'best_config.json')
    with open(save_path, 'r') as f:
        loaded_hp_dict = json.load(f)
    hp = kt.HyperParameters()
    for key, value in loaded_hp_dict.items():
        hp.Fixed(key, value) # treat it as if it was an optimizer choice
    # load the fresh model
    best_model = model.model_builder(hp)
    print('Parameters loaded')
    best_model.summary()

    # [PRE-TRAINING]
    print('Pre-training with early stopping')
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
    vis.plot_learning_curves(history)

    # [TRAINING]
    # first determine the optimal number of epochs
    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1 # due to indexing from 0
    print(f"Best epoch found: {best_epoch}")

    # create fresh model and retrain it on the full train set
    print('RE-Training on the entire training set for ', best_epoch, ' epochs')
    best_model = model.model_builder(hp)
    history = best_model.fit(
        train_set,
        epochs=best_epoch
    )

    save_path = os.path.join(config.save_directory, 'malaria_det.keras')
    if not os.path.exists(config.save_directory):
        print(f"Directory {config.save_directory} not found. Creating it...")
        os.makedirs(config.save_directory, exist_ok=True)
    keras.saving.save_model(best_model, save_path)
    print(f'trained model saved at {save_path}')

    if (config.evaluate) :
        # testing the model
        test_set = test_set.map(lambda x, y: (x/255, y))
        test_set = test_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        loss, accuracy = best_model.evaluate(test_set, verbose='0')
        print(f'Best model test loss: {loss:.4f}')
        print(f'Best model test accuracy: {accuracy:.4f}')


        # full classification report with precision, recall, F1-score
        y_true = []
        y_probs = []
        for x_batch, y_batch in test_set:
            preds = best_model.predict(x_batch, verbose='0').ravel()
            y_probs.extend(preds)
            y_true.extend(y_batch.numpy())

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = (y_probs > 0.5).astype(int) # uninfected or parasitized
        vis.classification_report(y_true, y_pred)

        # ROC curve and AUC-ROC parameter
        vis.plot_roc(y_true, y_probs)
        
        # confusion matrix
        vis.plot_cm(y_true, y_pred)


if __name__ == "__main__":
    main()