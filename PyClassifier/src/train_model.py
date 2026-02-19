###########################
# Author: Capuzzo Daniele #
###########################

#### IMPORTS #### 
from operator import neg
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

from sklearn.metrics import precision_recall_curve, classification_report

# set logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# check GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def tune_model(train_set, val_set, steps_per_epoch):
    builder = model.make_builder()

    # [SETUP] keras tuner
    tuner = kt.Hyperband(
        hypermodel=builder,
        objective='val_loss',
        max_epochs=config.max_epochs,
        factor = config.factor,
        directory = config.tuned_models_dir,
        project_name=config.project_name
    )

    # set early stopping to prune trials early
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min',
        patience=config.early_stop_patience
    )

    # run the procedure
    tuner.search(
        train_set,
        epochs=config.tuner_epochs,
        validation_data=val_set,
        callbacks=[stop_early],
        steps_per_epoch=steps_per_epoch
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


def get_optimal_decision_threshold(val_probs, val_true):

    # compute the f1-score
    precision, recall, thresholds = precision_recall_curve(val_true, val_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # find threshold with highest F1 score
    max_f1_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[max_f1_index]

    return optimal_threshold


def main():

    #### DATASETS #########################
    print('LOADING TRAINING/VAL DATASETS')
    train_ds_unbatched, val_ds_unbatched = ds.get_unbalanced_ds(batch_size=None)
    train_set, val_set, neg, pos = ds.prepare_and_balance(train_ds_unbatched, val_ds_unbatched)

    # determine steps per epoch
    print(f"Training dataset:\n  - Parasitized: {neg}\n  - Uninfected: {pos}")
    majority_count = max(neg, pos)
    steps_per_epoch = (majority_count * 2) // config.batch_size
    print(f"Steps per epoch (after balancing): {steps_per_epoch}")
    
    #### TRAIN THE BEST MODEL ####################
    # [LOADING] the hyperparameters from json file
    print('Training the model with the best_config.json hparams')
    save_path = os.path.join(config.save_directory, 'best_config.json')
    with open(save_path, 'r') as f:
        loaded_hp_dict = json.load(f)
    hp = kt.HyperParameters()
    for key, value in loaded_hp_dict.items():
        hp.Fixed(key, value) # treat it as if it was an optimizer choice
    

    # MODEL FACTORY build model
    builder = model.make_builder()
    cnn = builder(hp)
    cnn.summary()

    # [CALLBACKS]
    print('Pre-training with early stopping')
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=config.patience_train, 
        restore_best_weights=True
    )

    # [TRAINING]
    history = cnn.fit(
        train_set,
        epochs=config.epochs_train,
        validation_data=val_set,
        callbacks=[stop_early],
        steps_per_epoch=steps_per_epoch
    )
    vis.plot_learning_curves(history)

    # save the resulting model
    save_path = os.path.join(config.save_directory, 'malaria_det.keras')
    if not os.path.exists(config.save_directory):
        print(f"Directory {config.save_directory} not found. Creating it...")
        os.makedirs(config.save_directory, exist_ok=True)
    keras.saving.save_model(cnn, save_path)
    print(f'PRETRAINED model saved at {save_path}')

    print('Validation set report before optimal threshold')
    val_probs = cnn.predict(val_set, verbose=0).ravel()
    val_true = np.concatenate([y for x, y in val_set], axis=0)
    print(classification_report(val_true, (val_probs > 0.5).astype(int)))

    print('Validation set report AFTER optimal threshold')
    optimal_threshold = get_optimal_decision_threshold(val_probs, val_true)
    print(f'OPTIMAL THRESHOLD {optimal_threshold}')
    print(classification_report(val_true, (val_probs >optimal_threshold).astype(int)))
    
    # first determine the optimal number of epochs
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Early stopping triggered at: {best_epoch}")


if __name__ == "__main__":
    main()