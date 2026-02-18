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

# set logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# check GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def tune_model(train_set, val_set, pos, neg, weights):
    builder = model.make_builder(pos, neg)

    # [SETUP] keras tuner
    tuner = kt.Hyperband(
        hypermodel=builder,
        objective='val_accuracy',
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
        class_weight=weights,
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
    train_set, val_set, test_set, class_names = ds.get_datasets()

    print('Computing datasets statistics')
    train_class_counts = ds.num_img_per_class(train_set)
    val_class_counts = ds.num_img_per_class(val_set)
    test_class_counts = ds.num_img_per_class(test_set)
    mapping = {i: name for i, name in enumerate(class_names)}
    print(f"  Classes: {(mapping)}")
    print('  Training set:', train_class_counts)
    print('  Validation set:', val_class_counts)
    print('  Test set:', test_class_counts)

    # OPTIONAL: plot class distribution
    vis.plot_class_distribution(train_class_counts, val_class_counts)

    # [HANDLING IMBALANCE] through class weights
    neg = train_class_counts[0]  # 0
    pos = train_class_counts[1]  # 1
    total = pos + neg
    w0 = (1 / neg) * (total / 2.0)
    w1 = (1 / pos) * (total / 2.0)
    weights = {0: w0, 1: w1}
    print(f"Computed weights: Class 0: {w0:.2f}, Class 1: {w1:.2f}")


    #### MODEL SELECTION ####################
    # performed only if the user requires it
    # response = input('[WARNING: SLOW] Perform hyperparameter tuning? [y/n] ')
    # if (response=='y' or response=='Y'):
    #     tune_model(train_set, val_set, pos, neg, weights)

    
    #### TRAIN THE BEST MODEL ####################
    # [LOADING] the hyperparameters of the best model (avoids re-optimizing)
    print('Training the model with the best_config.json hparams')
    save_path = os.path.join(config.save_directory, 'best_config.json')
    with open(save_path, 'r') as f:
        loaded_hp_dict = json.load(f)
    hp = kt.HyperParameters()
    for key, value in loaded_hp_dict.items():
        hp.Fixed(key, value) # treat it as if it was an optimizer choice
    
    # MODEL FACTORY
    builder = model.make_builder(pos, neg)

    # # load a fresh model
    # best_model = builder(hp)
    # print('Parameters loaded')
    # best_model.summary()

    # # [CALLBACKS] to account for a small validation set!!!
    # print('Pre-training with early stopping')
    # stop_early = keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     mode='min',
    #     patience=config.patience_train, 
    #     restore_best_weights=True
    # )

    # reduce_lr = keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,
    #     patience=3,
    #     min_lr=1e-6,
    #     verbose=1
    # )

    # # [PRE-TRAINING]
    # history = best_model.fit(
    #     train_set,
    #     epochs=config.epochs_train,
    #     validation_data=val_set,
    #     class_weight=weights,
    #     callbacks=[stop_early, reduce_lr]
    # )
    # vis.plot_learning_curves(history)

    # # [TRAINING]
    # # first determine the optimal number of epochs
    # metric_per_epoch = history.history['val_loss']
    # best_epoch = metric_per_epoch.index(min(metric_per_epoch)) + 1 # due to indexing from 0
    # print(f"Best epoch found: {best_epoch}")

    # # create fresh model and retrain it on the full train set
    # print('RE-Training on the entire training set for ', best_epoch, ' epochs')
    # best_model = builder(hp)
    # history = best_model.fit(
    #     train_set.concatenate(val_set),
    #     epochs=best_epoch,
    #     class_weight=weights,
    #     callbacks=[reduce_lr]
    # )

    # # save the resulting model
    # save_path = os.path.join(config.save_directory, 'malaria_det.keras')
    # if not os.path.exists(config.save_directory):
    #     print(f"Directory {config.save_directory} not found. Creating it...")
    #     os.makedirs(config.save_directory, exist_ok=True)
    # keras.saving.save_model(best_model, save_path)
    # print(f'trained model saved at {save_path}')

    # TODO: load it from the saved file and move this part to classifier.py
    best_model = keras.models.load_model(os.path.join(config.save_directory, 'malaria_det.keras'))

    # evaluate only if required
    if (config.evaluate) :
        # testing the model
        # test_set = test_set.map(lambda x, y: (x/255, y)) # NOT NEEDED ANYMORE

        test_set = test_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        loss, accuracy, auc = best_model.evaluate(test_set, verbose='0')
        print(f'Best model test loss: {loss:.4f}')
        print(f'Best model test accuracy: {accuracy:.4f}')
        print(f'Best model test AUC-ROC: {auc:.4f}')

        # make sure the folder exists
        if not os.path.exists(config.image_directory):
            print(f"Directory {config.image_directory} not found. Creating it...")

        # prediction
        y_probs = best_model.predict(test_set).ravel()
        y_true = np.concatenate([y for x, y in test_set], axis=0)
        y_pred = (y_probs > 0.5).astype(int)

        # full classification report with precision, recall, F1-score
        vis.classification_report(y_true, y_pred)

        # ROC curve and AUC-ROC parameter
        vis.plot_roc(y_true, y_probs)
        
        # confusion matrix
        vis.plot_cm(y_true, y_pred)


if __name__ == "__main__":
    main()