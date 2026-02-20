###########################
# Author: Capuzzo Daniele #
###########################

import os
import sys
import numpy as np
import keras
import config
import dataset as ds
import visualization as vis
import tensorflow as tf

USE_TTA = True
N_AUGMENTATIONS = 5
DEF_THRESHOLD = 0.5

def predict_with_tta(model, dataset, n_augmentations=5, threshold=0.5):
    all_probs = []
    
    for _ in range(n_augmentations):
        probs = []
        for images, _ in dataset:
            # apply random augmentations manually
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)
            batch_probs = model(images, training=False)
            probs.append(batch_probs.numpy())
        all_probs.append(np.concatenate(probs))
    
    # average across augmentations
    avg_probs = np.mean(all_probs, axis=0).ravel()
    return avg_probs

def main():
    # optional threshold from command line argument
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else DEF_THRESHOLD
    print(f'Evaluating model with threshold: {threshold}')

    # load model
    model_name = sys.argv[2]  if len(sys.argv) > 2 else 'malaria_det.keras'
    cnn = keras.models.load_model(os.path.join(config.save_directory, model_name))
    cnn.summary()

    # load test set
    test_set = ds.get_test_ds()

    # predictions
    if (USE_TTA):
         y_probs = predict_with_tta(cnn, test_set, n_augmentations=5, threshold=threshold)
    else:
        y_probs = cnn.predict(test_set).ravel()

    y_true  = np.concatenate([y for x, y in test_set], axis=0)
    y_pred  = (y_probs > threshold).astype(int)

    # evaluation
    vis.classification_report(y_true, y_pred)
    vis.plot_prediction_samples(test_set, y_true, y_pred, y_probs)
    vis.plot_roc(y_true, y_probs)
    vis.plot_cm(y_true, y_pred)


if __name__ == "__main__":
    main()