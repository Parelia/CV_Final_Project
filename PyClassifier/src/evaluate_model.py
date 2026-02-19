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

def main():
    # optional threshold from command line argument
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    print(f'Evaluating model with threshold: {threshold}')

    # load model
    model_name = sys.argv[2]  if len(sys.argv) > 2 else 'malaria_det.keras'
    cnn = keras.models.load_model(os.path.join(config.save_directory, model_name))

    # load test set
    test_set = ds.get_test_ds()

    # predictions
    y_probs = cnn.predict(test_set).ravel()
    y_true  = np.concatenate([y for x, y in test_set], axis=0)
    y_pred  = (y_probs > threshold).astype(int)

    # evaluation
    vis.classification_report(y_true, y_pred)
    vis.plot_roc(y_true, y_probs)
    vis.plot_cm(y_true, y_pred)

if __name__ == "__main__":
    main()