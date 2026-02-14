###############################
#  Author: Gabriele Trevisan  #
###############################

#! /bin/bash

train_cnn=$1

if [ "$train_cnn" != "train" ]; then
    echo "To run the training process, please use the following command:"
    echo "$0 train"
    echo "Please be aware that the training process may take a long time ~2.5 hours."
    echo "Continuing without training the CNN using the pre-trained model..."
fi

# Compile all the necessary components
./compileAll.sh
if [ "$train_cnn" = "train" ]; then
    echo "Building the dataset..."
    cd Classifier/DatasetBuilder
    ./run.sh --no-compile
    cd ../PyClassifier
    echo "Training the CNN..."
    python3 cv_malaria_detection.py
    cd ../..
fi
cd Cropper

echo "Running the cropper..."
./run.sh --no-compile