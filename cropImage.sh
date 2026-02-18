###############################
#  Author: Gabriele Trevisan  #
###############################

#!/bin/bash
set -e # Optional: Exit immediately if a command exits with a non-zero status

echo "Starting in: $(pwd)"

# Check if the current directory name is 'build'
if [ "$(basename "$(pwd)")" = "build" ]; then
  echo "Currently in 'build', moving up..."
  cd ..
  echo "Now in: $(pwd)" # Should be the parent directory of 'build'
fi

echo "Changing to build directory..."
cd build/


IMG_DIR="../Malaria_Dataset"

# Check if the directory exists
if [ ! -d "$IMG_DIR" ]; then
  echo "Error: Directory $IMG_DIR not found."
  exit 1
fi
./Cropper "$IMG_DIR" --IoU


TEST_IMG_DIR="../Malaria_Dataset/CropperTest/Img"
DNN_MODEL="../PyClassifier/saved_model/malaria_det.onnx"

# Check if the directory exists
if [ ! -d "$TEST_IMG_DIR" ]; then
  echo "Error: Directory $TEST_IMG_DIR not found."
  exit 1
fi

# ./Cropper "$TEST_IMG_DIR" "$DNN_MODEL"

echo "------------------------------------------"

exit 0
