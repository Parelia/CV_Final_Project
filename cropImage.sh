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


IMG_DIR="../Malaria_Dataset/Img"
DNN_MODEL="../PyClassifier/saved_model/malaria_det.onnx"

# Check if the directory exists
if [ ! -d "$IMG_DIR" ]; then
  echo "Error: Directory $IMG_DIR not found."
  exit 1
fi

./Cropper "$IMG_DIR" "$DNN_MODEL" --quiet

echo "------------------------------------------"

exit 0
