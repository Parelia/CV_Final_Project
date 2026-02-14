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

if [ "$1" = "--no-compile" ]; then
  echo "Skipping compilation."
else
  echo "Compiling..."
  ./compile.sh
fi

echo "Changing to build directory..."
cd build/


IMG_DIR="../../Classifier/DatasetBuilder/Malaria_Dataset/Img"
PY_SCRIPT="../../Classifier/PyClassifier/classifier.py"

# Check if the directory exists
if [ ! -d "$IMG_DIR" ]; then
  echo "Error: Directory $IMG_DIR not found."
  exit 1
fi

for img in "$IMG_DIR"/*.jpg; do
    [ -e "$img" ] || continue
    
    echo "------------------------------------------"
    echo "Processing: $(basename "$img")"
    
    ./CV_Final_Project_Cropper "$img" "$PY_SCRIPT" --quiet
done

echo "------------------------------------------"

exit 0
