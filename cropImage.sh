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
PY_SCRIPT="../PyClassifier/classifier.py"

# Check if the directory exists
if [ ! -d "$IMG_DIR" ]; then
  echo "Error: Directory $IMG_DIR not found."
  exit 1
fi

for img in "$IMG_DIR"/*.jpg; do
    [ -e "$img" ] || continue
    
    echo "------------------------------------------"
    echo "Processing: $(basename "$img")"
    
    ./Cropper "$img" "$PY_SCRIPT" --quiet
done

echo "------------------------------------------"

exit 0
