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

echo "Compiling..."
./compile.sh

echo "Changing to build directory..."
cd build/

echo "Running executable..." # CMake project name (exe)
./CV_Final_Project "IMG_20150621_112023"

img_list=(
  "IMG_20150621_112023"
  "IMG_20150621_112043"
  "IMG_20150621_112116"
  "IMG_20150621_112138"
  "IMG_20150621_112227"
  "IMG_20150714_093512"
  "IMG_20150714_093600"
  "IMG_20150714_093636"
  "IMG_20150714_093715"
  "IMG_20150714_093806"
  "IMG_20150724_102330"
  "IMG_20150724_102843"
  "IMG_20150724_102951"
  "IMG_20150724_103054"
  "IMG_20150724_103233"
  "IMG_20150725_123938"
  "IMG_20150725_124606"
  "IMG_20150725_124830"
  "IMG_20150725_125213"
  "IMG_20150725_125847"
)

# for img in "${img_list[@]}"; do
#   ./CV_Final_Project $img "Malaria_Dataset_Cropped/"
# done

echo "Changing to main directory..."
cd ..
echo "Compressing dataset..."
tar -czf Malaria_Dataset_Cropped.tar.gz Malaria_Dataset_Cropped
#zip -r  test.zip Malaria_Dataset_Cropped

exit
