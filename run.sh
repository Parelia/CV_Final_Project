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
./CV_Final_Project "IMG_20150621_112023" "../Malaria_Dataset_Cropped/"

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
  "IMG_20150724_102054"
  "IMG_20150724_102233"
  "IMG_20150725_123938"
  "IMG_20150725_123606"
  "IMG_20150725_123830"
  "IMG_20150725_123213"
  "IMG_20150725_123847"
)

#for img in "${img_list[@]}"; do
  #./CV_Final_Project $img "Malaria_Dataset_Cropped/"
#done