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

echo "Running executable..." # CMake project name (exe)
./CV_Final_Project Malaria_Dataset ../Malaria_Dataset_FromGT # ../ to exit build

exit
