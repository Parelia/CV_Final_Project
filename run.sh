#!/bin/bash
set -e # Optional: Exit immediately if a command exits with a non-zero status

echo "Starting in: $(pwd)"

# Check if the current directory name is 'build'
if [ "$(basename "$(pwd)")" = "build" ]; then
  echo "Currently in 'build', moving up..."
  cd ..
fi

echo "Now in: $(pwd)" # Should be the parent directory of 'build'

echo "Compiling..."
./compile.sh

echo "Changing to build directory..."
cd build/

echo "Running executable..."
# Note: The path to the image assumes 'build' is directly under the dir where compile.sh is.
./main 