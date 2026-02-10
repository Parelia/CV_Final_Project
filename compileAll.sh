cd Classifier/DataSetBuilder
echo "Cleaning build files..."
rm -rf build

echo "Compiling DataSetBuilder..."
./compile.sh

cd ../..

cd Cropper
echo "Cleaning build files..."
rm -rf build

echo "Compiling Cropper..."
./compile.sh

cd ..