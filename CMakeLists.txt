##########################
# Author: Parussolo Elia #
##########################
cmake_minimum_required(VERSION 3.10)
project(CV_Final_Project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    DatasetBuilder/include
    Cropper/include
)

add_library(DatasetBuilderLib STATIC
    DatasetBuilder/src/DatasetBuilder.cpp
)

add_library(CropperLib STATIC
    Cropper/src/Parser.cpp
    Cropper/src/Utils.cpp
    Cropper/src/Cropper.cpp
    Cropper/src/Classifier.cpp
)


# Debug
message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")


# Executable
add_executable(DatasetBuilder
  DatasetBuilder/DatasetBuilderMain.cpp
)
target_link_libraries(DatasetBuilder
    DatasetBuilderLib
    CropperLib
    ${OpenCV_LIBS}
)

# Executable
add_executable(Cropper
  Cropper/CropperMain.cpp
)
target_link_libraries(Cropper
    CropperLib
    ${OpenCV_LIBS}
)
