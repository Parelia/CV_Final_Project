cmake_minimum_required(VERSION 3.10)
project(CV_Final_Project_Cropper)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    CropperMain.cpp
    src/Parser.cpp
    src/Utils.cpp
)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Debug
message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})