/* 
	###############################
    #  Author: Gabriele Trevisan  #
    ###############################
*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include "Parser.h"
#include "DatasetBuilder.h"

#define INPUT_DATASET_ARG 1
#define TRAIN_DATASET_ARG 2
#define TEST_DATASET_ARG 3

const std::string PARASITIZED_FOLDER = "/Parasitized";
const std::string UNINFECTED_FOLDER = "/Uninfected";
const float TRAINING_SET_PERCENT = 0.8;

int main(int argc, char* argv[]) {
    if(argc != 4) {
        std::cerr << "[DatasetBuilderMain] Usage: " << argv[0] << " <Dataset inputh path> <Training Dataset Output path> <Test Dataset Output path>" << std::endl;
        return -1;
    }
    const std::string DATASET_PATH = argv[INPUT_DATASET_ARG];
    const std::string DATASET_CROPPED_TRAIN_PATH = argv[TRAIN_DATASET_ARG];
    const std::string DATASET_CROPPED_TEST_PATH = argv[TEST_DATASET_ARG];

    // Check if database needs to be built
    if(!std::filesystem::exists(DATASET_CROPPED_TRAIN_PATH)) {
        std::filesystem::create_directories(DATASET_CROPPED_TRAIN_PATH);
    }
    if(!std::filesystem::exists(DATASET_CROPPED_TEST_PATH)) {
        std::filesystem::create_directories(DATASET_CROPPED_TEST_PATH);
    }
    
    if(
           !std::filesystem::exists(DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER)
    ) {
        std::cout << "[DatasetBuilderMain] Building ground truth dataset..." << std::endl;
        std::filesystem::create_directories(DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER);
        DatasetBuilder::build_cropped_dataset(
            "../" + DATASET_PATH + "/GT/", 
            "../" + DATASET_PATH + "/Img/", 
            DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER + "/",
            DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER + "/",
            DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER + "/",
            DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER + "/",
            TRAINING_SET_PERCENT
        );
        std::cout << "[DatasetBuilderMain] Ground truth dataset built to:" << std::endl;
        std::cout << "\t Train set: \"" + DATASET_CROPPED_TRAIN_PATH + "\"" << std::endl;
        std::cout << "\t Test set:  \"" + DATASET_CROPPED_TEST_PATH + "\"" << std::endl;
    }else{
        std::cout << "[DatasetBuilderMain] Ground truth dataset already exists, skipping building phase..." << std::endl;
    }
}
