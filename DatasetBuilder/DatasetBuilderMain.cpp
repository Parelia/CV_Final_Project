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
#define VAL_DATASET_ARG 3
#define CROPPERTEST_DATASET_ARG 4

const std::string PARASITIZED_FOLDER = "/Parasitized";
const std::string UNINFECTED_FOLDER = "/Uninfected";
const float TRAINING_SET_PERCENT = 14.0f / 17.0f;   // we want about 14 out of 
                                                    // 17 images for training

int main(int argc, char* argv[]) {
    if(argc != 5) {
        std::cerr << "[DatasetBuilderMain] Usage: " << argv[0] << " <Dataset inputh path> <Training Dataset Output path> <Validation Dataset Output path> <Test Dataset Output path>" << std::endl;
        return -1;
    }
    const std::string DATASET_PATH = argv[INPUT_DATASET_ARG];
    const std::string DATASET_CROPPED_TRAIN_PATH = argv[TRAIN_DATASET_ARG];
    const std::string DATASET_CROPPED_VAL_PATH = argv[VAL_DATASET_ARG];

    // separate test set folder
    const std::string DATASET_CROPPED_TEST_PATH = argv[CROPPERTEST_DATASET_ARG];

    // Check if database needs to be built
    if(!std::filesystem::exists(DATASET_CROPPED_TRAIN_PATH)) {  // training
        std::filesystem::create_directories(DATASET_CROPPED_TRAIN_PATH);
    }
    if(!std::filesystem::exists(DATASET_CROPPED_VAL_PATH)) {    // validation
        std::filesystem::create_directories(DATASET_CROPPED_VAL_PATH);
    }
    if(!std::filesystem::exists(DATASET_CROPPED_TEST_PATH)) {   // test/inference set
        std::filesystem::create_directories(DATASET_CROPPED_TEST_PATH);
    }
    
    if(
           !std::filesystem::exists(DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_VAL_PATH + PARASITIZED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_VAL_PATH + PARASITIZED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_VAL_PATH + UNINFECTED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_VAL_PATH + UNINFECTED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER) || std::filesystem::is_empty(DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER)
        || !std::filesystem::exists(DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER)  || std::filesystem::is_empty(DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER)
    ) {
        std::cout << "[DatasetBuilderMain] Building ground truth dataset..." << std::endl;
        std::filesystem::create_directories(DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_VAL_PATH + PARASITIZED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_VAL_PATH + UNINFECTED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER);
        std::filesystem::create_directories(DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER);
        DatasetBuilder::build_cropped_dataset(
            "../" + DATASET_PATH + "/GT/",          // common ground truths
            "../" + DATASET_PATH + "/Img/",         // training/validation
            "../" + DATASET_PATH + "/CropperTest/Img", // inference/test set 
            DATASET_CROPPED_TRAIN_PATH + PARASITIZED_FOLDER + "/",
            DATASET_CROPPED_TRAIN_PATH + UNINFECTED_FOLDER + "/",
            DATASET_CROPPED_VAL_PATH + PARASITIZED_FOLDER + "/",
            DATASET_CROPPED_VAL_PATH + UNINFECTED_FOLDER + "/",
            DATASET_CROPPED_TEST_PATH + PARASITIZED_FOLDER + "/",
            DATASET_CROPPED_TEST_PATH + UNINFECTED_FOLDER + "/",
            TRAINING_SET_PERCENT
        );
        std::cout << "[DatasetBuilderMain] Ground truth dataset built to:" << std::endl;
        std::cout << "\t Train set: \"" + DATASET_CROPPED_TRAIN_PATH + "\"" << std::endl;
        std::cout << "\t Validation set:  \"" + DATASET_CROPPED_VAL_PATH + "\"" << std::endl;
        std::cout << "\t Test set:  \"" + DATASET_CROPPED_TEST_PATH + "\"" << std::endl;
    }else{
        std::cout << "[DatasetBuilderMain] Ground truth dataset already exists, skipping building phase..." << std::endl;
    }
}
