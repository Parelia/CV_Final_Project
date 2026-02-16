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

#define IMG_ARG 1


int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cerr << "[DatasetBuilderMain] Usage: " << argv[0] << " <Dataset path> <Output path>" << std::endl;
        return -1;
    }
    const std::string DATASET_PATH = argv[1];
    const std::string DATASET_CROPPED_PATH = argv[2];
    const std::string DATSET_CROPPED_PARASITIZED_PATH = DATASET_CROPPED_PATH + "/Parasitized";
    const std::string DATSET_CROPPED_UNINFECTED_PATH = DATASET_CROPPED_PATH + "/Uninfected";

    // Check if database needs to be built
    if(!std::filesystem::exists(DATASET_CROPPED_PATH)) {
        std::filesystem::create_directories(DATASET_CROPPED_PATH);
    }
    if(
        !std::filesystem::exists(DATSET_CROPPED_PARASITIZED_PATH) || std::filesystem::is_empty(DATSET_CROPPED_PARASITIZED_PATH)
        || !std::filesystem::exists(DATSET_CROPPED_UNINFECTED_PATH) || std::filesystem::is_empty(DATSET_CROPPED_UNINFECTED_PATH)
    ) {
        std::cout << "[DatasetBuilderMain] Building ground truth dataset..." << std::endl;
        std::filesystem::create_directories(DATSET_CROPPED_PARASITIZED_PATH);
        std::filesystem::create_directories(DATSET_CROPPED_UNINFECTED_PATH);
        DatasetBuilder::build_cropped_dataset(
            "../" + DATASET_PATH + "/GT/", 
            "../" + DATASET_PATH + "/Img/", 
            DATSET_CROPPED_PARASITIZED_PATH + "/", 
            DATSET_CROPPED_UNINFECTED_PATH + "/"
        );
        std::cout << "[DatasetBuilderMain] Ground truth dataset built to \"" << DATASET_CROPPED_PATH << "\"" << std::endl;
    }else{
        std::cout << "[DatasetBuilderMain] Ground truth dataset already exists, skipping building phase..." << std::endl;
    }
}
