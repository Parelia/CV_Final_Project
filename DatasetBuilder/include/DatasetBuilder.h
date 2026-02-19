/* 
	###############################
    #  Author: Gabriele Trevisan  #
    ###############################
*/

#ifndef DatasetBuilder_h
#define DatasetBuilder_h

#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

// Transforms enum
enum class Transform {
    FLIPS,
    ROTATIONS_0_90,
    ELASTIC_DEFORMATIONS
};

namespace DatasetBuilder {

    void build_cropped_dataset(
        std::string groundThruth,
        std::string ImagesFolder,
        std::string cropperTestFolder,      // here separately the images for testing
        std::string parasitizedOutputTrain,
        std::string uninfectedOutputTrain,
        std::string parasitizedOutputVal,
        std::string uninfectedOutputVal,
        std::string parasitizedOutputTest,
        std::string uninfectedOutputTest,
        float trainSetPercentage
    );

    void apply_transform(std::string inputImagePath, Transform transformType);

    void process_folder(
        const std::string& imagesFolder,
        const std::string& groundTruthFolder,
        const std::string& parasitizedOutputA,  // train or test
        const std::string& uninfectedOutputA,
        const std::string& parasitizedOutputB,  // val or empty
        const std::string& uninfectedOutputB,
        std::mt19937& prng,
        float probabilityA                      // probability of going to A
    );

};


#endif
