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

// Transforms enum
enum class Transform {
    FLIPS,
    ROTATIONS_0_90,
    ELASTIC_DEFORMATIONS
};

namespace DatasetBuilder {
    void build_cropped_dataset(std::string, std::string, std::string, std::string);

    void apply_transform(std::string inputImagePath, Transform transformType);

};


#endif
