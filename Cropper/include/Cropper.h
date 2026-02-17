/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#ifndef Cropper_h
#define Cropper_h

#include <iostream>
#include <vector>
#include <opencv2/core/types.hpp>


namespace Cropper {
    std::vector<cv::Rect> crop_image(const std::string& input_path);
}

#endif
