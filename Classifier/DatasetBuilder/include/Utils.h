/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#ifndef Utils_h
#define Utils_h

#include <opencv2/imgproc.hpp>
#include <opencv2/flann.hpp>

namespace Utils {
    // Removes black circular frame from image
    void cut_background(const cv::Mat&, cv::Mat&, cv::Point&);

    // Applies thresholding and morphological operations to clean up elements
    void refined_threshold(const cv::Mat&, cv::Mat&);

    // Compute IoU values over ground truth
    std::vector<double> compute_IoU(const std::vector<cv::Rect>&, const std::vector<cv::Rect>&);

    // Standard IoU computation
    double IoU(const cv::Rect&, const cv::Rect&);
};

#endif
