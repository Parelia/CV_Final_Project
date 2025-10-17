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
    void cut_background(const cv::Mat&, cv::Mat&, cv::Point&);

    void refined_threshold(const cv::Mat&, cv::Mat&);

    std::vector<double> compute_IoU(const std::vector<cv::Rect>&, const std::vector<cv::Rect>&);

    double IoU(const cv::Rect&, const cv::Rect&);
};


#endif
