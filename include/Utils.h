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
    void cut_circle_borders(const cv::Mat&, cv::Mat&, cv::Point&);

    void refined_threshold(const cv::Mat&, cv::Mat&);

    void save_boxes(std::vector<cv::Rect>&, const std::vector<cv::Vec3f>&, const cv::Mat&, const cv::Point&);
    void save_boxes(std::vector<cv::Rect>&, const std::vector<cv::Vec3f>&, const cv::Mat&);

    void clean_boxes(std::vector<cv::Rect>&, int);

    std::vector<double> compute_IoU(const std::vector<cv::Rect>&, const std::vector<cv::Rect>&);

    double IoU(const cv::Rect&, const cv::Rect&);

    double Rect_distance(const cv::Rect&, const cv::Rect&);
};


#endif