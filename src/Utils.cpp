/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#include "Utils.h"

#include <opencv2/highgui.hpp>

void Utils::cut_background(const cv::Mat& src, cv::Mat& dst, cv::Point& off) {
    // Mask for non black pixel
    if(src.channels() == 1) return;
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask = gray > 10;

    // Find main circle
    std::vector<cv::Vec3f> circle;
    cv::HoughCircles(mask, circle, cv::HOUGH_GRADIENT, 1,
                1300,           // min dist btw centers
                100, 12,        // param1 = Canny threshold, param2 = accumulator threshold
                1300, 1800);    // min max radius

    int minX = std::max(static_cast<int>(std::round(circle[0][0] - circle[0][2])), 0);
    int minY = std::max(static_cast<int>(std::round(circle[0][1] - circle[0][2])), 0);
    int maxX = std::min(static_cast<int>(std::round(circle[0][0] + circle[0][2])), mask.cols);
    int maxY = std::min(static_cast<int>(std::round(circle[0][1] + circle[0][2])), mask.rows);
    int width = maxX - minX;
    int height = maxY - minY;

    // Cut area
    dst = src(cv::Rect(minX, minY, width, height)).clone();
    off.x = minX;
    off.y = minY;

    return;
}

void Utils::refined_threshold(const cv::Mat& src, cv::Mat& dst) {
    if(src.channels() == 1) return;

    cv::Mat temp_dst;
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask = gray <= 10;
    
    // ADJUST PARAMETERS!!!!
    cv::threshold(gray, temp_dst, 155, 255, cv::THRESH_BINARY_INV);
    cv::subtract(temp_dst, mask, temp_dst);

    cv::morphologyEx(temp_dst, temp_dst, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)), cv::Point(-1,-1), 4);
    cv::morphologyEx(temp_dst, temp_dst, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)), cv::Point(-1,-1), 4);
    
    cv::erode(temp_dst, temp_dst, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5)), cv::Point(-1,-1), 2); 

    // -------------------------------------------------------------------------------------------
    cv::Mat temp, mask_2;
    cv::bitwise_not(temp_dst, temp_dst);
    cv::connectedComponents(temp_dst, temp);

    temp.convertTo(mask_2, CV_8U);
    mask_2.setTo(255, mask_2 > 1);
    cv::subtract(temp_dst, mask_2, temp_dst);
    // -------------------------------------------------------------------------------------------

    cv::bitwise_not(temp_dst, temp_dst);
    cv::erode(temp_dst, temp_dst, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1,-1), 4);

    dst = temp_dst.clone();
    return;
}


void Utils::save_boxes(std::vector<cv::Rect>& dst, const std::vector<cv::Vec3f>& circles, const cv::Mat& src, const cv::Point& off) {
    for (int i = 0; i < circles.size(); i++) {
        // Set safe bounding boxes
        int minX = std::max((int) std::round(circles[i][0] - circles[i][2]), 0);
        int minY = std::max((int) std::round(circles[i][1] - circles[i][2]), 0);
        int maxX = std::min((int) std::round(circles[i][0] + circles[i][2]), src.cols);
        int maxY = std::min((int) std::round(circles[i][1] + circles[i][2]), src.rows);

        dst.push_back(std::move(cv::Rect(minX + off.x, minY + off.y, maxX - minX, maxY - minY)));
    }
    return;
}

void Utils::save_boxes(std::vector<cv::Rect>& dst, const std::vector<cv::Vec3f>& circles, const cv::Mat& src) {
    for (int i = 0; i < circles.size(); i++) {
        // Set safe bounding boxes
        int minX = std::max((int) std::round(circles[i][0] - circles[i][2]), 0);
        int minY = std::max((int) std::round(circles[i][1] - circles[i][2]), 0);
        int maxX = std::min((int) std::round(circles[i][0] + circles[i][2]), src.cols);
        int maxY = std::min((int) std::round(circles[i][1] + circles[i][2]), src.rows);

        dst.push_back(std::move(cv::Rect(minX, minY, maxX - minX, maxY - minY)));
    }
    return;
}

void Utils::clean_boxes(std::vector<cv::Rect>& src, int threshold) {
    int i=0;
    for(const auto& rect : src)  {
        if(rect.area() < threshold) src.erase(src.begin() + i);
        else i++;
    }
}

std::vector<double> Utils::compute_IoU(const std::vector<cv::Rect>& GT_boxes, const std::vector<cv::Rect>& det_boxes) {
    

    if (det_boxes.empty() || GT_boxes.empty()) {
        std::cerr << "[compute_IoU] Errore: det_boxes o GT_boxes è vuoto." << std::endl;
        return {}; }
    
    

    // Translate det_boxes into a tl dataset
    cv::Mat dataset(det_boxes.size(), 2, CV_32F);
    for (int i = 0; i < det_boxes.size(); i++)  {
        dataset.at<float>(i, 0) = det_boxes[i].x;
        dataset.at<float>(i, 1) = det_boxes[i].y;
    }

    // cv::flann::Index flannIndex(dataset, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);


    try {
        cv::flann::Index flannIndex(dataset, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);
        ...
    } catch (const cv::Exception& e) {
        std::cerr << "[compute_IoU] OpenCV FLANN error: " << e.what() << std::endl;
        return {};
    }









    std::vector<int> indices(1);
    std::vector<float> dists(1);
    
    std::vector<cv::Rect> matched(GT_boxes.size(), cv::Rect(0,0,0,0));
    
    // Adgust det_boxes to fit GT_boxes
    for(int i=0; i < GT_boxes.size(); i++) {
        // Query
        cv::Mat queryMat(1, 2, CV_32F);
        queryMat.at<float>(0, 0) = GT_boxes[i].x;
        queryMat.at<float>(0, 1) = GT_boxes[i].y;

        // Output
        flannIndex.knnSearch(queryMat, indices, dists, 1, cv::flann::SearchParams(32));
        matched[i] = det_boxes[indices[0]];
    }


    std::vector<double> IoUs;
    for(int i=0; i < GT_boxes.size(); i++) {
        IoUs.emplace_back(IoU(GT_boxes[i], matched[i]));
    }
    return IoUs;
}

double Utils::IoU(const cv::Rect& a, const cv::Rect& b) {
    int inter = (a & b).area();
    int uni = a.area() + b.area() - inter;

    if (uni == 0) return 0.0;
    return static_cast<double>(inter) / uni;
}

double Utils::Rect_distance(const cv::Rect& a, const cv::Rect& b) {
    double dx = a.x + a.width / 2 - b.x + b.width / 2;
    double dy = a.y + a.height / 2 - b.y + b.height / 2;
    return std::sqrt(dx * dx + dy * dy);
}
