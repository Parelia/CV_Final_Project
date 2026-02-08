/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#include "Utils.h"

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
    cv::Mat gray;
    if(src.channels() == 1) gray = src.clone();
	else {
		cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	}
	
	cv::Mat temp_dst;
    cv::Mat mask = gray <= 10;
   
    cv::adaptiveThreshold(gray, temp_dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 47, 2);
	cv::subtract(temp_dst, mask, temp_dst); // remove radial distortion
	
	cv::erode(temp_dst, temp_dst, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1,-1), 3);
    cv::morphologyEx(temp_dst, temp_dst, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)), cv::Point(-1,-1), 1);

    dst = temp_dst.clone();
    return;
}


std::vector<double> Utils::compute_IoU(const std::vector<cv::Rect>& GT_boxes, const std::vector<cv::Rect>& det_boxes) {
    if (det_boxes.empty()) {
        std::cerr << "[Utils][compute_IoU] Error: det_boxes is empty\n";
 
        }
    if (GT_boxes.empty()) {
        std::cerr << "[Utils][compute_IoU] Error: GT_boxes is empty\n";

        }
   

    // Translate det_boxes into a tl dataset
    cv::Mat dataset(det_boxes.size(), 2, CV_32F);
    for (int i = 0; i < det_boxes.size(); i++)  {
        dataset.at<float>(i, 0) = det_boxes[i].x;
        dataset.at<float>(i, 1) = det_boxes[i].y;
    }

    cv::flann::Index flannIndex;

    try {
        flannIndex.build(dataset, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);
     
    } catch (const cv::Exception& e) {
        std::cerr << "[Utils][compute_IoU] OpenCV FLANN error: " << e.what() << std::endl;
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
