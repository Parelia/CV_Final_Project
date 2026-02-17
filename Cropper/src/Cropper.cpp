/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/
#include "Cropper.h"
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include "Utils.h"


namespace Cropper {

    std::vector<cv::Rect> crop_image(const std::string& input_path) {
        cv::Mat test_image = cv::imread(input_path);
        
        if(test_image.empty()) {
            std::cerr << "[main] Error: test_image is empty\n";
            return {};
        }

        // ---------------------------------- END VALIDATING INPUT -----------------------------------
        
        // --------------------------------- START IMAGE PROCESSING ----------------------------------
        // Cut image
        cv::Mat cutted_image;
        cv::Point diff;
        Utils::cut_background(test_image, cutted_image, diff);

        // Thresholding
        cv::Mat thresh;
        Utils::refined_threshold(cutted_image, thresh);


        // -------------------------------------------------------------------------------------------
        // Find connected components #1
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        if(cv::connectedComponentsWithStats(thresh, labels, stats, centroids) == 0) {
            std::cerr << "[main] Error: connectedComponentsWithStats output is 0\n";
            return {};
        }
        
        // Discard imperfections (small elements) #1
        for(int i = 1; i < stats.rows; i++) { // label 0 = background
            if(stats.at<int>(i, cv::CC_STAT_AREA) < 500) {
                labels.setTo(0, labels == i);
                stats.at<int>(i, cv::CC_STAT_AREA) = 0;
            }
        }


        // Show results
        cv::Mat markers;
        labels.convertTo(markers, CV_8U);
        //img_show(markers, "markers");

        // Apply watershed #1
        cv::watershed(cutted_image, labels);
        
        // Show results
        labels.convertTo(markers, CV_8U);
        //img_show(markers, "markers after w1");

        for(int i = 1; i < stats.rows; i++) {  // label 0 = background
            if(stats.at<int>(i, cv::CC_STAT_AREA) != 0) {
                cv::Mat mask = (labels == i);
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                if(!contours.empty()) {
                    cv::Rect temp = cv::boundingRect(contours[0]);
                    if(temp.area() < 800 || temp.area() > 100000) labels.setTo(0, labels == i);
                }
            }	
        }
        

        // -------------------------------------------------------------------------------------------
        // Adjust for second watershed
        cv::Mat labels8u;
        labels.convertTo(labels8u, CV_8U);
        cv::erode(labels8u, labels8u, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)), cv::Point(-1,-1), 4);
        cv::erode(labels8u, labels8u, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1,-1), 1);
        
        // Find connected components #2
        cv::Mat labels2;
        cv::Mat stats2;
        cv::Mat centroids2;
        if(cv::connectedComponentsWithStats(labels8u, labels2, stats2, centroids2) == 0) {
            std::cerr << "[main] Error: connectedComponentsWithStats2 output is 0\n";
            return {};
        }
        
        // Discard imperfections (small elements) #2
        for(int i = 1; i < stats2.rows; i++) {  // label 0 = background
            if(stats2.at<int>(i, cv::CC_STAT_AREA) < 1400) {
                labels2.setTo(0, labels2 == i);
                stats2.at<int>(i, cv::CC_STAT_AREA) = 0;
            }	
        }


        // Show results
        cv::Mat markers2;
        labels2.convertTo(markers2, CV_8U);
        //img_show(markers2, "markers before water2");

        // Apply watershed #2
        cv::watershed(cutted_image, labels2);

        // Show results
        labels2.convertTo(markers2, CV_8U);
        //img_show(markers2, "img after water2");
        
        // Compute bounding boxes
        std::vector<cv::Rect> waterBoxes;
        for(int i = 1; i < stats2.rows; i++) {  // label 0 = background
            if(stats2.at<int>(i, cv::CC_STAT_AREA) != 0) {
                cv::Mat mask = (labels2 == i);
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                if(!contours.empty()) {
                    cv::Rect temp = cv::boundingRect(contours[0]);
                    waterBoxes.push_back(temp + diff);
                }
            }	
        }
        
        if(waterBoxes.empty()) {
            std::cerr << "[main] Error: waterBoxes is empty\n";
            return {};
        }


        // -------------------------------------------------------------------------------------------
        // Remove waterBoxes with area greater than 400 * 400 pixels
        waterBoxes.erase( std::remove_if(
            waterBoxes.begin(),
            waterBoxes.end(),
            [](const cv::Rect& box) { return box.area() > (400 * 400); }),
            waterBoxes.end()
        );
        return waterBoxes;
    }
} // namespace Cropper