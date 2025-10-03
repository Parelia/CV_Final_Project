/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include "Parser.h"
#include "Utils.h"

void img_show(cv::Mat img, std::string window_name) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1200, 700);
    cv::imshow(window_name, img);

    return;
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cerr << "Missing input image (required)" << std::endl;
        return -1;
    }
    
    // Get image
    cv::Mat test_image = cv::imread("../Malaria_Dataset/Img/" + std::string(argv[1]) + ".jpg");
    cv::fastNlMeansDenoisingColored(test_image, test_image);
    
    // Process image
    cv::Mat cutted_image;
    cv::Point diff;
    Utils::cut_circle_borders(test_image, cutted_image, diff);
    
    // Thresholding
    cv::Mat thresh;
    Utils::refined_threshold(cutted_image, thresh);
    cv::Mat neg_thresh;
    cv::bitwise_not(thresh, neg_thresh);

    // aggiungere a thresh 1 o 4 cerchi pieno per prendere lo sfondo
    cv::circle(thresh, cv::Point(100, 100), 100, 0, -1);
    // cv::circle(thresh, cv::Point(thresh.cols - 100, 100), 100, 0, -1);
    // cv::circle(thresh, cv::Point(100, thresh.rows - 100), 100, 0, -1);
    // cv::circle(thresh, cv::Point(thresh.cols - 100, thresh.rows - 100), 100, 0, -1);

    img_show(thresh, "thresh");

    //try connected without neg and fill the zone, then super erode

    // -------------------------------------------------------------------------------------------
    // Find valid connected components
    cv::Mat labels, stats, centroids;
    int components = cv::connectedComponentsWithStats(neg_thresh, labels, stats, centroids);
    for (int i = 1; i < components; i++) { // label i = 0 means background
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        // std::cout << area << std::endl;
        if (area <= 5000 || area >= 50000) {
            labels.setTo(0, labels == i);
            stats.at<int>(i, cv::CC_STAT_AREA) = 0;
        }
    }

    // Save connected components bounding boxes
    std::vector<cv::Rect> CcBoxes;
    for (int i = 1; i < stats.rows; i++)
    CcBoxes.push_back(cv::Rect(
        stats.at<int>(i, cv::CC_STAT_LEFT),
        stats.at<int>(i, cv::CC_STAT_TOP),
        stats.at<int>(i, cv::CC_STAT_WIDTH), 
        stats.at<int>(i, cv::CC_STAT_HEIGHT))
        + diff);

    // View results
    cv::Mat markers;
    labels.convertTo(markers, CV_8U);
    img_show(markers, "markers");
    // -------------------------------------------------------------------------------------------


    // -------------------------------------------------------------------------------------------
    // // Find circles approach
    // std::vector<cv::Vec3f> circles;
    // cv::HoughCircles(thresh, circles, cv::HOUGH_GRADIENT, 1,
    //             55,         // min dist btw centers
    //             50, 9,      // param1 = Canny threshold, param2 = accumulator threshold
    //             55, 80);    // min max radius

    // // // Draw circles on img
    // // for (size_t i = 0; i < circles.size(); i++) {
    // //     cv::circle(img, cv::Point(circles[i][0] + diff.x, circles[i][1] + diff.y), circles[i][2], cv::Scalar(255, 0, 0), 2); // draw circle
    // //     cv::circle(img, cv::Point(circles[i][0] + diff.x, circles[i][1] + diff.y), 2, cv::Scalar(0, 0, 255), 3);      // draw centers
    // // }
    // // cv::namedWindow("Circles", cv::WINDOW_NORMAL);
    // // cv::resizeWindow("Circles", 1200, 700);
    // // cv::imshow("Circles", img);

    // cv::Mat markers = cv::Mat::zeros(markers.size(), CV_32SC1);
    // for (int i = 0; i < circles.size(); i++)
    //     markers.at<int>(cv::Point(circles[i][0], circles[i][1])) = i+1;

    // Apply watershed
    // cv::watershed(cutted_image, markers);
    // -------------------------------------------------------------------------------------------
    

    // Apply watershed
    cv::watershed(cutted_image, labels);

    labels.convertTo(markers, CV_8U);
    img_show(markers, "markers after water");
 
    // Draw and paint borders(-1)
    cv::Mat result = test_image.clone();
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            if (labels.at<int>(i, j) == -1) {
                result.at<cv::Vec3b>(i + diff.y, j + diff.x) = cv::Vec3b(255, 0, 0); // border
            }
        }
    }
    img_show(result, "result");


    // operazione rimozione regioni troppo grandi
    std::vector<cv::Rect> boundingBoxes;
    for (int i = 1; i < stats.rows; i++) {
        // std::cout << area << std::endl;
        if (stats.at<int>(i, cv::CC_STAT_AREA) != 0) {
            // Crea una maschera binaria per questa etichetta
            cv::Mat mask = (labels == i);

            // Trova i contorni della maschera
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if (!contours.empty()) {
                // Calcola il bounding box del primo contorno
                cv::Rect bbox = cv::boundingRect(contours[0]);
                boundingBoxes.push_back(bbox + diff);
            }
        }
    }


    // // Acquire bounding boxes for IoU calculus
    // std::vector<cv::Rect> image_boxes;
    // Utils::save_boxes(image_boxes, circles, processed_image, diff);
    // Utils::clean_boxes(image_boxes, 200);
    

    // Upload file and get dataset bounding boxes
    std::vector<Parser::GT_Entry> dst;
    Parser::parse_file(Parser::get_file_content("../Malaria_Dataset/GT/" + std::string(argv[1]) + ".txt"), dst);

    std::vector<cv::Rect> GT_Bboxes;
    result = test_image.clone();
    for(const Parser::GT_Entry e : dst) {
        cv::Rect temp_box = cv::Rect(cv::boundingRect(e.points));
        GT_Bboxes.emplace_back(temp_box);

        cv::rectangle(result, temp_box, cv::Scalar(0, 255, 0), 2);    // Draw bounding boxes
    }
    img_show(result, "GT_Bboxes");


    // Compute IoU over acquired data
    // std::vector<double> out(std::move(Utils::compute_IoU(GT_Bboxes, image_boxes)));
    // std::vector<double> out(std::move(Utils::compute_IoU(GT_Bboxes, CcBoxes)));
    std::vector<double> out(std::move(Utils::compute_IoU(GT_Bboxes, boundingBoxes)));


    // View results
    // std::cout << "size: " << image_boxes.size() << std::endl;
    // std::cout << "size: " << CcBoxes.size() << std::endl;
    std::cout << "size: " << boundingBoxes.size() << std::endl;
    std::cout << "size: " << out.size() << std::endl;
    int i=0; double count=0;
    for(const auto& d : out)
        if(d > 0) {
            count += d;
            i++;
        }
    std::cout << "Matched: " << std::to_string(count / i * 100) << std::endl;
    std::cout << "Global: " << std::to_string(count / out.size() * 100) << std::endl;



    // Output dataset
    // if (!std::filesystem::exists(argv[2])) std::filesystem::create_directories(argv[2]);
    // for (int i = 0; i < boundingBoxes.size(); i++)
    //     if(!cv::imwrite(std::string(argv[2]) + std::string(argv[1]) + "_" + std::to_string(i+1) + ".jpg", test_image(boundingBoxes[i])))
    //         std::cerr << "Saving image error" << std::endl;

            
    cv::waitKey(0);
    return 0;
}