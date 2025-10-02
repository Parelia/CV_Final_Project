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

/*
    std::vector<cv::Mat> data_images;
    std::string dir_path = "../Malaria_Dataset/Img/";

    if(!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) return -1;

    for(const auto& current_file : std::filesystem::directory_iterator(dir_path)) {
        if(!current_file.is_regular_file()) continue;
        data_images.emplace_back(cv::imread(current_file.path()));
    }
*/
// Firstly sort both vectors
    // std::sort(GT_boxes.begin(), GT_boxes.end(), [](const cv::Rect& box_a, const cv::Rect& box_b) {
    //     if(box_a.y == box_b.y)
    //         return box_a.x < box_b.x;
    //     return box_a.y < box_b.y;
    // });

int main() {

    std::string input = "IMG_20150621_112023";
    
    cv::Mat test_image = cv::imread("../Malaria_Dataset/Img/" + input + ".jpg");
    cv::fastNlMeansDenoisingColored(test_image, test_image);

    cv::Mat img = test_image.clone();   // Circles
    cv::Mat img2 = test_image.clone();  // Bboxes
    
    // Process image
    cv::Mat processed_image;
    Utils::offset diff;
    Utils::cut_circle_borders(test_image, processed_image, diff);
    
    cv::Mat gray;
    cv::cvtColor(processed_image, gray, cv::COLOR_BGR2GRAY);


    cv::Mat mask = gray.clone() > 10;
    cv::bitwise_not(mask, mask);
    
    // ADJUST PARAMETERS!!!!
    // Thresholding
    cv::Mat thresh;
    cv::threshold(gray, thresh, 165, 255, 0);
    cv::add(thresh, mask, thresh);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)), cv::Point(-1,-1), 4);
    
    // cv::erode(thresh, thresh, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 3);
    cv::dilate(thresh, thresh, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1,-1), 2);
    
    
    // 2. Binarizza (Otsu per esempio)
    //cv::threshold(gray, thresh, 155, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // cv::adaptiveThreshold(gray, thresh, 255,
    //     cv::ADAPTIVE_THRESH_MEAN_C, // o GAUSSIAN_C
    //     cv::THRESH_BINARY,
    //     11, // blocco
    //     2); // costante sottratta

    //cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)), cv::Point(-1,-1), 2);

    cv::Mat neg;
    cv::bitwise_not(thresh, neg);

    cv::namedWindow("thresh", cv::WINDOW_NORMAL);
    cv::resizeWindow("thresh", 1200, 700);
    cv::imshow("thresh", thresh);


    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(neg, labels, stats, centroids);

    for (int i = 1; i < numLabels; i++) { // label i = 0 means background
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        // std::cout << area << std::endl;
        if (area <= 5000 || area >= 50000) labels.setTo(0, labels == i); 
    }

    
    std::vector<cv::Rect> CcBoxes;
    for (int i = 1; i < stats.rows; i++)
    CcBoxes.push_back(cv::Rect(
        stats.at<int>(i, cv::CC_STAT_LEFT),
        stats.at<int>(i, cv::CC_STAT_TOP),
        stats.at<int>(i, cv::CC_STAT_WIDTH), 
        stats.at<int>(i, cv::CC_STAT_HEIGHT)));


    // // Find circles approach
    // std::vector<cv::Vec3f> circles;
    // cv::HoughCircles(thresh, circles, cv::HOUGH_GRADIENT, 1,
    //             55,         // min dist btw centers
    //             50, 9,      // param1 = Canny threshold, param2 = accumulator threshold
    //             55, 80);    // min max radius

    // // // Draw circles on img
    // // for (size_t i = 0; i < circles.size(); i++) {
    // //     cv::circle(img, cv::Point(circles[i][0] + diff.dx, circles[i][1] + diff.dy), circles[i][2], cv::Scalar(255, 0, 0), 2); // draw circle
    // //     cv::circle(img, cv::Point(circles[i][0] + diff.dx, circles[i][1] + diff.dy), 2, cv::Scalar(0, 0, 255), 3);      // draw centers
    // // }
    // // cv::namedWindow("Circles", cv::WINDOW_NORMAL);
    // // cv::resizeWindow("Circles", 1200, 700);
    // // cv::imshow("Circles", img);


    // cv::Mat markers;
    // cv::connectedComponents(neg, markers);
    // // markers += 1;  // Incrementa di 1 per distinguere da background


    cv::Mat markers;
    // = cv::Mat::zeros(markers.size(), CV_32SC1);
    // for (int i = 0; i < circles.size(); i++)
    //     markers.at<int>(cv::Point(circles[i][0], circles[i][1])) = i+1;
    labels.convertTo(markers, CV_8U);
    cv::namedWindow("markers", cv::WINDOW_NORMAL);
    cv::resizeWindow("markers", 1200, 700);
    cv::imshow("markers", markers);

    
    // // Set up for watershed
    // // cv::Mat markers = cv::Mat::zeros(processed_image.size(), CV_32SC1);
    // // for (int i = 0; i < circles.size(); i++)
    // //     markers.at<int>(cv::Point(circles[i][0], circles[i][1])) = i+1;
    
    // Apply watershed
    cv::watershed(processed_image, labels);
 
    // Draw and paint borders(-1)
    cv::Mat result = test_image.clone();
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            if (labels.at<int>(i, j) == -1) {
                result.at<cv::Vec3b>(i + diff.dy, j + diff.dx) = cv::Vec3b(255, 0, 0); // border
            }
        }
    }
    cv::namedWindow("Result", cv::WINDOW_NORMAL);
    cv::resizeWindow("Result", 1200, 700);
    cv::imshow("Result", result);

    labels.convertTo(markers, CV_8U);
    cv::namedWindow("markers2", cv::WINDOW_NORMAL);
    cv::resizeWindow("markers2", 1200, 700);
    cv::imshow("markers2", markers);


    // // Acquire bounding boxes for IoU calculus
    // std::vector<cv::Rect> image_boxes;
    // Utils::save_boxes(image_boxes, circles, processed_image, diff);
    // Utils::clean_boxes(image_boxes, 200);
    
    // Upload file and get dataset bounding boxes
    std::vector<Parser::GT_Entry> dst;
    Parser::parse_file(Parser::get_file_content("../Malaria_Dataset/GT/" + input + ".txt"), dst);

    std::vector<cv::Rect> GT_boxes;
    for(const Parser::GT_Entry e : dst) {
        cv::Rect temp_box = cv::Rect(cv::boundingRect(e.points));
        GT_boxes.emplace_back(temp_box);

        // cv::rectangle(img2, temp_box, cv::Scalar(0, 255, 0), 2);    // Draw bounding boxes
    }
    // cv::namedWindow("Bbox", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Bbox", 1200, 700);
    // cv::imshow("Bbox", img2);


    // Compute IoU over acquired data
    // std::vector<double> out(std::move(Utils::compute_IoU(GT_boxes, image_boxes)));
    std::vector<double> out(std::move(Utils::compute_IoU(GT_boxes, CcBoxes)));

    // View results
    // std::cout << "size: " << image_boxes.size() << std::endl;
    std::cout << "size: " << CcBoxes.size() << std::endl;
    std::cout << "size: " << out.size() << std::endl;
    int i=0; double count=0;
    for(const auto& d : out)
        if(d > 0) {
            // std::cout << std::to_string(d) << std::endl;
            count += d;
            i++;
        }
    std::cout << std::to_string(count / i * 100) << std::endl;
    std::cout << std::to_string(count / out.size() * 100) << std::endl;
            

    cv::waitKey(0);
    return 0;
}