#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/photo.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>


/*
    blob
    watershed
    region grwoing

    canny edge

    data augmentation

*/

/*
    std::vector<cv::Mat> data_images;
    std::string dir_path = "../Malaria_Dataset/Img/";

    if(!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) return -1;

    for(const auto& current_file : std::filesystem::directory_iterator(dir_path)) {
        if(!current_file.is_regular_file()) continue;
        data_images.emplace_back(cv::imread(current_file.path()));
    }
*/

/*
    std::ifstream fstr("../Malaria_Dataset/GT/IMG_20150621_112023.txt");
    if(!fstr.is_open() || !fstr.good()){
        std::cerr << "Error: could not read from the filestream\n";
        return -1;
    }
    std::stringstream stream;
    stream << fstr.rdbuf();
    // std::cout << stream.str() << std::endl;
*/

int main() {

    cv::Mat test_image = cv::imread("../Malaria_Dataset/Img/IMG_20150621_112023.jpg");
    cv::fastNlMeansDenoisingColored(test_image, test_image);

    cv::Mat img = test_image.clone();


    cv::Mat gray;
    cv::cvtColor(test_image, gray, cv::COLOR_BGR2GRAY);

    // Thresholding
    cv::Mat thresh;
    cv::threshold(gray, thresh, 150, 255, 0);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)), cv::Point(-1,-1), 3);
    //cv::dilate(thresh, thresh, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

    cv::namedWindow("Thresh", cv::WINDOW_NORMAL);
    cv::resizeWindow("Thresh", 1200, 700);
    cv::imshow("Thresh", thresh);

    cv::Mat markers = cv::Mat::zeros(test_image.size(), CV_32SC1);
    
    // Find circles
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(thresh, circles, cv::HOUGH_GRADIENT, 1,
                55,  // distanza minima tra i centri
                50, 8,         // param1 = Canny threshold, param2 = accumulator threshold
                55, 80);        // raggio minimo e massimo

    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(circles[i][0], circles[i][1]);
        int radius = circles[i][2];
        cv::circle(img, center, radius, cv::Scalar(255, 0, 0), 2); // draw circle
        cv::circle(img, center, 2, cv::Scalar(0, 0, 255), 3);      // draw centers
        markers.at<int>(center) = i+1;
    }

    cv::namedWindow("Circles", cv::WINDOW_NORMAL);
    cv::resizeWindow("Circles", 1200, 700);
    cv::imshow("Circles", img);

    // Applica watershed
    cv::watershed(test_image, markers);

    // Disegna i bordi (pixels marcati -1)
    cv::Mat result = test_image.clone();
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) == -1) {
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); // border
            }
        }
    }

    cv::namedWindow("Result", cv::WINDOW_NORMAL);
    cv::resizeWindow("Result", 1200, 700);
    cv::imshow("Result", result);



    cv::waitKey(0);
    return 0;
}