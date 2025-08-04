#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
   

    cv::Mat test_image = cv::imread("../Malaria_Dataset/Img/IMG_20150621_112023.jpg");

    /*
    std::vector<cv::Mat> data_images;
    std::string dir_path = "../Malaria_Dataset/Img/";

    if(!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) return -1;

    for(const auto& current_file : std::filesystem::directory_iterator(dir_path)) {
        if(!current_file.is_regular_file()) continue;
        data_images.emplace_back(cv::imread(current_file.path()));
    }*/


    std::ifstream fstr("../Malaria_Dataset/GT/IMG_20150621_112023.txt");
    if(!fstr.is_open() || !fstr.good()){
        std::cerr << "Error: could not read from the filestream\n";
        return -1;
    }
    std::stringstream stream;
    stream << fstr.rdbuf();
    // std::cout << stream.str() << std::endl;


    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image", 1200, 700);
    cv::imshow("Image", test_image);

    cv::waitKey(0);

    return 0;
}