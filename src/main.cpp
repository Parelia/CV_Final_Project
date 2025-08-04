#include <opencv2/highgui.hpp>
#include <filesystem>

int main() {
    cv::Mat test_image = cv::imread("../Malaria_Dataset/IMG_20150621_112023.jpg");
    
    /*
    std::vector<cv::Mat> data_images;
    std::string dir_path = "../Malaria_Dataset/";

    if(!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) return -1;

    for(const auto& current_file : std::filesystem::directory_iterator(dir_path)) {
        if(!current_file.is_regular_file()) continue;
        data_images.emplace_back(cv::imread(current_file.path()));
    }*/

    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::imshow("Image", test_image);

    cv::waitKey(0);

    return 0;
}