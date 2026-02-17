/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <cmath>
#include "Parser.h"
#include "Utils.h"
#include "Cropper.h"
#include "Classifier.h"


#define IMG_ARG 1
#define MODEL_ARG 2
#define MODE_ARG 3

// Test mode (debug purpose):
// CV_Final_Project_Cropper --test [dataset_location]
// Normal mode:
// CV_Final_Project_Cropper [input_image] [model_file] (optional: --quiet)
bool bTestMode = false;
bool QUIET_MODE = false;
std::string dataset_location;




int main(int argc, char* argv[]) {
    // --------------------------------- START VALIDATING INPUT ----------------------------------
    if(argc < 2) {
        std::cerr << "[main] Missing input image (required)\n";
        return -1;
    }
    
    if(argc > 2 && std::string(argv[2]) == "--test") {
        if(argc < 4) {
            std::cerr << "[main] Dataset location required in test mode\n";
            return -1;
        }

        dataset_location = argv[3];
        if (!dataset_location.empty() && dataset_location.back() == '/') {
            dataset_location.pop_back();
        }

        std::cout << "[main] Test mode enabled\n";
        std::cout << "[main] Using dataset location: " << dataset_location << std::endl;
        bTestMode = true;
    }
    else {
        if(argc < 3) {
            std::cerr << "[main] Missing model file (required in normal mode)\n";
            return -1;
        }
        // check if model file exists
        if(!std::filesystem::exists(argv[MODEL_ARG])) {
            std::cerr << "[main] Error: model file does not exist\n";
            return -1;
        }
        if(argc > 3 && std::string(argv[3]) == "--quiet") {
            QUIET_MODE = true;
            std::cout << "[main] Quiet mode enabled. No intermediate results will be shown.\n";
        }
    }

    std::filesystem::path p(argv[IMG_ARG]);

    if(!std::filesystem::exists(p)) {
        // print pwd
        std::cerr << "[main] Current working directory: " << std::filesystem::current_path() << std::endl;
        std::cerr << "[main] Error: input path does not exist\n";
        // print FULL path
        std::cerr << "[main] Error: input path: " << std::filesystem::absolute(p) << std::endl;
        return -1;
    }
    Classifier classifier = Classifier();
    if(!bTestMode) {
        try {
            classifier.load_model(argv[MODEL_ARG]);
        } catch (const cv::Exception &e) {
            std::cerr << "[main] Error loading model: " << e.what() << std::endl;
            return -1;
        }
    }
    // loop through all .jpg files in the input path p
    for(const auto& entry : std::filesystem::directory_iterator(p)) {
        if(entry.path().extension() == ".jpg") {
            std::cout << "[main] Processing: " << entry.path().filename() << std::endl;

            std::filesystem::path img_path = entry.path();

            std::vector<cv::Rect> waterBoxes = Cropper::crop_image(img_path.string());   
            
            // ----------------------------------- SAVE RESULT IMAGES ------------------------------------
            // Save waterboxes to .txt file
            std::string fileName = std::filesystem::path(img_path).stem().string();
            if(!std::filesystem::exists("../Malaria_Dataset/BBox")) std::filesystem::create_directories("../Malaria_Dataset/BBox"); // ../ to exit build
            std::string outputFileName = "../Malaria_Dataset/BBox/" + fileName + "_bb.txt";
            std::ofstream outputFile(outputFileName);

            if(!outputFile.is_open()) {
                std::cerr << "[main] Error: Unable to open file for writing: " << outputFileName << std::endl;
                return {};
            }

            for(const auto& box : waterBoxes) {
                outputFile << box.x << " " << box.y << " " 
                        << box.width << " " << box.height << "\n";
            }

            outputFile.close();
            std::cout << "[main] WaterBoxes written to file: " << outputFileName << std::endl;
        
            if(!std::filesystem::exists("../OutputImages/Cropper")) std::filesystem::create_directories("../OutputImages/Cropper");

            // Draw bounding boxes
            cv::Mat test_image = cv::imread(img_path.string());
            cv::Mat results = test_image.clone();
            for(const auto& box : waterBoxes)
                cv::rectangle(results, box, cv::Scalar(0, 255, 0), 2);
            cv::imwrite("../OutputImages/Cropper/" + fileName + "_boxes.jpg", results);
            
            // Draw borders
            results = test_image.clone();
            for(const auto& box : waterBoxes){
                cv::rectangle(results, box, cv::Scalar(255, 0, 0), 2);
            }
            cv::imwrite("../OutputImages/Cropper/" + fileName + "_contours.jpg", results);
            // -------------------------------------------------------------------------------------------

            // --------------------------------- START CLASSIFICATION -----------------------------------
            // foreach waterBox, crop the image and classify it
            std::vector<std::pair<cv::Rect, Classifier_Output>> classifications;
            for(const auto& box : waterBoxes) {
                cv::Mat cropped = test_image(box);
                Classifier_Output output = classifier.classify(cropped);
                classifications.push_back({box, output});
                std::cout << "[main] Box at (" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "): "
                          << output.label << " (" << output.probability * 100 << "%), Confidence: " << output.confidence << std::endl;
            }

            // Draw and paint entries rects
            cv::Mat result = test_image.clone();
            for(const auto& entry : classifications) {
                const cv::Rect& rect = entry.first;
                const Classifier_Output& output = entry.second;
                cv::Scalar color = output.parasitized ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::rectangle(result, rect, color, 2);
                // Print confidence on the top of the rectangle
                std::string label = std::to_string(static_cast<int>(output.probability * 100)) + "%";
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::rectangle(result, cv::Point(rect.x, rect.y - labelSize.height - baseLine), cv::Point(rect.x + labelSize.width, rect.y), color, cv::FILLED);
                cv::putText(result, label, cv::Point(rect.x, rect.y - baseLine), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }

            // Save images as [original_image_name]_result.jpg
            if(!std::filesystem::exists("../OutputImages/Classifier/")) {
                std::filesystem::create_directory("../OutputImages/Classifier/");
            }
            std::string outputImageName = "../OutputImages/Classifier/" + std::filesystem::path(img_path).stem().string() + "_result.jpg";
            cv::imwrite(outputImageName, result);
            std::cout << "[main] Result image saved to: " << outputImageName << std::endl;

            // Utils::img_show(result, "result with classifier", true);
        }
    }

    int key;
    while((key = cv::waitKey(0)) != 27 && key != -1); // 27 is ESC, -1 is window close

    return 0;
}
