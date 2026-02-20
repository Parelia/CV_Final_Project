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


#define DIR_ARG 1
#define MODEL_ARG 2

// Test mode (debug purpose):
// CV_Final_Project_Cropper --test [dataset_img_location]
// Normal mode:
// CV_Final_Project_Cropper [input_image] [model_file] (optional: --quiet)

int main(int argc, char* argv[]) {
    
    bool bIoUTestMode = false;
    std::filesystem::path dataset_img_location;
    std::filesystem::path dataset_gt_location;
    std::string model_location;
    std::vector<float> IoUs;

    // --------------------------------- START VALIDATING INPUT ----------------------------------
    
    if(argc == 3) {
        if(std::string(argv[MODEL_ARG]) == "--IoU") {
            // ./Cropper <images dir> --IoU
            std::string str_dataset_location = std::string(argv[DIR_ARG]);
            if (!str_dataset_location.empty() && str_dataset_location.back() == '/') {
                str_dataset_location.pop_back();
            }
            dataset_img_location = str_dataset_location + "/Img/";
            dataset_gt_location = str_dataset_location + "/GT/";

            std::cout << "[Cropper] IoU Test mode enabled\n";
            std::cout << "[Cropper] Using dataset location: " << dataset_img_location << std::endl;
            bIoUTestMode = true;
        } else {
            // .Cropper <images dir> <Onnx model>
            // check if model file exists
            if(!std::filesystem::exists(argv[MODEL_ARG])) {
                std::cerr << "[Cropper] Error: model file does not exist\n";
                return -1;
            }
            dataset_img_location = std::string(argv[DIR_ARG]);
            model_location = std::string(argv[MODEL_ARG]);
        }
    }else{
        std::cout << "[Cropper] Usage:" << std::endl;
        std::cout << "\t " + std::string(argv[0]) + " <dataset dir> --IoU: Generate andh show IoU report" << std::endl;
        std::cout << "\t " + std::string(argv[0]) + " .Cropper <images dir> <Onnx model>: Generate andh show IoU report" << std::endl;
        std::cout << std::endl;
    }


    if(!std::filesystem::exists(dataset_img_location)) {
        // print pwd
        std::cerr << "[Cropper] Current working directory: " << std::filesystem::current_path() << std::endl;
        std::cerr << "[Cropper] Error: input path does not exist\n";
        // print FULL path
        std::cerr << "[Cropper] Error: input path: " << std::filesystem::absolute(dataset_img_location) << std::endl;
        return -1;
    }
    Classifier classifier = Classifier();
    if(!bIoUTestMode) {
        try {
            classifier.load_model(model_location);
        } catch (const cv::Exception &e) {
            std::cerr << "[Cropper] Error loading model: " << e.what() << std::endl;
            return -1;
        }
    }
    // loop through all .jpg files in the input path dataset_img_location
    for(const auto& entry : std::filesystem::directory_iterator(dataset_img_location)) {
        if(entry.path().extension() == ".jpg") {
            std::cout << "[Cropper] Processing: " << entry.path().filename() << std::endl;

            std::filesystem::path img_path = entry.path();

            std::vector<cv::Rect> waterBoxes = Cropper::crop_image(img_path.string());   
            
            // ----------------------------------- SAVE RESULT IMAGES ------------------------------------
            // Save waterboxes to .txt file
            std::string fileName = std::filesystem::path(img_path).stem().string();
            if(!std::filesystem::exists("../Malaria_Dataset/BBox")) std::filesystem::create_directories("../Malaria_Dataset/BBox"); // ../ to exit build
            std::string outputFileName = "../Malaria_Dataset/BBox/" + fileName + "_bb.txt";
            std::ofstream outputFile(outputFileName);

            if(!outputFile.is_open()) {
                std::cerr << "[Cropper] Error: Unable to open file for writing: " << outputFileName << std::endl;
                return {};
            }

            for(const auto& box : waterBoxes) {
                outputFile << box.x << " " << box.y << " " 
                        << box.width << " " << box.height << "\n";
            }

            outputFile.close();
            std::cout << "[Cropper] WaterBoxes written to file: " << outputFileName << std::endl;
        
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
            if(!bIoUTestMode){
                // --------------------------------- START CLASSIFICATION -----------------------------------
                // foreach waterBox, crop the image and classify it
                std::vector<std::pair<cv::Rect, Classifier_Output>> classifications;
                for(const auto& box : waterBoxes) {
                    cv::Mat cropped = test_image(box);
                    Classifier_Output output = classifier.classify(cropped);
                    classifications.push_back({box, output});
                    std::cout << "[Cropper] Box at (" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "): "
                            << output.label << " (" << output.probability * 100.0f << "%), Confidence: " << output.confidence << std::endl;
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
                std::cout << "[Cropper] Result image saved to: " << outputImageName << std::endl;
            }else{
                // --------------------------------- START IoU COMPUTATION -----------------------------------

                // Upload file and get original dataset bounding boxes
                std::vector<Parser::GT_Entry> entries;
                std::cout << dataset_gt_location.string() + fileName + ".txt" << std::endl;
                Parser::parse_file(Parser::get_file_content(dataset_gt_location.string() + fileName + ".txt"), entries);

                if (entries.empty())
                {
                    std::cerr << "[main] Error: entries is empty\n";
                    return 0;
                }

                // Populate GT bounding boxes vector
                std::vector<cv::Rect> GT_Bboxes;
                for (const auto &entry : entries)
                    GT_Bboxes.push_back(cv::boundingRect(entry.points));

                // -------------------------------------------------------------------------------------------
                // Compute IoU over acquired data
                std::vector<double> out(Utils::compute_IoU(GT_Bboxes, waterBoxes));

                // Show results
                std::cout << "Showing IoU results" << std::endl;
                std::cout << "Detected cells: " << waterBoxes.size() << " out of " << out.size() << std::endl;

                int i = 0;
                double count = 0;
                for (const auto &d : out){
                    if (d > 0) {
                        count += d;
                        i++;
                    }
                }
                std::cout << "Mean IoU over matched: " << std::to_string(std::round(count / i * 100000) / 1000) << "%" << std::endl;
                std::cout << "Mean IoU overall: " << std::to_string(std::round(count / out.size() * 100000) / 1000) << "%" << std::endl;
                IoUs.push_back(count / out.size());

                // ---------------------------------- END IoU COMPUTATION ------------------------------------
            }

            // Utils::img_show(result, "result with classifier", true);
        }
    }

    if(bIoUTestMode){
        float sumOfIoUs = 0.0f;
        for (auto& n : IoUs){
            sumOfIoUs += n;
        }
        std::cout << "IoU over the whole dataset: " << std::to_string(sumOfIoUs / IoUs.size() * 100) << "%" << std::endl;
    }

    int key;
    while((key = cv::waitKey(0)) != 27 && key != -1); // 27 is ESC, -1 is window close

    return 0;
}
