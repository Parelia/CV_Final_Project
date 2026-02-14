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
#include "ClassifiedParser.h"
#include "Parser.h"
#include "Utils.h"

#define IMG_ARG 1
#define PYTHON_ARG 2
#define MODE_ARG 3

//TODO: clean the code (Put everything in a class and call it from main)
bool TEST_MODE = false;
bool QUIET_MODE = false;
std::string dataset_location;

// Compact the code
void img_show(cv::Mat img, std::string window_name, bool force = false) {
    if(QUIET_MODE){ // Cannot override quiet mode
        return;
    }
    if(!TEST_MODE && !force) {
        return;
    }
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1200, 700);
    cv::imshow(window_name, img);
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "[main] Missing input image (required)\n";
        return -1;
    }
    // test mode: 
    // CV_Final_Project_Cropper --test [dataset_location]

    // normal mode
    // CV_Final_Project_Cropper [input_image] [python_classifier_script] (optional: --quiet)
    
    if(argc > 2 && std::string(argv[2]) == "--test") {
        if(argc < 4) {
            std::cerr << "[main] dataset location required in test mode\n";
            return -1;
        }
        dataset_location = argv[3];
        if (!dataset_location.empty() && dataset_location.back() == '/') {
            dataset_location.pop_back();
        }
        std::cout << "[main] Test mode enabled.\n";
        std::cout << "[main] Using dataset location: " << dataset_location << std::endl;
        TEST_MODE = true;
    } else {
        if(argc < 3) {
            std::cerr << "[main] Missing python classifier script (required in normal mode)\n";
            return -1;
        }
        if(argc > 3 && std::string(argv[3]) == "--quiet") {
            QUIET_MODE = true;
            std::cout << "[main] Quiet mode enabled. No intermediate results will be shown.\n";
        }
    }

    if(!std::filesystem::exists(argv[IMG_ARG])) {
        // print pwd
        std::cerr << "[main] Current working directory: " << std::filesystem::current_path() << std::endl;
        std::cerr << "[main] Error: input image does not exist\n";
        // print FULL path
        std::cerr << "[main] Error: input image path: " << std::filesystem::absolute(argv[IMG_ARG]) << std::endl;

        return -1;
    }

    cv::Mat test_image = cv::imread(std::string(argv[IMG_ARG]));
    
    if(test_image.empty()) {
        std::cerr << "[main] Error: test_image is empty\n";
		return -1;
    }
     
    
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
		return -1;
    }
    
    // Discard imperfections (small elements) #1
    for(int i = 1; i < stats.rows; i++) { // label 0 = background
        if(stats.at<int>(i, cv::CC_STAT_AREA) < 500) {
            labels.setTo(0, labels == i);
            stats.at<int>(i, cv::CC_STAT_AREA) = 0;
        }
    }


    // // View results
    // cv::Mat markers;
    // labels.convertTo(markers, CV_8U);
    // img_show(markers, "markers");

    // Apply watershed #1
    cv::watershed(cutted_image, labels);
    
    // // View results
    // labels.convertTo(markers, CV_8U);
    // img_show(markers, "markers after w1");

    for(int i = 1; i < stats.rows; i++) {  // label 0 = background
        if(stats.at<int>(i, cv::CC_STAT_AREA) != 0) {
            cv::Mat mask = (labels == i);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if(!contours.empty()) {
				cv::Rect temp = cv::boundingRect(contours[0]);
				if(temp.area() < 800 || temp.area() > 100000)
					labels.setTo(0, labels == i);
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
		return 0;
    }
	
	// Discard imperfections (small elements) #2
	for(int i = 1; i < stats2.rows; i++) {  // label 0 = background
		if(stats2.at<int>(i, cv::CC_STAT_AREA) < 1400) {
			labels2.setTo(0, labels2 == i);
			stats2.at<int>(i, cv::CC_STAT_AREA) = 0;
		}	
	}


	// // View results
    // cv::Mat markers2;
	// labels2.convertTo(markers2, CV_8U);
    // img_show(markers2, "markers before water2");

	// Apply watershed #2
    cv::watershed(cutted_image, labels2);

    // // View results
    // labels2.convertTo(markers2, CV_8U);
    // img_show(markers2, "img after water2");
	
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
		return -1;
    }


    // -------------------------------------------------------------------------------------------
    // Remove waterBoxes with area greater than 400 * 400 pixels
    waterBoxes.erase(
        std::remove_if(waterBoxes.begin(), waterBoxes.end(), [](const cv::Rect& box) {
            return box.area() > 400 * 400;
        }),
        waterBoxes.end()
    );

    // Write waterBoxes to a .txt file
    std::string outputFileName = "output_bb.txt";
    std::ofstream outputFile(outputFileName);

    if(!outputFile.is_open()) {
        std::cerr << "[main] Error: Unable to open file for writing: " << outputFileName << std::endl;
        return -1;
    }

    for(const auto& box : waterBoxes) {
        outputFile << box.x << " " << box.y << " " 
                   << box.width << " " << box.height << "\n";
    }

    outputFile.close();
    std::cout << "[main] WaterBoxes written to file: " << outputFileName << std::endl;


    // -------------------------------------------------------------------------------------------
    // Write output images
    if(!std::filesystem::exists("OutputImages/Parser")) std::filesystem::create_directories("OutputImages/Parser");

    // Draw bounding boxes
    cv::Mat results = test_image.clone();
    for(const auto& box : waterBoxes)
        cv::rectangle(results, box, cv::Scalar(0, 255, 0), 2);
    cv::imwrite("OutputImages/Parser/" + std::string(argv[IMG_ARG]) + "_boxes.png", results);
    
    // Draw borders
    results = test_image.clone();
    for(int i = 0; i < labels2.rows; i++) {
        for(int j = 0; j < labels2.cols; j++) {
            if(labels2.at<int>(i, j) == -1) {
                // Draw border
                results.at<cv::Vec3b>(i + diff.y, j + diff.x) = cv::Vec3b(0, 255, 0);
            }
        }
    }
    cv::imwrite("OutputImages/Parser/" + std::string(argv[IMG_ARG]) + "_contours.png", results);
 
    // ---------------------------------- END IMAGE PROCESSING -----------------------------------


    if(!TEST_MODE) {
        std::cout << "[main] Calling classifier" << std::endl;

        std::string pythonScript = argv[PYTHON_ARG];
        std::string command = "python3 " + pythonScript + " " + std::string(argv[IMG_ARG]) + " " + outputFileName;

        int retCode = std::system(command.c_str());
        if (retCode != 0) {
            std::cerr << "[main] Error: Python classifier script execution failed with code " << retCode << std::endl;
            return -1;
        }

        std::cout << "[main] Python classifier script executed successfully. Please see predictions.txt for the raw output" << std::endl;

        // load predictions.txt
        std::vector<ClassifiedParser::Classifier_Entry> entries;
        std::cout << "[main] Parsing classifier output" << std::endl;
        ClassifiedParser::parse_file(ClassifiedParser::get_file_content("predictions.txt"), entries);
        if(entries.empty()) {
            std::cerr << "[main] Error: entries is empty\n";
            return 0;
        }
        std::cout << "[main] Classifier output parsed successfully. Number of entries: " << entries.size() << std::endl;
        // Draw and paint entries rects
        cv::Mat result = test_image.clone();
        for(const auto& e : entries) {
            cv::Scalar color = e.infected ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(result, e.rect, color, 2);
            // print confidence on the top of the rectangle
            std::string label = std::to_string(e.confidence) + "%";
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(result, cv::Point(e.rect.x, e.rect.y - labelSize.height - baseLine), cv::Point(e.rect.x + labelSize.width, e.rect.y), color, cv::FILLED);
            cv::putText(result, label, cv::Point(e.rect.x, e.rect.y - baseLine), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        // save image int outputs/[original_image_name]_result.jpg
        if(!std::filesystem::exists("OutputImages/Classifier/")) {
            std::filesystem::create_directory("OutputImages/Classifier/");
        }
        std::string outputImageName = "OutputImages/Classifier/" + std::filesystem::path(argv[IMG_ARG]).stem().string() + "_result.jpg";
        cv::imwrite(outputImageName, result);
        std::cout << "[main] Result image saved to: " << outputImageName << std::endl;

        img_show(result, "result with classifier", true);
        
    } else {
        // Upload file and get original dataset bounding boxes
        std::vector<Parser::GT_Entry> entries;
        Parser::parse_file(Parser::get_file_content(dataset_location + "/GT/" + std::string(argv[IMG_ARG]) + ".txt"), entries);
        
        if(entries.empty()) {
            std::cerr << "[main] Error: entries is empty\n";
            return 0;
        }

        // Populate GT bounding boxes vector
        std::vector<cv::Rect> GT_Bboxes;
        for(const auto &e : entries) GT_Bboxes.push_back(std::move(cv::boundingRect(e.points)));


        // -------------------------------------------------------------------------------------------
        // Compute IoU over acquired data
        std::vector<double> out(std::move(Utils::compute_IoU(GT_Bboxes, waterBoxes)));

        // Results ****
        std::cout << "size: " << waterBoxes.size() << std::endl;
        std::cout << "size: " << out.size() << std::endl;
        int i=0; double count=0;
        for(const auto& d : out)
            if(d > 0) {
                count += d;
                i++;
            }
        std::cout << "Matched: " << std::to_string(count / i * 100) << std::endl;
        std::cout << "Global: " << std::to_string(count / out.size() * 100) << std::endl;
    }
    

    int key;
    while((key = cv::waitKey(0)) != 27 && key != -1); // 27 is ESC, -1 is window close

    return 0;
}
