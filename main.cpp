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

// Compact the code
void img_show(cv::Mat img, std::string window_name) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1200, 700);
    cv::imshow(window_name, img);

    return;
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cerr << "[main] Missing input image (required)\n";
        return -1;
    }
    
    cv::Mat test_image = cv::imread("../Malaria_Dataset/Img/" + std::string(argv[1]) + ".jpg");
    
    if(test_image.empty()) {
        std::cerr << "[main] Error: test_image is empty\n";
		return 0;
     }
     
    
    // Cut image
    cv::Mat cutted_image;
    cv::Point diff;
    Utils::cut_background(test_image, cutted_image, diff);

    // Thresholding
    cv::Mat thresh;
    Utils::refined_threshold(cutted_image, thresh);
    //img_show(thresh, "thresh");


    // -------------------------------------------------------------------------------------------
    // Find connected components
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    if(cv::connectedComponentsWithStats(thresh, labels, stats, centroids) == 0) {
        std::cerr << "[main] Error: connectedComponentsWithStats output is 0\n";
		return 0;
    }
    
    // Discard imperfections (small elements)
    for(int i = 1; i < stats.rows; i++) { // label 0 = background
        if(stats.at<int>(i, cv::CC_STAT_AREA) < 500) {
            labels.setTo(0, labels == i);
            stats.at<int>(i, cv::CC_STAT_AREA) = 0;
        }
    }

    // View results
    cv::Mat markers;
    labels.convertTo(markers, CV_8U);
    img_show(markers, "markers");
    // -------------------------------------------------------------------------------------------
 

    // -------------------------------------------------------------------------------------------
    // Apply watershed #1
    cv::watershed(cutted_image, labels);
    
    labels.convertTo(markers, CV_8U);
    img_show(markers, "markers after w1");
    
    
    for(int i = 1; i < stats.rows; i++) {  // label 0 = background
        if(stats.at<int>(i, cv::CC_STAT_AREA) != 0) {
            cv::Mat mask = (labels == i);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if(!contours.empty()) {
				cv::Rect temp = cv::boundingRect(contours[0]);
				if(temp.area() < 800 || temp.area()  > 100000)
					labels.setTo(0, labels == i);
			}
        }	
    }
    
    
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

	// View results
    cv::Mat markers2;
	labels2.convertTo(markers2, CV_8U);
    img_show(markers2, "markers before water2");
	
	
	// Apply watershed #2
    cv::watershed(cutted_image, labels2);
	
	// Compute bounding boxes
	cv::Mat result = test_image.clone();
	std::vector<cv::Rect> waterBoxes;
	for(int i = 1; i < stats2.rows; i++) {  // label 0 = background
        if(stats2.at<int>(i, cv::CC_STAT_AREA) != 0) {
            cv::Mat mask = (labels2 == i);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if(!contours.empty()) {
				cv::Rect temp = cv::boundingRect(contours[0]);
				waterBoxes.push_back(temp + diff);
				
				cv::rectangle(result, temp + diff, cv::Scalar(0, 255, 0), 2); //************
			}
        }	
    }
    
    if(waterBoxes.empty()) {
        std::cerr << "[main] Error: waterBoxes is empty\n";
		//return 0;
     }
	
	// View results
    cv::Mat markers3;
    labels2.convertTo(markers3, CV_8U);
    img_show(markers3, "markers after water2");
    
 
    // Draw and paint borders(-1) **
   for (int i = 0; i < labels2.rows; i++) {
        for (int j = 0; j < labels2.cols; j++) {
            if (labels2.at<int>(i, j) == -1) {
                result.at<cv::Vec3b>(i + diff.y, j + diff.x) = cv::Vec3b(255, 0, 0); // border
            }
        }
    }
    img_show(result, "result");
    // -------------------------------------------------------------------------------------------


    // -------------------------------------------------------------------------------------------
    // Upload file and get original dataset bounding boxes
    std::vector<Parser::GT_Entry> entries;
    Parser::parse_file(Parser::get_file_content("../Malaria_Dataset/GT/" + std::string(argv[1]) + ".txt"), entries);
    
    if(entries.empty()) {
        std::cerr << "[main] Error: entries is empty\n";
		return 0;
     }

	// Populate GT bounding boxes vector
    std::vector<cv::Rect> GT_Bboxes;
    for(const auto &e : entries) GT_Bboxes.push_back(std::move(cv::boundingRect(e.points)));


    // Compute new dataset for deep learning
    std::string target_parasitized = "../" + std::string(argv[2]) + "/Parasitized";
    if(!std::filesystem::exists(target_parasitized)) std::filesystem::create_directories(target_parasitized);
    std::string target_uninfected = "../" + std::string(argv[2]) + "/Uninfected";
    if(!std::filesystem::exists(target_uninfected)) std::filesystem::create_directories(target_uninfected);
    
	for(int i = 0; i < GT_Bboxes.size(); i++) {
		if(entries[i].infected) {
			if(!cv::imwrite(target_parasitized + "/" + std::string(argv[1]) + "_" + std::to_string(entries[i].n_cell) + ".jpg", test_image(GT_Bboxes[i])))
				std::cerr << "Saving image error" << std::endl;
		}
		else {
			if(!cv::imwrite(target_uninfected + "/" + std::string(argv[1]) + "_" + std::to_string(entries[i].n_cell) + ".jpg", test_image(GT_Bboxes[i])))
				std::cerr << "Saving image error" << std::endl;
		}
	}
    // -------------------------------------------------------------------------------------------


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
    // -------------------------------------------------------------------------------------------

            
    cv::waitKey(0);
    return 0;
}
