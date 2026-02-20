/* 
	###############################
    #  Author: Gabriele Trevisan  #
    ###############################
*/

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Classifier.h"
#include "Utils.h"
#include "Parser.h"

void Classifier::load_model(const std::string &model_path) {
    if (model_loaded) {
        return;    
    }
    try {
        model = cv::dnn::readNetFromONNX(model_path);
        std::cout << "[Classifier] Model loaded successfully from: " << model_path << std::endl;
        model_loaded = true;
    } catch (const cv::Exception &e) {
        std::cerr << "[Classifier] Error loading model: " << e.what() << std::endl;
        model_loaded = false;
    }
}

Classifier_Output Classifier::classify(const cv::Mat &img) {
    if (!model_loaded) {
        std::cerr << "[Classifier] Error: Model not loaded. Call load_model() first." << std::endl;
        throw std::runtime_error("Model not loaded");
    }

   
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256));

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0); 

    int sz[] = {1, 256, 256, 3};
    cv::Mat inputBlob = cv::Mat(4, sz, CV_32F);

    memcpy(inputBlob.data, floatImg.data, floatImg.total() * floatImg.elemSize());

    model.setInput(inputBlob);
    cv::Mat output = model.forward();

    float score = output.at<float>(0, 0);
    std::string label;
    float probability;
    bool parasitized = false;

    if (score > 0.5f) {
        label = "Uninfected";
        probability = score;
    } else {
        label = "Parasitized";
        probability = 1.0f - score;
        parasitized = true;
    }

    std::string confidence;
    if (score >= 0.4f && score < 0.6f) {
        confidence = "UNCERTAIN";
    } else {
        confidence = "CONFIDENT";
    }

    return {label, parasitized, probability, confidence};
}
