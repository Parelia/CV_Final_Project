#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

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

    // Resize and normalize (Standard OpenCV is HWC: Height, Width, Channels)
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(256, 256));
    
    // Note: We keep it in BGR or convert to RGB depending on model needs. 
    // Most Keras models need RGB.
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB); 
    
    // Wrap as 4D Blob with NHWC Layout [1, 256, 256, 3]
    // We do NOT use blobFromImage here because it forces a transpose to NCHW.
    int dimensions[4] = {1, 256, 256, 3};

    cv::Mat blob = cv::Mat(4, dimensions, CV_32F, resized_img.data).clone();

    model.setInput(blob);
    cv::Mat output = model.forward();

    // ---------------------------------------------------------------------------------------
    // The output logic remains the same
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
