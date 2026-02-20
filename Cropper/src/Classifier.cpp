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

    //TODO: pulire!!!!!!!!!
    /*cv::Mat resized, floatImg;
    cv::resize(img, resized, cv::Size(256, 256));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB); // Fix the BGR -> RGB swap
    resized.convertTo(floatImg, CV_32FC3); // Model expects floats

    // 2. Wrap the Mat in a 4D Blob WITHOUT rearranging channels
    // The constructor: Mat(dims, sizes, type, data_pointer)
    int sizes[] = {1, 256, 256, 3};
    cv::Mat blob(4, sizes, CV_32F, floatImg.data);

    // 3. Forward pass
    model.setInput(blob, "keras_tensor_276");
    cv::Mat output = model.forward("output_0");*/

    // 1. Load image and resize
    // cv::Mat img = cv::imread("cell.png");
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256));

    // 2. Convert Color (Essential!)
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 3. Convert to Float 
    // IMPORTANT: Check if you need 1.0 or 1.0/255.0 here.
    // Since your ONNX has a 'Mul' node, we use 1.0 (keep values 0-255).
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0); 

    // 4. Create a 4D Blob that is "Interleaved" (NHWC)
    // We use the 'blobFromImages' (plural) or manual construction to avoid 
    // the auto-transpose that happens in 'blobFromImage' (singular).
    int sz[] = {1, 256, 256, 3};
    cv::Mat inputBlob = cv::Mat(4, sz, CV_32F);

    // Copy the data manually to ensure the memory layout is correct and persistent
    memcpy(inputBlob.data, floatImg.data, floatImg.total() * floatImg.elemSize());

    // 5. Run Inference
    cv::Scalar meanVal = cv::mean(floatImg);

    std::cout << "Input Mean (should be ~100-200 if 0-255 range): " << meanVal << std::endl;
    model.setInput(inputBlob);
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
