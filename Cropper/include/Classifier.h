#ifndef Classifier_h
#define Classifier_h

#include <iostream>
#include <opencv2/dnn.hpp>


struct Classifier_Output {
    std::string label;
    bool parasitized;
    float probability;
    std::string confidence;
};

class Classifier {
private:

    cv::dnn::Net model;
    bool model_loaded = false;

public:
	void load_model(const std::string& model_path);
    Classifier_Output classify(const cv::Mat& img);
};

#endif