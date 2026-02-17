#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "Classifier.h"
#include "Utils.h"
#include "Parser.h"

void Classifier::load_model(const std::string &model_path)
{
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

Classifier_Output Classifier::classify(const cv::Mat &img)
{
    // -------------------------------- START DATA CLASSIFICATION --------------------------------
    bool  TEST_MODE = false;
    if (!TEST_MODE) {

        if (!model_loaded) {
            std::cerr << "[Classifier] Error: Model not loaded. Call load_model() first." << std::endl;
            throw std::runtime_error("Model not loaded");
        }

        // 1. Resize and normalize (Standard OpenCV is HWC: Height, Width, Channels)
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(256, 256));
        
        // Convert to float32 and normalize to [0, 1]
        // Note: We keep it in BGR or convert to RGB depending on model needs. 
        // Most Keras models need RGB.
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB); 
        resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);

        // 2. Wrap as 4D Blob with NHWC Layout [1, 256, 256, 3]
        // We do NOT use blobFromImage here because it forces a transpose to NCHW.
        int dimensions[4] = {1, 256, 256, 3};
        
        // Create a blob that shares memory with resized_img (careful with scope!)
        // or .clone() it to ensure memory safety.
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

        //     // --------------------------------- END DATA CLASSIFICATION ---------------------------------
    } else {
        // --------------------------------- START IoU COMPUTATION -----------------------------------

        // Upload file and get original dataset bounding boxes
        /*std::vector<Parser::GT_Entry> entries;
        Parser::parse_file(Parser::get_file_content(dataset_location + "/GT/" + std::string(argv[IMG_ARG]) + ".txt"), entries);

        if (entries.empty())
        {
            std::cerr << "[main] Error: entries is empty\n";
            return 0;
        }

        // Populate GT bounding boxes vector
        std::vector<cv::Rect> GT_Bboxes;
        for (const auto &entry : entries)
            GT_Bboxes.push_back(std::move(cv::boundingRect(entry.points)));

        // -------------------------------------------------------------------------------------------
        // Compute IoU over acquired data
        std::vector<double> out(std::move(Utils::compute_IoU(GT_Bboxes, waterBoxes)));

        // Show results
        std::cout << "Showing IoU results" << std::endl;
        std::cout << "Detected cells: " << waterBoxes.size() << " out of " << out.size() << std::endl;

        int i = 0;
        double count = 0;
        for (const auto &d : out)
            if (d > 0)
            {
                count += d;
                i++;
            }
        std::cout << "Matched elements percentage: " << std::to_string(std::round(count / i * 100000) / 1000) << "%" << std::endl;
        std::cout << "Overall percentage: " << std::to_string(std::round(count / out.size() * 100000) / 1000) << "%" << std::endl;
*/
        // ---------------------------------- END IoU COMPUTATION ------------------------------------
        return {};
    }
}
