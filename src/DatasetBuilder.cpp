/* 
	###############################
    #  Author: Gabriele Trevisan  #
    ###############################
*/
#include "DatasetBuilder.h"
#include "Parser.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void DatasetBuilder::build_cropped_dataset(std::string groundThruth, std::string ImagesFolder, std::string parasitizedOutput, std::string uninfectedOutput) {
    std::vector<std::filesystem::path> files;
    for(const auto& entry : std::filesystem::directory_iterator(ImagesFolder)) {
        if(entry.path().extension() == ".jpg") {
            files.push_back(entry.path().stem());
        }
    }

    for(const auto& file : files) {
        cv::Mat image = cv::imread(ImagesFolder + "/" + file.string() + ".jpg");
        if(image.empty()) {
            std::cerr << "[DatasetBuilder] Error: image " << file.string() << " is empty\n";
            continue;
        }

        std::vector<Parser::GT_Entry> entries;
        Parser::parse_file(Parser::get_file_content(groundThruth + file.string() + ".txt"), entries);

        for(const auto &e : entries) {
            cv::Rect bbox = cv::boundingRect(e.points);
            cv::Mat cropped_image = image(bbox);
            std::string name = file.string() + "_" + std::to_string(e.n_cell) + ".jpg";
            if(e.infected) {
                if(!cv::imwrite(parasitizedOutput + "/" + name, cropped_image)) {
                    std::cerr << "[DatasetBuilder] Saving parasitized image error: " << file.string() << "\n";
                }

                apply_transform(parasitizedOutput + "/" + name, Transform::FLIPS);

            }
            else {
                if(!cv::imwrite(uninfectedOutput + "/" + name, cropped_image)) {
                    std::cerr << "[DatasetBuilder] Saving uninfected image error: " << file.string() << "\n";
                }
            }
        }
    }

    // Apply the remaining transforms to all the parasitized images
    for(const auto& entry : std::filesystem::directory_iterator(parasitizedOutput)) {
        if(entry.path().extension() == ".jpg") {
            apply_transform(entry.path().string(), Transform::ROTATIONS_0_90);
        }
    }
    for(const auto& entry : std::filesystem::directory_iterator(parasitizedOutput)) {
        if(entry.path().extension() == ".jpg") {
            apply_transform(entry.path().string(), Transform::ELASTIC_DEFORMATIONS);
        }
    }
    return;

}


void DatasetBuilder::apply_transform(std::string inputImagePath, Transform transformType) {
    cv::Mat image = cv::imread(inputImagePath);
    if(image.empty()) {
        std::cerr << "[DatasetBuilder] Error: image " << inputImagePath << " is empty\n";
        return;
    }
    std::string outputPath;
    cv::Mat transformedImage;
    switch(transformType) {
        case Transform::FLIPS:
            // Horizontal flip
            cv::flip(image, transformedImage, 1);
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_h" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << "\n";
            }
            // Vertical flip
            cv::flip(image, transformedImage, 0);
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_v" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << "\n";
            }
            // Both 
            cv::flip(image, transformedImage, -1);
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_hv" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << "\n";
            }
            break;
        case Transform::ROTATIONS_0_90:
            cv::rotate(image, transformedImage, cv::ROTATE_90_CLOCKWISE);
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_90" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << "\n";
            }
            break;
        case Transform::ELASTIC_DEFORMATIONS:
            {
                cv::Mat dx(image.size(), CV_32FC1), dy(image.size(), CV_32FC1);
                cv::randu(dx, -10.f, 10.f);
                cv::randu(dy, -10.f, 10.f);

                const int kernelSize = 45;
                const float alpha = 15.0f;

                cv::GaussianBlur(dx, dx, cv::Size(kernelSize, kernelSize), 0);
                cv::GaussianBlur(dy, dy, cv::Size(kernelSize, kernelSize), 0);
                dx *= alpha;
                dy *= alpha;

                cv::Mat mapX(image.size(), CV_32FC1), mapY(image.size(), CV_32FC1);
                for(int y = 0; y < image.rows; ++y) {
                    float* mx = mapX.ptr<float>(y);
                    float* my = mapY.ptr<float>(y);
                    const float* dxx = dx.ptr<float>(y);
                    const float* dyy = dy.ptr<float>(y);
                    for(int x = 0; x < image.cols; ++x) {
                        mx[x] = static_cast<float>(x) + dxx[x];
                        my[x] = static_cast<float>(y) + dyy[x];
                    }
                }

                cv::remap(image, transformedImage, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_REFLECT101);
            }
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_elastic" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << "\n";
            }
            break;
        default:
            std::cerr << "[DatasetBuilder] Error: Unknown transform type\n";
            return;
    }

    return;
}