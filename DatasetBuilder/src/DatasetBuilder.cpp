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
#include <random>

/**
 * @brief Builds a cropped dataset from the original dataset, given the ground truth and the images folders.
 * The cropped images are saved in the specified output folders.
 * @param groundThruth Path to the folder containing the ground truth .txt files.
 * @param ImagesFolder Path to the folder containing the original images.
 * @param parasitizedOutput Path to the folder where the cropped parasitized images will be saved
 * @param uninfectedOutput Path to the folder where the cropped uninfected images will be saved
 */
void DatasetBuilder::build_cropped_dataset(
    std::string groundThruth,
    std::string ImagesFolder,
    std::string cropperTestFolder,      // here separately the images for testing
    std::string parasitizedOutputTrain,
    std::string uninfectedOutputTrain,
    std::string parasitizedOutputVal,
    std::string uninfectedOutputVal,
    std::string parasitizedOutputTest,
    std::string uninfectedOutputTest,
    float trainSetPercentage
) {

    std::random_device rd; 
    std::mt19937 prng(rd());
    
    // process folders for the training and validation
    process_folder(
        ImagesFolder, groundThruth,
        parasitizedOutputTrain, uninfectedOutputTrain,
        parasitizedOutputVal,   uninfectedOutputVal,
        prng, trainSetPercentage
    );

    // Apply the remaining transforms to all the parasitized images (also val time augmentation)
    std::cout << "Performing data augmentation on train and validation set, separately!" << std::endl;
    for(const auto& entry : std::filesystem::directory_iterator(parasitizedOutputTrain)) {
        if(entry.path().extension() == ".jpg") {
            apply_transform(entry.path().string(), Transform::ROTATIONS_0_90);
        }
    }
    for(const auto& entry : std::filesystem::directory_iterator(parasitizedOutputTrain)) {
        if(entry.path().extension() == ".jpg") {
            apply_transform(entry.path().string(), Transform::ELASTIC_DEFORMATIONS);
        }
    }
    for(const auto& entry : std::filesystem::directory_iterator(parasitizedOutputVal)) {
        if(entry.path().extension() == ".jpg") {
            apply_transform(entry.path().string(), Transform::ROTATIONS_0_90);
        }
    }
    for(const auto& entry : std::filesystem::directory_iterator(parasitizedOutputVal)) {
        if(entry.path().extension() == ".jpg") {
            apply_transform(entry.path().string(), Transform::ELASTIC_DEFORMATIONS);
        }
    }

    // test set: no split
    process_folder(
        cropperTestFolder, groundThruth + "../CropperTest/GT/",
        parasitizedOutputTest, uninfectedOutputTest,    // A = test
        "", "",                                         // B = unused
        prng, 1.0f                                      // always goes to A
    );

    return;
}


void DatasetBuilder::process_folder(
    const std::string& imagesFolder,
    const std::string& groundTruthFolder,
    const std::string& parasitizedOutputA,  // train or test
    const std::string& uninfectedOutputA,
    const std::string& parasitizedOutputB,  // val or empty
    const std::string& uninfectedOutputB,
    std::mt19937& prng,
    float probabilityA                      // probability of going to A
) {
    std::cout << "Processing folder: " << imagesFolder << ", using gt from: " << groundTruthFolder << std::endl;
    std::bernoulli_distribution distrib(probabilityA);

    std::vector<std::filesystem::path> files;
    for(const auto& entry : std::filesystem::directory_iterator(imagesFolder))
        if(entry.path().extension() == ".jpg")
            files.push_back(entry.path().stem());

    for(const auto& file : files) {
        cv::Mat image = cv::imread(imagesFolder + "/" + file.string() + ".jpg");
        if(image.empty()) {
            std::cerr << "[DatasetBuilder] Error: image " << file.string() << " is empty" << std::endl;
            continue;
        }

        // GT filename matches image filename
        std::vector<Parser::GT_Entry> entries;

        // get the ground truth (names match)
        Parser::parse_file(Parser::get_file_content(groundTruthFolder + file.string() + ".txt"), entries);

        for(const auto& e : entries) {
            cv::Rect bbox = cv::boundingRect(e.points);
            cv::Mat cropped = image(bbox);
            std::string name = file.string() + "_" + std::to_string(e.n_cell) + ".jpg";

            bool goesToA = distrib(prng);
            std::string parasitizedOut = goesToA ? parasitizedOutputA : parasitizedOutputB;
            std::string uninfectedOut  = goesToA ? uninfectedOutputA  : uninfectedOutputB;

            if(e.infected) {
                if(!cv::imwrite(parasitizedOut + name, cropped))
                    std::cerr << "[DatasetBuilder] Saving parasitized error: " << name << std::endl;
                apply_transform(parasitizedOut + name, Transform::FLIPS);
            } else {
                if(!cv::imwrite(uninfectedOut + name, cropped))
                    std::cerr << "[DatasetBuilder] Saving uninfected error: " << name << std::endl;
            }
        }
    }
}

/**
 * @brief Applies a specified transformation to an image and saves the transformed image with an descriptive name.
 * @param inputImagePath Path to the input image to be transformed.
 * @param transformType Type of transformation to be applied (FLIPS, ROTATIONS_0_90, ELASTIC_DEFORMATIONS).
 */
void DatasetBuilder::apply_transform(std::string inputImagePath, Transform transformType) {
    cv::Mat image = cv::imread(inputImagePath);
    if(image.empty()) {
        std::cerr << "[DatasetBuilder] Error: image " << inputImagePath << " is empty" << std::endl;
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
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << std::endl;
            }
            // Vertical flip
            cv::flip(image, transformedImage, 0);
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_v" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << std::endl;
            }
            // Both 
            cv::flip(image, transformedImage, -1);
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_hv" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << std::endl;
            }
            break;
        case Transform::ROTATIONS_0_90:
            cv::rotate(image, transformedImage, cv::ROTATE_90_CLOCKWISE);
            outputPath = inputImagePath.substr(0, inputImagePath.find_last_of('.')) + "_90" + ".jpg";
            if(!cv::imwrite(outputPath, transformedImage)) {
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << std::endl;
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
                std::cerr << "[DatasetBuilder] Saving transformed image error: " << outputPath << std::endl;
            }
            break;
        default:
            std::cerr << "[DatasetBuilder] Error: Unknown transform type" << std::endl;
            return;
    }

    return;
}