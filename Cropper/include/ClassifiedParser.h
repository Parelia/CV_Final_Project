/* 
	###############################
    #  Author: Gabriele Trevisan  #
    ###############################
*/

#ifndef ClassifiedParser_h
#define ClassifiedParser_h

#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace ClassifiedParser {
    struct Classifier_Entry{
        bool infected;
        double confidence;
        cv::Rect rect;
    };

    // Handle and extract .txt file content
    std::string get_file_content(const std::filesystem::path&);
    
    // Parse an entry file of Classifier_Entry and save them on dst
    void parse_file(const std::string&, std::vector<Classifier_Entry>&);
    
    // Parse a single Classifier_Entry entry
    int parse_entry(const std::string&, Classifier_Entry&);

    // View entry content
    std::string view_Classifier_Entry(const Classifier_Entry&);
};


#endif
