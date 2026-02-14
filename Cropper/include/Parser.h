/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#ifndef Parser_h
#define Parser_h

#include <string>
#include <vector>
#include <opencv2/core/types.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace Parser {
    struct GT_Entry{
        int n_patient;
        int n_cell;
        bool infected;
        std::string comment;
        std::string shape;
        int n_points;
        std::vector<cv::Point> points;
    };

    // Handle and extract .txt file content
    std::string get_file_content(const std::filesystem::path&);
    
    // Parse an entiry file of GT_Entry and save them on target vector
    void parse_file(const std::string&, std::vector<GT_Entry>&);
    
    // Parse a single GT_Entry entry
    int parse_entry(const std::string&, GT_Entry&);

    // Compute output string with complete GT_Entry content
    std::string view_GT_Entry(const GT_Entry&);
};


#endif
