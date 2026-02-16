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
    // Handle and extract .txt file content
    std::string get_file_content(const std::filesystem::path&);

    // ----------------------------------------- GT Entry ----------------------------------------
    struct GT_Entry{
        int n_patient;
        int n_cell;
        bool infected;
        std::string comment;
        std::string shape;
        int n_points;
        std::vector<cv::Point> points;
    };

    // Parse an entiry file of GT_Entry and save them on target vector
    void parse_file(const std::string&, std::vector<GT_Entry>&);
    
    // Parse a single GT_Entry entry
    int parse_entry(const std::string&, GT_Entry&);

    // Compute output string with complete GT_Entry content
    std::string view_GT_Entry(const GT_Entry&);


    // ------------------------------------- Classifier Entry ------------------------------------
    struct Classifier_Entry{
        bool infected;
        double confidence;
        cv::Rect rect;
    };

    // Parse an entry file of Classifier_Entry and save them on dst
    void parse_file(const std::string&, std::vector<Classifier_Entry>&);
    
    // Parse a single Classifier_Entry entry
    int parse_entry(const std::string&, Classifier_Entry&);

    // Compute output string with complete Classifier_Entry content
    std::string view_Classifier_Entry(const Classifier_Entry&);
};


#endif
