/* 
    #############################
    #  Author: Parussolo Elia   #
    #############################
*/

#include "Parser.h"


// Handle and extract .txt file content
std::string Parser::get_file_content(const std::filesystem::path& file_path) {
	// Preliminary check on existence and type
    if(!std::filesystem::exists(file_path)) {
        std::cerr << "[Parser] Error: invalid file path \"" << file_path << "\"" << std::endl;
        return "";
    }
    if(file_path.extension() != ".txt") {
        std::cerr << "[Parser] Error: invalid file type\n";
        return "";
    }

    std::ifstream file_stream = std::ifstream(file_path);
	
	// Verify opening
    if(!file_stream.is_open() || !file_stream.good()){
        std::cerr << "[Parser] Error: could not read from the filestream\n";
        return "";
    }

    // Read content
    std::stringstream data;
    data << file_stream.rdbuf();
    file_stream.close();

    return data.str();
}



// ----------------------------------------- GT Entry ----------------------------------------

// Parse an entiry file of GT_Entry and save them on target vector
void Parser::parse_file(const std::string& file_content, std::vector<Parser::GT_Entry>& dst) {
    // Read file
    std::stringstream read_content(file_content);
    std::string temp;
    std::getline(read_content, temp, '\n');

    // Parse
    int start_index = temp.size() + 1, progression_counter = 1;
    Parser::GT_Entry entry;
    while(start_index < file_content.length() && progression_counter > 0){
        progression_counter = parse_entry(file_content.substr(start_index), entry);
        dst.push_back(std::move(entry));
        start_index += progression_counter; // move to newline
    }
    return;
}

// Parse a single GT_Entry entry
int Parser::parse_entry(const std::string& content, Parser::GT_Entry& entry) {
    std::stringstream read_content(content);
    std::string token;
    int counter = 0;

    // Element before points
    std::getline(read_content, token, '-');         // int n_patient
    counter += token.size() + 1;                    // +1 because "If the delimiter is found, it is extracted and discarded"
    entry.n_patient = stoi(token);

    
    std::getline(read_content, token, ',');         // int n_cell
    counter += token.size() + 1;
    entry.n_cell = stoi(token);


    std::getline(read_content, token, ',');         // bool infected;
    counter += token.size() + 1;
    if(token == "Uninfected") entry.infected = 0;
    else entry.infected = 1;


    std::getline(read_content, entry.comment, ','); // std::string comment
    counter += entry.comment.size() + 1;

    std::getline(read_content, entry.shape, ',');   // std::string shape
    counter += entry.shape.size() + 1;

    std::getline(read_content, token, ',');         // int n_points
    counter += token.size() + 1;
    entry.n_points = std::stoi(token);


    // Get all subject points
    std::string token2;
    std::getline(read_content, token2, '\n');       // std::vector<cv::Point> points;
    counter += token2.size() + 1;
    // Parse point line
    std::stringstream read_content2(token2);
    while(std::getline(read_content2, token, ',')) {
        double x = std::round(std::stod(token));    // width = cols
        
        std::getline(read_content2, token, ',');
        double y = std::round(std::stod(token));    // height = rows

        entry.points.emplace_back(cv::Point(x, y));
    }

    return counter;
}

// Compute output string with complete GT_Entry content
std::string Parser::view_GT_Entry(const Parser::GT_Entry& entry) {
    std::string temp = "";
    temp += std::to_string(entry.n_patient) + "-";
    temp += std::to_string(entry.n_cell) + ",";
    if(entry.infected)
        temp += "Parasitized,";
    else
        temp += "Uninfected,";
    temp += entry.comment + ",";
    temp += entry.shape + ",";
    temp += std::to_string(entry.n_points);

    for(const auto& p : entry.points) {
        temp += ",(" + std::to_string(p.x) + ",";
        temp += std::to_string(p.y) + ")";
    }
    
    return temp;
}
