/* 
	###############################
    #  Author: Gabriele Trevisan  #
    ###############################
*/

#include "ClassifiedParser.h"

// Handle and extract .txt file content
std::string ClassifiedParser::get_file_content(const std::filesystem::path& file_path) {
	// 1st check
    if(!std::filesystem::exists(file_path)) {
        std::cerr << "[ClassifiedParser] Error: invalid file path\n";
        return "";
    }
    if(file_path.extension() != ".txt") {
        std::cerr << "[ClassifiedParser] Error: invalid file type\n";
        return "";
    }

    std::ifstream file_stream = std::ifstream(file_path);
	
	// 2nd check
    if(!file_stream.is_open() || !file_stream.good()){
        std::cerr << "[ClassifiedParser] Error: could not read from the filestream\n";
        return "";
    }

    std::stringstream data;
    data << file_stream.rdbuf();
    file_stream.close();

    return data.str();
}

// Parse an entiry file of Classifier_Entry and save them on dst
void ClassifiedParser::parse_file(const std::string& file_content, std::vector<ClassifiedParser::Classifier_Entry>& dst) {
    std::stringstream read_content(file_content);
    std::string temp;
    std::getline(read_content, temp, '\n');

    int start_index = temp.size() + 1, progression_counter = 1;
    ClassifiedParser::Classifier_Entry entry;
    while(start_index < file_content.length() && progression_counter > 0){
        progression_counter = parse_entry(file_content.substr(start_index), entry);
        dst.push_back(std::move(entry));
        start_index += progression_counter; // skip to newline
    }
    return;
}

// Parse a single Classifier_Entry entry
int ClassifiedParser::parse_entry(const std::string& content, ClassifiedParser::Classifier_Entry& entry) {
    std::stringstream read_content(content);
    std::string token;
    int counter = 0;

    // Element before points
    std::getline(read_content, token, ',');
    counter += token.size() + 1;
    if(token == "Uninfected") entry.infected = 0;
    else entry.infected = 1;

    std::getline(read_content, token, ',');
    counter += token.size() + 1;
    entry.confidence = std::stod(token.substr(0, token.size() - 1)); // remove %

    std::getline(read_content, token, '\n');
    counter += token.size() + 1;
    std::stringstream points_stream(token);
    points_stream >> entry.rect.x >> entry.rect.y >> entry.rect.width >> entry.rect.height;

    return counter;
}

// View entry content
std::string ClassifiedParser::view_Classifier_Entry(const ClassifiedParser::Classifier_Entry& entry) {
    std::string temp = "";
    temp += "infected: " + std::to_string(entry.infected) + ", ";
    temp += "confidence: " + std::to_string(entry.confidence) + ", ";
    temp += "rect: (" + std::to_string(entry.rect.x) + ",";
    temp += std::to_string(entry.rect.y) + ",";
    temp += std::to_string(entry.rect.width) + ",";
    temp += std::to_string(entry.rect.height) + ")";
    
    return temp;
}
