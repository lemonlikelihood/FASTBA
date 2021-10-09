//
// Created by lemon on 2021/1/29.
//

#include "read_file.h"

std::stringstream get_line_ss(std::ifstream &fin, bool &valid) {
    std::stringstream ss;
    std::string tmp_string;
    valid = true;
    if (!std::getline(fin, tmp_string)) {
        valid = false;
    }
    ss << tmp_string;
    return ss;
}