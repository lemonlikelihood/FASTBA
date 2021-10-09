//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_READ_FILE_H
#define FASTBA_READ_FILE_H

#include <fstream>
#include <iostream>
#include <sstream>

std::stringstream get_line_ss(std::ifstream &fin,bool &valid);

#endif //FASTBA_READ_FILE_H
