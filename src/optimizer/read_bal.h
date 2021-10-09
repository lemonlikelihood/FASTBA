//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_READ_BAL_H
#define FASTBA_READ_BAL_H

#include "../utils/read_file.h"
#include "map.h"

class BalReader{
public:
    BalReader(std::string &path);
    BalReader();
    ~BalReader();

    bool read_map();
    std::string path;
    std::unique_ptr<Map> map;

};
#endif //FASTBA_READ_BAL_H
