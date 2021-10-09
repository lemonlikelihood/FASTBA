//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_READ_COLMAP_H
#define FASTBA_READ_COLMAP_H

#include "../utils/read_file.h"
#include "map.h"

class ColmapReader {
public:
    ColmapReader(std::string &path);

    ColmapReader();

    ~ColmapReader();

    void read_cameras(const std::string &cameras_file);

    void read_images(const std::string &images_file);

    void read_points_3D(const std::string &points3D_file);

    void read_map(std::string &path);

    void associate_OBS();

    bool read_map();

    std::string path;
    std::unique_ptr<Map> map;
};

#endif //FASTBA_READ_COLMAP_H
