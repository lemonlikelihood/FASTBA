//
// Created by lemon on 2020/9/22.
//

#ifndef BA_COST_IO_COLMAP_H
#define BA_COST_IO_COLMAP_H

#include <iostream>
#include <fstream>
#include <sstream>
#include "map.h"
#include "utils/tic_toc.h"

namespace colmap {
    void read_cameras(const std::string &cameras_file, Map &map);

    void read_images(const std::string &images_file, Map &map);

    void read_points3D(const std::string &points3D_file, Map &map);

    void read_map(std::string &path, Map &map);

    void associate_OBS(Map &map);

    double calculate_reprojection_error(Map &map);
}
#endif //BA_COST_IO_COLMAP_H
