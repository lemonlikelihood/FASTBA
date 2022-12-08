//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_READ_FILE_H
#define FASTBA_READ_FILE_H

#include <fstream>
#include <iostream>
#include <sstream>

#include "../../dataset/dataset.h"

std::stringstream get_line_ss(std::ifstream &fin, bool &valid);

class TrajectoryWriter {
public:
    virtual ~TrajectoryWriter() = default;
    virtual void write_pose(const double &t, const Pose &pose) = 0;
};

class TumTrajectoryWriter : public TrajectoryWriter {
    std::ofstream file;

public:
    TumTrajectoryWriter(const std::string &filename) {
        file.open(filename.c_str());
        if (!file.is_open()) {
            std::cout << "Cannot open file " << std::quoted(filename) << std::endl;
        }
        file.precision(15);
    }

    ~TumTrajectoryWriter() = default;

    void write_pose(const double &t, const Pose &pose) override {
        file << t << " " << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " "
             << pose.q.x() << " " << pose.q.y() << " " << pose.q.z() << " " << pose.q.w() << "\n";
        file.flush();
    }
};

#endif //FASTBA_READ_FILE_H
