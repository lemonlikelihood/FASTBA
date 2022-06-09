#pragma once

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <vector>

struct Pose {
    double t;
    Eigen::Quaterniond q;
    Eigen::Vector3d p;
};

class TrajectoryReader {
public:
    virtual ~TrajectoryReader() = default;
    virtual void read_poses(std::vector<double> &timestamps, std::vector<Pose> &poses) = 0;
};

class TumTrajectoryReader : public TrajectoryReader {
    std::ifstream file;

public:
    TumTrajectoryReader(const std::string &filename) {
        file.open(filename.c_str());
        if (not file.is_open()) {
            std::cerr << "Cannot open file " << filename << std::endl;
        }
    }

    ~TumTrajectoryReader() = default;

    void read_poses(std::vector<double> &timestamps, std::vector<Pose> &poses) override {
        double t;
        Pose pose;
        while (not file.eof()) {
            file >> t >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >> pose.q.y()
                >> pose.q.z() >> pose.q.w();
            timestamps.push_back(t);
            pose.q.normalize();
            poses.push_back(pose);
            // std::cout << t << " " << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " " << pose.q.x() << " " << pose.q.y() << " " << pose.q.z() << " " << pose.q.w() << "\n";
        }
        file.close();
    }
};
