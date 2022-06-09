#pragma once

#include "dataset_reader.h"

#include <deque>
#include <string>

class DiorDatasetReader : public DatasetReader {
public:
    DiorDatasetReader(const std::string &filename);
    ~DiorDatasetReader();
    NextDataType next() override;
    std::pair<double, Eigen::Vector3d> read_gyroscope() override;
    std::pair<double, Eigen::Vector3d> read_accelerometer() override;
    std::pair<double, Eigen::Vector3d> read_gravity();
    std::pair<double, Eigen::Vector4d> read_attitude();
    std::pair<double, std::shared_ptr<cv::Mat>> read_image() override;

private:
    std::deque<std::pair<double, NextDataType>> all_data;
    std::deque<std::pair<double, Eigen::Vector3d>> gyroscope_data;
    std::deque<std::pair<double, Eigen::Vector3d>> accelerometer_data;
    std::deque<std::pair<double, Eigen::Vector4d>> attitude_data;
    std::deque<std::pair<double, Eigen::Vector3d>> gravity_data;
    std::deque<std::pair<double, std::string>> image_data;
};
