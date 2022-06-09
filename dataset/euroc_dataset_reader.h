#pragma once

#include "dataset_reader.h"

#include <deque>
#include <string>

class EurocDatasetReader : public DatasetReader {
public:
    EurocDatasetReader(const std::string &filename);
    ~EurocDatasetReader();
    NextDataType next() override;
    std::pair<double, Eigen::Vector3d> read_gyroscope() override;
    std::pair<double, Eigen::Vector3d> read_accelerometer() override;
    std::pair<double, std::shared_ptr<cv::Mat>> read_image() override;

private:
    std::deque<std::pair<double, NextDataType>> all_data;
    std::deque<std::pair<double, Eigen::Vector3d>> gyroscope_data;
    std::deque<std::pair<double, Eigen::Vector3d>> accelerometer_data;
    std::deque<std::pair<double, std::string>> image_data;
};
