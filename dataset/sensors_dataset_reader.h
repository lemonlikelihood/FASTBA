#pragma once

#include "dataset_reader.h"
#include <deque>
#include <fstream>

namespace libsensors {
class Sensors;
}

class SensorsDataParser;

class SensorsDatasetReader : public DatasetReader {
public:
    SensorsDatasetReader(const std::string &filename);
    ~SensorsDatasetReader();
    NextDataType next() override;
    std::pair<double, Eigen::Vector3d> read_gyroscope() override;
    std::pair<double, Eigen::Vector3d> read_accelerometer() override;
    std::pair<double, Eigen::Vector4d> read_attitude();
    std::pair<double, Eigen::Vector3d> read_gravity();
    std::pair<double, std::shared_ptr<cv::Mat>> read_image() override;

private:
    friend class SensorsDataParser;
    std::ifstream datafile;
    std::unique_ptr<libsensors::Sensors> sensors;
    std::deque<std::pair<double, Eigen::Vector3d>> pending_gyroscopes;
    std::deque<std::pair<double, Eigen::Vector3d>> pending_accelerometers;
    std::deque<std::pair<double, Eigen::Vector4d>> pending_attitudes;
    std::deque<std::pair<double, Eigen::Vector3d>> pending_gravities;
    std::deque<std::pair<double, std::shared_ptr<cv::Mat>>> pending_images;
};
