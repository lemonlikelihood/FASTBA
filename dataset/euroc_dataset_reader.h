#pragma once

#include "dataset.h"

#include "configurator.h"
#include <deque>
#include <iostream>
#include <string>

class EurocDatasetReader {
public:
    EurocDatasetReader(const std::string &filename);
    ~EurocDatasetReader();
    NextDataType next();
    IMUData read_imu();
    std::shared_ptr<ImageData> read_image();
    void preprocess_image(std::shared_ptr<ImageData> image);
    std::unique_ptr<DatasetConfigurator> dataset_config;

private:
    std::string m_euroc_path;
    std::deque<IMUData> m_imu_deque;
    std::deque<CameraCsvData> m_image_deque;
};
