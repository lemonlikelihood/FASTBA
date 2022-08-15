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
    std::shared_ptr<Image> read_image();
    std::unique_ptr<DatasetConfigurator> dataset_config;
    Pose get_groundtruth_pose(double t) const;

private:
    std::string m_euroc_path;
    std::deque<IMUData> m_imu_deque;
    std::deque<CameraCsvData> m_image_deque;
    std::map<double, Pose> m_gt_pose_map;
};
