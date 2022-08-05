#include "euroc_dataset_reader.h"

#include "dataset.h"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

EurocDatasetReader::EurocDatasetReader(const std::string &euroc_path) {
    dataset_config = std::make_unique<DatasetConfigurator>(DatasetType::EUROC, euroc_path);
    dataset_config->load_configurator();
    dataset_config->print_configurator();
    CameraCsv cam_csv;
    cam_csv.load(dataset_config->camera_datacsv_path());
    ImuCsv imu_csv;
    imu_csv.load(dataset_config->imu_datacsv_path());
    EurocPoseCsv pose_csv;
    std::string gt_pose_filename = euroc_path + "/mav0/state_groundtruth_estimate0/data.csv";
    pose_csv.load(gt_pose_filename);
    for (auto x : pose_csv.items) {
        m_gt_pose_map[x.t] = x;
    }

    m_image_deque = cam_csv.items;
    m_imu_deque = imu_csv.items;
}

EurocDatasetReader::~EurocDatasetReader() = default;

NextDataType EurocDatasetReader::next() {
    if (m_image_deque.size() == 0 && m_imu_deque.size() == 0) {
        return NextDataType::END;
    }
    double t_image = std::numeric_limits<double>::max(), t_imu = std::numeric_limits<double>::max();
    if (m_image_deque.size() > 0) {
        t_image = m_image_deque.front().t;
    }
    if (m_imu_deque.size() > 0) {
        t_imu = m_imu_deque.front().t;
    }
    if (t_imu < t_image) {
        return NextDataType::IMU;
    } else {
        return NextDataType::IMAGE;
    }
}

IMUData EurocDatasetReader::read_imu() {
    IMUData data = m_imu_deque.front();
    m_imu_deque.pop_front();
    return data;
}

// 去畸变和直方图均衡化
void EurocDatasetReader::preprocess_image(std::shared_ptr<ImageData> image) {
    cv::Mat new_image;
    cv::Mat cv_K(3, 3, CV_32FC1), cv_coeffs(1, 4, CV_32FC1);
    auto K = dataset_config->camera_intrinsic();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cv_K.at<float>(i, j) = K(i, j);
        }
    }
    auto distortion_coeffs = dataset_config->distortion_coeffs();
    for (int i = 0; i < 4; i++) {
        cv_coeffs.at<float>(i) = distortion_coeffs(i);
    }

    cv::undistort(image->image, new_image, cv_K, cv_coeffs);
    cv::equalizeHist(new_image, new_image);

    image->image = new_image;
}

std::shared_ptr<ImageData> EurocDatasetReader::read_image() {
    auto data = m_image_deque.front();
    m_image_deque.pop_front();
    std::string image_absolute_path = dataset_config->camera_data_path() + "/" + data.filename;
    // std::cout << "image_absolute_path: " << image_absolute_path << std::endl;
    auto image_data = std::make_shared<ImageData>(data.t, image_absolute_path);
    preprocess_image(image_data);
    return image_data;
}

Pose EurocDatasetReader::get_groundtruth_pose(double t) const {
    if (m_gt_pose_map.size() == 0) {
        Pose pose;
        pose.t = -1;
        pose.p.setZero();
        pose.q.setIdentity();
        return pose;
    } else if (m_gt_pose_map.count(t) > 0) {
        return m_gt_pose_map.at(t);
    } else {
        auto it = m_gt_pose_map.lower_bound(t);
        if (it == m_gt_pose_map.begin()) {
            return it->second;
        } else if (it == m_gt_pose_map.end()) {
            return m_gt_pose_map.rbegin()->second;
        } else {
            auto it0 = std::prev(it);
            double lambda = (t - it0->first) / (it->first - it0->first);
            Pose pose;
            pose.t = t;
            pose.q = it0->second.q.slerp(lambda, it->second.q);
            pose.p = it0->second.p * (1 - lambda) + it->second.p * lambda;
            return pose;
        }
    }
}
