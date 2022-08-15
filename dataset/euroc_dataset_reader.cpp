#include "euroc_dataset_reader.h"

#include "dataset.h"
#include "opencv_image.h"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

EurocDatasetReader::EurocDatasetReader(const std::string &euroc_path) {
    dataset_config = std::make_unique<DatasetConfigurator>(DatasetType::EUROC, euroc_path);
    dataset_config->load_configurator();
    dataset_config->print_configurator();
    CameraCsv cam_csv;
    bool is_ns = true;
    cam_csv.load(dataset_config->camera_datacsv_path(), is_ns);
    ImuCsv imu_csv;
    imu_csv.load(dataset_config->imu_datacsv_path(), is_ns);
    EurocPoseCsv pose_csv;
    std::string gt_pose_filename = euroc_path + "/mav0/state_groundtruth_estimate0/data.csv";
    pose_csv.load(gt_pose_filename);
    for (auto x : pose_csv.items) {
        // std::cout << x.t << " " << x.p << std::endl;
        // getchar();
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


std::shared_ptr<Image> EurocDatasetReader::read_image() {
    auto data = m_image_deque.front();
    m_image_deque.pop_front();
    std::string image_absolute_path = dataset_config->camera_data_path() + "/" + data.filename;
    // std::cout << "image_absolute_path: " << image_absolute_path << std::endl;
    // 去畸变和直方图均衡化
    auto image_data = std::make_shared<OpenCvImage>(data.t, image_absolute_path);
    image_data->correct_distortion(
        dataset_config->camera_intrinsic(), dataset_config->distortion_coeffs());
    image_data->preprocess();
    // preprocess_image(image_data);
    return image_data;
}

Pose EurocDatasetReader::get_groundtruth_pose(double t) const {
    std::cout.precision(19);
    // std::cout << "get_groundtruth_pose: " << t << std::endl;
    if (m_gt_pose_map.size() == 0) {
        Pose pose;
        pose.t = -1;
        pose.p.setZero();
        pose.q.setIdentity();
        return pose;
    } else if (m_gt_pose_map.count(t) > 0) {
        Pose pose;
        pose = m_gt_pose_map.at(t);
        pose.t = t;
        // std::cout << "same" << std::endl;
        // getchar();
        return pose;
    } else {
        auto it = m_gt_pose_map.lower_bound(t);
        if (it == m_gt_pose_map.begin()) {
            Pose pose = it->second;
            pose.t = t;
            // std::cout << "begin" << std::endl;
            // getchar();
            return pose;
        } else if (it == m_gt_pose_map.end()) {
            Pose pose = m_gt_pose_map.rbegin()->second;
            pose.t = t;
            // std::cout << "end" << std::endl;
            // getchar();
            return pose;
        } else {
            auto it0 = std::prev(it);
            double lambda = (t - it0->first) / (it->first - it0->first);
            Pose pose;
            pose.t = t;
            pose.q = it0->second.q.slerp(lambda, it->second.q);
            pose.p = it0->second.p * (1 - lambda) + it->second.p * lambda;
            // std::cout << "middle" << std::endl;
            // getchar();
            return pose;
        }
    }
}
