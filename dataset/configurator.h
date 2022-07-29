#pragma once
#include <string>
#include <yaml-cpp/yaml.h>

enum DatasetType {
    EUROC,
    TUM,
    DIOR,
    SENSORS,
};

class DatasetConfigurator {

public:
    DatasetConfigurator(const DatasetType &dataset_type, const std::string &dataset_path)
        : m_dataset_type(dataset_type), m_dataset_path(dataset_path) {
        switch (dataset_type) {
            case DatasetType::EUROC: {
                m_camera_yaml_path = dataset_path + "/mav0/cam0/sensor.yaml";
                m_imu_yaml_path = dataset_path + "/mav0/imu0/sensor.yaml";
                m_camera_datacsv_path = dataset_path + "/mav0/cam0/data.csv";
                m_imu_datacsv_path = dataset_path + "/mav0/imu0/data.csv";
                m_camera_data_path = dataset_path + "/mav0/cam0/data";
            } break;
            default: {
            } break;
        }
    }

    void load_configurator() {
        try {
            YAML::Node camera_node = YAML::LoadFile(m_camera_yaml_path);

            YAML::Node k_node = camera_node["intrinsics"]; // 相机内参
            m_K.setIdentity();
            m_K(0, 0) = k_node[0].as<double>();
            m_K(1, 1) = k_node[1].as<double>();
            m_K(0, 2) = k_node[2].as<double>();
            m_K(1, 2) = k_node[3].as<double>();

            YAML::Node dc_node = camera_node["distortion_coefficients"]; // 畸变系数
            m_distortion_coeffs[0] = dc_node[0].as<double>();
            m_distortion_coeffs[1] = dc_node[1].as<double>();
            m_distortion_coeffs[2] = dc_node[2].as<double>();
            m_distortion_coeffs[3] = dc_node[3].as<double>();

            YAML::Node cbs_node = camera_node["T_BS"]["data"]; // 相机外参，camera to body
            Eigen::Matrix3d R_cam2body;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_cam2body(i, j) = cbs_node[i * 4 + j].as<double>();
                }
                m_p_cam2body(i) = cbs_node[i * 4 + 3].as<double>();
            }
            m_q_cam2body = R_cam2body;
        } catch (...) { // for the sake of example, we don't really handle the errors.
            throw;
        }

        try {
            YAML::Node imu_node = YAML::LoadFile(m_imu_yaml_path); // imu 外参 IMU to body
            YAML::Node ibs_node = imu_node["T_BS"]["data"];
            Eigen::Matrix3d R_imu2body;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_imu2body(i, j) = ibs_node[i * 4 + j].as<double>();
                }
                m_p_imu2body(i) = ibs_node[i * 4 + 3].as<double>();
            }
            m_q_imu2body = R_imu2body;

            const double sigma_correction = 1.0; // imu 噪声参数
            double sigma_w = imu_node["gyroscope_noise_density"].as<double>() * sigma_correction;
            double sigma_a =
                imu_node["accelerometer_noise_density"].as<double>() * sigma_correction;
            double sigma_bg = imu_node["gyroscope_random_walk"].as<double>() * sigma_correction;
            double sigma_ba = imu_node["accelerometer_random_walk"].as<double>() * sigma_correction;
            m_gyro_noise = sigma_w * sigma_w * Eigen::Matrix3d::Identity();
            m_accl_noise = sigma_a * sigma_a * Eigen::Matrix3d::Identity();
            m_gyro_random_walk = sigma_bg * sigma_bg * Eigen::Matrix3d::Identity();
            m_accl_random_walk = sigma_ba * sigma_ba * Eigen::Matrix3d::Identity();
        } catch (...) { throw; }
    }

    DatasetType dataset_type() const { return m_dataset_type; }
    std::string dataset_path() const { return m_dataset_path; }
    std::string camera_yaml_path() const { return m_camera_yaml_path; }
    std::string imu_yaml_path() const { return m_imu_yaml_path; }
    std::string camera_datacsv_path() const { return m_camera_datacsv_path; }
    std::string imu_datacsv_path() const { return m_imu_datacsv_path; }
    std::string camera_data_path() const { return m_camera_data_path; }
    // std::string imu_yaml_path() const { return m_imu_yaml_path; }

    Eigen::Matrix3d camera_intrinsic() const { return m_K; }

    Eigen::Vector4d distortion_coeffs() const { return m_distortion_coeffs; }

    Eigen::Quaterniond camera_to_center_rotation() const { return m_q_cam2body; }

    Eigen::Vector3d camera_to_center_translation() const { return m_p_cam2body; }

    Eigen::Quaterniond imu_to_center_rotation() const { return m_q_imu2body; }

    Eigen::Vector3d imu_to_center_translation() const { return m_p_imu2body; }

    Eigen::Matrix3d imu_gyro_white_noise() const { return m_gyro_noise; }

    Eigen::Matrix3d imu_accel_white_noise() const { return m_accl_noise; }

    Eigen::Matrix3d imu_gyro_random_walk() const { return m_gyro_random_walk; }

    Eigen::Matrix3d imu_accel_random_walk() const { return m_accl_random_walk; }

    void print_configurator() {
        std::cout << "K: "
                  << "\n"
                  << m_K << std::endl;
        std::cout << "distortion_coeffs: "
                  << "\n"
                  << m_distortion_coeffs.transpose() << std::endl;
        std::cout << "q_cam2body: "
                  << "\n"
                  << m_q_cam2body.coeffs() << std::endl;
        std::cout << "p_cam2body: "
                  << "\n"
                  << m_p_cam2body.transpose() << std::endl;
        std::cout << "q_imu2body: "
                  << "\n"
                  << m_q_imu2body.coeffs() << std::endl;
        std::cout << "p_imu2body: "
                  << "\n"
                  << m_p_imu2body.transpose() << std::endl;
        std::cout << "gyro_noise: "
                  << "\n"
                  << m_gyro_noise << std::endl;
        std::cout << "gyro_random_walk: "
                  << "\n"
                  << m_gyro_random_walk << std::endl;
        std::cout << "accl_noise;: "
                  << "\n"
                  << m_accl_noise << std::endl;
        std::cout << "accl_random_walk: "
                  << "\n"
                  << m_accl_random_walk << std::endl;
    }

private:
    DatasetType m_dataset_type;
    std::string m_dataset_path;
    std::string m_camera_yaml_path;
    std::string m_imu_yaml_path;
    std::string m_camera_datacsv_path;
    std::string m_imu_datacsv_path;
    std::string m_camera_data_path;
    // std::string m_imu_data_path;

    Eigen::Matrix3d m_K;
    Eigen::Vector4d m_distortion_coeffs;
    Eigen::Quaterniond m_q_cam2body;
    Eigen::Vector3d m_p_cam2body;
    Eigen::Quaterniond m_q_imu2body;
    Eigen::Vector3d m_p_imu2body;
    Eigen::Matrix3d m_gyro_noise;
    Eigen::Matrix3d m_gyro_random_walk;
    Eigen::Matrix3d m_accl_noise;
    Eigen::Matrix3d m_accl_random_walk;
};