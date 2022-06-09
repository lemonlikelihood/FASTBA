#include "dior_dataset_reader.h"

#include "dior.h"

#include <opencv2/opencv.hpp>

DiorDatasetReader::DiorDatasetReader(const std::string &dior_path) {
    CameraCsv cam_csv;
    cam_csv.load(dior_path + "/camera/data.csv");
    ImuCsv imu_csv;
    imu_csv.load(dior_path + "/imu/data.csv");
    AttitudeCsv att_csv;
    att_csv.load(dior_path + "/attitude/data.csv");

    for (auto &item : cam_csv.items) {
        image_data.emplace_back(item.t, dior_path + "/camera/images/" + item.filename);
        all_data.emplace_back(item.t, NextDataType::IMAGE);
    }

    for (auto &item : imu_csv.items) {
        Eigen::Vector3d gyr = {item.w.x, item.w.y, item.w.z};
        gyroscope_data.emplace_back(item.t, gyr);
        all_data.emplace_back(item.t, NextDataType::GYROSCOPE);

        Eigen::Vector3d acc = {item.a.x, item.a.y, item.a.z};
        accelerometer_data.emplace_back(item.t, acc);
        all_data.emplace_back(item.t, NextDataType::ACCELEROMETER);
    }

    for (auto &item : att_csv.items) {
        Eigen::Vector3d gra = {item.g.x, item.g.y, item.g.z};
        gravity_data.emplace_back(item.t, gra);
        all_data.emplace_back(item.t, NextDataType::GRAVITY);

        Eigen::Vector4d atti = {item.atti.x, item.atti.y, item.atti.z, item.atti.w};
        attitude_data.emplace_back(item.t, atti);
        all_data.emplace_back(item.t, NextDataType::ATTITUDE);
    }

    std::sort(all_data.begin(), all_data.end(), [](auto &a, auto &b) { return a.first < b.first; });
    std::sort(
        image_data.begin(), image_data.end(), [](auto &a, auto &b) { return a.first < b.first; });
    std::sort(gyroscope_data.begin(), gyroscope_data.end(), [](auto &a, auto &b) {
        return a.first < b.first;
    });
    std::sort(accelerometer_data.begin(), accelerometer_data.end(), [](auto &a, auto &b) {
        return a.first < b.first;
    });
    std::sort(attitude_data.begin(), attitude_data.end(), [](auto &a, auto &b) {
        return a.first < b.first;
    });
    std::sort(gravity_data.begin(), gravity_data.end(), [](auto &a, auto &b) {
        return a.first < b.first;
    });
}

DiorDatasetReader::~DiorDatasetReader() = default;

DatasetReader::NextDataType DiorDatasetReader::next() {
    if (all_data.empty()) {
        return NextDataType::END;
    }
    auto [t, type] = all_data.front();
    return type;
}

std::pair<double, std::shared_ptr<cv::Mat>> DiorDatasetReader::read_image() {
    if (image_data.empty()) {
        return {-1, nullptr};
    }
    auto [t, filename] = image_data.front();
    cv::Mat cv_img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    std::shared_ptr<cv::Mat> image = std::make_shared<cv::Mat>();
    *image = cv_img;
    all_data.pop_front();
    image_data.pop_front();
    return {t, image};
}

std::pair<double, Eigen::Vector3d> DiorDatasetReader::read_gyroscope() {
    if (gyroscope_data.empty()) {
        return {};
    }
    auto item = gyroscope_data.front();

    all_data.pop_front();
    gyroscope_data.pop_front();
    return item;
}

std::pair<double, Eigen::Vector3d> DiorDatasetReader::read_accelerometer() {
    if (accelerometer_data.empty()) {
        return {};
    }
    auto item = accelerometer_data.front();

    all_data.pop_front();
    accelerometer_data.pop_front();
    return item;
}

std::pair<double, Eigen::Vector4d> DiorDatasetReader::read_attitude() {
    if (attitude_data.empty()) {
        return {};
    }
    auto item = attitude_data.front();

    all_data.pop_front();
    attitude_data.pop_front();
    return item;
}

std::pair<double, Eigen::Vector3d> DiorDatasetReader::read_gravity() {
    if (gravity_data.empty()) {
        return {};
    }
    auto item = gravity_data.front();

    all_data.pop_front();
    gravity_data.pop_front();
    return item;
}
