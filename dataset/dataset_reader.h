#ifndef FASTBA_DATASET_READER_H
#define FASTBA_DATASET_READER_H

#include <Eigen/Eigen>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

class DatasetReader {
public:
    enum NextDataType { AGAIN, GYROSCOPE, ACCELEROMETER, ATTITUDE, GRAVITY, IMAGE, END };
    virtual ~DatasetReader() = default;
    virtual NextDataType next() = 0;
    virtual std::pair<double, Eigen::Vector3d> read_gyroscope() = 0;
    virtual std::pair<double, Eigen::Vector3d> read_accelerometer() = 0;
    virtual std::pair<double, std::shared_ptr<cv::Mat>> read_image() = 0;

    static std::unique_ptr<DatasetReader>
    create_reader(const std::string &type, const std::string &filename);
    // virtual std::pair<double,>
};

#endif