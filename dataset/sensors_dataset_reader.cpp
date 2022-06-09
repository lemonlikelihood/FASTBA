#include "sensors_dataset_reader.h"

#include "libsensors.h"

#include <array>
#include <iostream>

class SensorsDataParser : public libsensors::Sensors {
public:
    SensorsDataParser(SensorsDatasetReader *reader) : reader(reader) {}

protected:
    void on_gyroscope(double t, double x, double y, double z) override {
        reader->pending_gyroscopes.emplace_back(t, Eigen::Vector3d {x, y, z});
    }

    void on_accelerometer(double t, double x, double y, double z) override {
        reader->pending_accelerometers.emplace_back(t, Eigen::Vector3d {x, y, z});
    }

    void on_attitude(double t, double x, double y, double z, double w) override {
        reader->pending_attitudes.emplace_back(t, Eigen::Vector4d {x, y, z, w});
    }

    void on_gravity(double t, double x, double y, double z) override {
        reader->pending_gravities.emplace_back(t, Eigen::Vector3d {x, y, z});
    }

    void on_image(double t, int32_t width, int32_t height, const uint8_t *bytes) override {
        // std::shared_ptr<lvo::Image> image = std::make_shared<lvo::Image>();
        // image->t = t;
        // image->image = elvy::Mat(bytes, height, width);
        cv::Mat cv_img =
            cv::Mat(cv::Size(width, height), CV_8UC1, const_cast<uint8_t *>(bytes)).clone();
        std::shared_ptr<cv::Mat> image = std::make_shared<cv::Mat>();
        *image = cv_img;
        reader->pending_images.emplace_back(t, image);
    }

private:
    SensorsDatasetReader *reader;
};

SensorsDatasetReader::SensorsDatasetReader(const std::string &filename)
    : datafile(filename.c_str(), std::ifstream::in | std::ifstream::binary) {
    if (not datafile) {
        std::cout << "Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    sensors = std::make_unique<SensorsDataParser>(this);
}

SensorsDatasetReader::~SensorsDatasetReader() = default;

DatasetReader::NextDataType SensorsDatasetReader::next() {
    bool data_available = false;
    double gyroscope_time = std::numeric_limits<double>::max();
    double accelerometer_time = std::numeric_limits<double>::max();
    double attitude_time = std::numeric_limits<double>::max();
    double gravity_time = std::numeric_limits<double>::max();
    double image_time = std::numeric_limits<double>::max();
    if (pending_gyroscopes.size() > 0) {
        gyroscope_time = pending_gyroscopes.front().first;
        data_available = true;
    }
    if (pending_accelerometers.size() > 0) {
        accelerometer_time = pending_accelerometers.front().first;
        data_available = true;
    }
    if (pending_attitudes.size() > 0) {
        attitude_time = pending_attitudes.front().first;
        data_available = true;
    }
    if (pending_gravities.size() > 0) {
        gravity_time = pending_gravities.front().first;
        data_available = true;
    }
    if (pending_images.size() > 0) {
        image_time = pending_gravities.front().first;
        data_available = true;
    }
    if (data_available) {
        double next_time = gyroscope_time;
        DatasetReader::NextDataType next_type = DatasetReader::GYROSCOPE;
        if (next_time > accelerometer_time) {
            next_time = accelerometer_time;
            next_type = DatasetReader::ACCELEROMETER;
        }
        if (next_time > attitude_time) {
            next_time = attitude_time;
            next_type = DatasetReader::ATTITUDE;
        }
        if (next_time > gravity_time) {
            next_time = gravity_time;
            next_type = DatasetReader::GRAVITY;
        }
        if (next_time > image_time) {
            next_time = image_time;
            next_type = DatasetReader::IMAGE;
        }
        return next_type;
    } else {
        std::array<char, 8192> buffer;
        if (not datafile.read(buffer.data(), 8192)) {
            return DatasetReader::END;
        }
        size_t len = datafile.gcount();
        if (len == 0) {
            return DatasetReader::END;
        }
        sensors->parse_data(buffer.data(), len);
        return DatasetReader::AGAIN;
    }
}

std::pair<double, Eigen::Vector3d> SensorsDatasetReader::read_gyroscope() {
    std::pair<double, Eigen::Vector3d> gyroscope = pending_gyroscopes.front();
    pending_gyroscopes.pop_front();
    return gyroscope;
}

std::pair<double, Eigen::Vector3d> SensorsDatasetReader::read_accelerometer() {
    std::pair<double, Eigen::Vector3d> accelerometer = pending_accelerometers.front();
    pending_accelerometers.pop_front();
    return accelerometer;
}

std::pair<double, Eigen::Vector4d> SensorsDatasetReader::read_attitude() {
    std::pair<double, Eigen::Vector4d> attitude = pending_attitudes.front();
    pending_attitudes.pop_front();
    return attitude;
}

std::pair<double, Eigen::Vector3d> SensorsDatasetReader::read_gravity() {
    std::pair<double, Eigen::Vector3d> gravity = pending_gravities.front();
    pending_gravities.pop_front();
    return gravity;
}

std::pair<double, std::shared_ptr<cv::Mat>> SensorsDatasetReader::read_image() {
    std::pair<double, std::shared_ptr<cv::Mat>> image = pending_images.front();
    pending_images.pop_front();
    return image;
}
