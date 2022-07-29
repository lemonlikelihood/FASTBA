#include "../dataset/euroc_dataset_reader.h"
#include "../src/fastba/fastba.h"
#include "../src/utils/debug.h"

int main() {
    std::string euroc_path = "/Users/lemon/dataset/MH_01";
    auto dataset_reader = std::make_unique<EurocDatasetReader>(euroc_path);

    double t;
    Eigen::Vector3d w;
    Eigen::Vector3d a;
    Eigen::Vector4d atti;
    Eigen::Vector3d gravity;
    std::shared_ptr<cv::Mat> image;

    FASTBA fastba;
    while (true) {
        NextDataType next_data_type;
        while ((next_data_type = dataset_reader->next()) == NextDataType::AGAIN) {}
        switch (next_data_type) {
            case NextDataType::AGAIN: { // impossible but we put it here
                log_error("dataset again shouldn't appear!");
            } break;
            case NextDataType::IMU: {
                // log_debug("get imu");
                IMUData imu = dataset_reader->read_imu();
                log_info("U: {} {} {}", imu.t, imu.w.transpose(), imu.a.transpose());
            } break;
            case NextDataType::IMAGE: {
                // log_debug("get image");
                auto image = dataset_reader->read_image();
                fastba.feed_image(image, dataset_reader->dataset_config.get());
                log_info("I: {}", image->t);
                cv::imshow("image", image->image);
                cv::waitKey(1);
            } break;
            case NextDataType::END: {
                log_info("get end");
                // exit(EXIT_SUCCESS);
                return 1;
            } break;
            default: {
                // exit(EXIT_SUCCESS);
            } break;
        }
    }
    return 0;
}