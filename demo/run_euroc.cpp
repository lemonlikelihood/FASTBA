#include "../dataset/euroc_dataset_reader.h"

int main() {
    std::string euroc_path = "/Users/lemon/dataset/MH_01/mav0";
    std::unique_ptr<DatasetReader> dataset_reader =
        DatasetReader::create_reader("euroc", euroc_path);

    double t;
    Eigen::Vector3d w;
    Eigen::Vector3d a;
    Eigen::Vector4d atti;
    Eigen::Vector3d gravity;
    std::shared_ptr<cv::Mat> image;
    while (true) {
        DatasetReader::NextDataType next_data_type;
        while ((next_data_type = dataset_reader->next()) == DatasetReader::AGAIN) {}
        switch (next_data_type) {
            case DatasetReader::AGAIN: { // impossible but we put it here
            } break;
            case DatasetReader::GYROSCOPE: {
                std::tie(t, w) = dataset_reader->read_gyroscope();
                std::cout << "w: " << t << " " << w.transpose() << std::endl;
            } break;
            case DatasetReader::ACCELEROMETER: {
                std::tie(t, a) = dataset_reader->read_accelerometer();
                std::cout << "w: " << t << " " << w.transpose() << std::endl;
            } break;
            case DatasetReader::IMAGE: {
                std::tie(t, image) = dataset_reader->read_image();
                std::cout << "i: " << t << std::endl;
                cv::imshow("image", *image);
                cv::waitKey(0);
            } break;
            case DatasetReader::END: {
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