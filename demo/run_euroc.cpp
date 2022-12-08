#include "../dataset/euroc_dataset_reader.h"
#include "../src/fastba/fastba.h"
#include "../src/geometry/lie_algebra.h"
#include "../src/utils/debug.h"
#include "../src/utils/read_file.h"

#include <lyra/lyra.hpp>

int main(int argc, const char *argv[]) {
    bool show_help = false;

    std::string euroc_data_path;
    std::string config_file_path;

    auto cli =
        lyra::cli() | lyra::help(show_help).description("Run FastBA")
        | lyra::opt(euroc_data_path, "dataset path")["-d"]["--dataset-path"]("Euroc dataset path")
              .required()
        | lyra::opt(config_file_path, "config path")["-c"]["--config-path"]("Config file path");


    auto cli_result = cli.parse({argc, argv});
    if (!cli_result) {
        fmt::print(stderr, "{}\n\n{}\n", cli_result.message(), cli);
        return -1;
    }

    // std::string euroc_path = "/Users/lemon/dataset/MH_05";
    auto dataset_reader = std::make_unique<EurocDatasetReader>(euroc_data_path);

    TumTrajectoryWriter output_writer("trajectory.txt");

    double t;
    Eigen::Vector3d w;
    Eigen::Vector3d a;
    Eigen::Vector4d atti;
    Eigen::Vector3d gravity;
    // std::shared_ptr<cv::Mat> image;

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
                fastba.feed_imu(imu);
                // log_info("U: {} {} {}", imu.t, imu.w.transpose(), imu.a.transpose());
            } break;
            case NextDataType::IMAGE: {
                // log_debug("get image");
                auto image = dataset_reader->read_image();
                Pose imu_pose = dataset_reader->get_groundtruth_pose(image->t);
                // log_info("image t: {}", image->t);
                // log_info("imu t: {}", imu_pose.t);
                // log_info("imu R: {}", imu_pose.q.coeffs().transpose());
                // log_info("imu t: {}", imu_pose.p.transpose());
                Pose cam_pose;
                cam_pose.q = imu_pose.q * dataset_reader->dataset_config->camera_to_body_rotation();
                cam_pose.p =
                    imu_pose.p
                    + imu_pose.q * dataset_reader->dataset_config->camera_to_body_translation();

                // Eigen::Matrix3d gt_essential = hat(cam_pose.p) * cam_pose.q.toRotationMatrix();
                // log_info("gt essential: \n{}", gt_essential);
                // log_info("gt R: {}", cam_pose.q.coeffs().transpose());
                // log_info("gt t: {}", cam_pose.p.transpose());
                fastba.feed_image(image, dataset_reader->dataset_config.get());
                auto [fid, tracking_state, pose, motion] = fastba.get_lastest_state();
                output_writer.write_pose(image->t, pose);
                // fastba.feed_gt_camera_pose(cam_pose);
                // log_info("I: {}", image->t);
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