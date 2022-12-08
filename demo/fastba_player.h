#pragma once
#include "../dataset/euroc_dataset_reader.h"
#include "../src/fastba/fastba.h"
#include "../src/geometry/lie_algebra.h"
#include "../src/utils/debug.h"
#include "../src/utils/read_file.h"

#include <lyra/lyra.hpp>

#include "config.h"

class FastBAPlayer {
public:
    std::unique_ptr<EurocDatasetReader> dataset_reader;
    std::shared_ptr<Config> config;
    std::unique_ptr<FASTBA> fastba_instance;

    size_t fid;
    std::shared_ptr<Image> image;
    IMUData imu;
    Pose pose;
    MotionState motion;
    TrackingState tracking_state;
    Eigen::Matrix3d K;

    FastBAPlayer(const std::string &euroc_data_path, const std::string &config_file_path) {
        dataset_reader = std::make_unique<EurocDatasetReader>(euroc_data_path);
        config = std::make_unique<Config>(config_file_path);
        fastba_instance = std::make_unique<FASTBA>(config);
        K = dataset_reader->dataset_config->camera_intrinsic();
    }

    bool step() {
        static bool has_imu = false;
        while (true) {
            NextDataType next_data_type;
            while ((next_data_type = dataset_reader->next()) == NextDataType::AGAIN) {}
            switch (next_data_type) {
                case NextDataType::AGAIN: { // impossible but we put it here
                    log_error("dataset again shouldn't appear!");
                } break;
                case NextDataType::IMU: {
                    // log_debug("get imu");
                    imu = dataset_reader->read_imu();
                    fastba_instance->feed_imu(imu);
                    has_imu = true;
                    // log_info("U: {} {} {}", imu.t, imu.w.transpose(), imu.a.transpose());
                } break;
                case NextDataType::IMAGE: {
                    // log_debug("get image");
                    image = dataset_reader->read_image();
                    if (has_imu) {
                        fastba_instance->feed_image(image, dataset_reader->dataset_config.get());
                        std::tie(fid, tracking_state, pose, motion) =
                            fastba_instance->get_lastest_state();
                    }
                    // Pose imu_pose = dataset_reader->get_groundtruth_pose(image->t);
                    // log_info("image t: {}", image->t);
                    // log_info("imu t: {}", imu_pose.t);
                    // log_info("imu R: {}", imu_pose.q.coeffs().transpose());
                    // log_info("imu t: {}", imu_pose.p.transpose());
                    // Pose cam_pose;
                    // cam_pose.q =
                    //     imu_pose.q * dataset_reader->dataset_config->camera_to_body_rotation();
                    // cam_pose.p =
                    //     imu_pose.p
                    //     + imu_pose.q * dataset_reader->dataset_config->camera_to_body_translation();

                    // Eigen::Matrix3d gt_essential = hat(cam_pose.p) * cam_pose.q.toRotationMatrix();
                    // log_info("gt essential: \n{}", gt_essential);
                    // log_info("gt R: {}", cam_pose.q.coeffs().transpose());
                    // log_info("gt t: {}", cam_pose.p.transpose());

                    // output_writer.write_pose(image->t, pose);
                    // fastba.feed_gt_camera_pose(cam_pose);
                    // log_info("I: {}", image->t);
                    cv::imshow("image", image->image);
                    cv::waitKey(1);
                    return true;
                } break;
                case NextDataType::END: {
                    log_info("get end");
                    // exit(EXIT_SUCCESS);
                    return false;
                } break;
                default: {
                    // exit(EXIT_SUCCESS);
                } break;
            }
        }
        return false;
    }
};