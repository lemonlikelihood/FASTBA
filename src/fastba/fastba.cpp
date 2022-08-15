#include "fastba.h"
#include "../../dataset/dataset.h"
#include "../geometry/essential.h"
#include "../geometry/lie_algebra.h"
#include "../geometry/stereo.h"
#include "../map/feature.h"
#include "../map/frame.h"
#include "../map/sliding_window.h"
#include "../utils/debug.h"

bool FASTBA::feed_monocular(Frame *frame) {
    // if (!m_initialized) {
    //     if (m_initializer->m_raw_sliding_window->frame_num() > 0) {
    //         Frame *last_frame = m_initializer->m_raw_sliding_window->get_last_frame();
    //         frame->m_preintegration.integrate(
    //             frame->m_image->t, last_frame->m_motion.bg, last_frame->m_motion.ba, true, false);
    //         log_info("integrate success");
    //     }
    //     if ((m_sliding_window = m_initializer->init_sfm())) {
    //         m_initialized = true;
    //         for (size_t frame_id = 0; frame_id < m_sliding_window->frame_num(); ++frame_id) {
    //             Frame *frame = m_sliding_window->get_frame(frame_id);
    //             log_info("frame id: {}", frame->id());
    //             log_info("frame t: {}", frame->m_image->t);
    //             log_info("frame p: {}", frame->get_body_pose().p.transpose());
    //             log_info("frame q: {}", frame->get_body_pose().q.coeffs().transpose());
    //         }
    //         log_info("[feed_monocular]: init success");
    //         getchar();
    //     } else {
    //         log_info("[feed_monocular]: init failed");
    //         return false;
    //     }
    // }
    return true;
}

std::unique_ptr<Frame>
FASTBA::create_frame(std::shared_ptr<Image> image, DatasetConfigurator *dataset_config) {
    auto frame = std::make_unique<Frame>();
    // frame->m_K = config->camera_intrinsic;
    log_debug("create frame id: {}", frame->id());
    frame->image = image;
    frame->K = dataset_config->camera_intrinsic();
    frame->sqrt_inv_cov = frame->K.block<2, 2>(0, 0) / ::sqrt(0.7);
    frame->camera_extri.q = dataset_config->camera_to_body_rotation();
    frame->camera_extri.p = dataset_config->camera_to_body_translation();
    frame->imu_extri.q = dataset_config->camera_to_body_rotation();
    frame->imu_extri.p = dataset_config->camera_to_body_translation();
    frame->preintegration.cov_w = dataset_config->imu_gyro_white_noise();
    frame->preintegration.cov_a = dataset_config->imu_accel_white_noise();
    frame->preintegration.cov_bg = dataset_config->imu_gyro_random_walk();
    frame->preintegration.cov_ba = dataset_config->imu_accel_random_walk();
    return frame;
}

void FASTBA::feed_image(std::shared_ptr<Image> image, DatasetConfigurator *dataset_config) {
    auto frame = create_frame(image, dataset_config);
    get_imu(frame.get());
    if (!f_initialized) {
        initializer->append_frame(std::move(frame));
        if (initializer->init()) {
            f_initialized = true;
            for (size_t frame_id = 0; frame_id < sw->frame_num(); ++frame_id) {
                double reprojection_error = 0;
                int reprojection_num = 0;
                Frame *frame = sw->get_frame(frame_id);

                log_info("frame id: {}", frame->id());
                log_info("frame t: {}", frame->image->t);
                log_info("frame p: {}", frame->get_body_pose().p.transpose());
                log_info("frame q: {}", frame->get_body_pose().q.coeffs().transpose());

                for (size_t i = 0; i < frame->keypoint_num(); ++i) {
                    Feature *feature = frame->get_feature(i);
                    if (!feature)
                        continue;
                    if (feature->flag(FeatureFlag::FF_VALID)) {
                        Pose pose = frame->get_camera_pose();
                        Eigen::Vector2d r =
                            frame->apply_k(
                                (pose.q.conjugate() * (feature->p_in_G - pose.p)).hnormalized())
                            - frame->get_keypoint(i);
                        Eigen::Vector3d y_body =
                            frame->pose.q.conjugate() * (feature->p_in_G - frame->pose.p);
                        Eigen::Vector3d y_cam =
                            frame->camera_extri.q.conjugate() * (y_body - frame->camera_extri.p);

                        Eigen::Vector2d r1 =
                            frame->apply_k(y_cam.hnormalized()) - frame->get_keypoint(i);

                        log_info("r1: {}", r1.transpose());
                        log_info("r: {}", r.transpose());
                        reprojection_error += r.norm();
                        reprojection_num++;
                    }
                }
                log_info("reprojection error: {}", reprojection_error / reprojection_num);
            }
            log_info("[feed_monocular]: init success");
            getchar();
        } else {
            log_info("[feed_monocular]: init failed");
            return;
        }
    }
}

void FASTBA::feed_gt_camera_pose(const Pose &pose) {
    Frame *curr_frame = sw->get_frame(sw->frame_num() - 1);
    curr_frame->gt_pose = pose;
    // compute_essential();
}

void FASTBA::compute_essential() {
    if (sw->frame_num() >= 2) {
        Frame *curr_frame = sw->get_frame(sw->frame_num() - 1);
        Frame *last_frame = sw->get_frame(sw->frame_num() - 2);
        log_info("curr_frame->m_keypoints_normalized.size()", curr_frame->keypoint_num());
        std::vector<Eigen::Vector2d> last_keypoint_vec;
        std::vector<Eigen::Vector2d> curr_keypoint_vec;
        for (int i = 0; i < curr_frame->keypoint_num(); ++i) {
            if (curr_frame->get_feature(i) != nullptr) {
                Feature *feature = curr_frame->get_feature(i);
                Eigen::Vector2d last_keypoint_normalized =
                    feature->get_observation_normalized(last_frame);
                Eigen::Vector2d curr_keypoint_normalized = curr_frame->get_keypoint_normalized(i);
                last_keypoint_vec.push_back(last_keypoint_normalized);
                curr_keypoint_vec.push_back(curr_keypoint_normalized);
            }
        }

        std::vector<Eigen::Vector3d> result_points;
        std::vector<char> result_status;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        Eigen::Matrix3d E = find_essential_matrix(last_keypoint_vec, curr_keypoint_vec);
        triangulate_from_essential(
            last_keypoint_vec, curr_keypoint_vec, E, result_points, result_status, R, T);
        log_info("Essential Matrix:\n {}", E);
        log_info("Essential R:\n {}", R);
        log_info("Essential T:\n {}", T);

        Pose last_gt = last_frame->gt_pose;
        Pose curr_gt = curr_frame->gt_pose;
        Eigen::Matrix3d relative_R = (curr_gt.q.conjugate() * last_gt.q).toRotationMatrix();
        Eigen::Vector3d relative_t = last_gt.q.conjugate() * (curr_gt.p - last_gt.p);
        Eigen::Matrix3d gt_essential = hat(relative_t) * relative_R;
        // log_info("gt last R: \n{}", last_gt.q.coeffs().transpose());
        // log_info("gt last t: \n{}", last_gt.p.transpose());
        // log_info("gt curr R: \n{}", curr_gt.q.coeffs().transpose());
        // log_info("gt curr t: \n{}", curr_gt.p.transpose());
        // log_info("gt R: \n{}", relative_R);
        // log_info("gt t: \n{}", relative_t.normalized());
        // log_info("gt essential: \n{}", gt_essential);
        // log_info("curr frame id: {}", curr_frame->id());
        // log_info("last frame id: {}", last_frame->id());
    }
}

FASTBA::FASTBA() {
    // tracker = std::make_unique<KLTTracker>();
    sw = std::make_unique<SlidingWindow>();
    initializer = std::make_unique<Initializer>();
    // m_initializer->m_raw_sliding_window = m_sliding_window;
}

FASTBA::~FASTBA() {}

void FASTBA::feed_imu(const IMUData &imu) {
    imu_buff.push_back(imu);
}

void FASTBA::get_imu(Frame *frame) {
    while (imu_buff.size() > 0) {
        if (imu_buff.front().t <= frame->image->t) {
            frame->preintegration.data.push_back(imu_buff.front());
            imu_buff.pop_front();
        }
    }
}
