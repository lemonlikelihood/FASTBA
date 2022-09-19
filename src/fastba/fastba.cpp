#include "fastba.h"
#include "../../dataset/dataset.h"
#include "../geometry/essential.h"
#include "../geometry/lie_algebra.h"
#include "../geometry/stereo.h"
#include "../map/feature.h"
#include "../map/frame.h"
#include "../map/map.h"
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
    log_info("[fastba]: create frame id: {}", frame->id());
    log_info("[fastba]: create frame timestamp: {}", image->t);
    frame->image = image;
    frame->K = dataset_config->camera_intrinsic();
    frame->sqrt_inv_cov = frame->K.block<2, 2>(0, 0) / ::sqrt(0.7);
    frame->camera_extri.q = dataset_config->camera_to_body_rotation();
    frame->camera_extri.p = dataset_config->camera_to_body_translation();
    frame->imu_extri.q = dataset_config->imu_to_body_rotation();
    frame->imu_extri.p = dataset_config->imu_to_body_translation();
    frame->preintegration.cov_w = dataset_config->imu_gyro_white_noise();
    frame->preintegration.cov_a = dataset_config->imu_accel_white_noise();
    frame->preintegration.cov_bg = dataset_config->imu_gyro_random_walk();
    frame->preintegration.cov_ba = dataset_config->imu_accel_random_walk();
    return frame;
}

void FASTBA::feed_image(std::shared_ptr<Image> image, DatasetConfigurator *dataset_config) {
    auto frame = create_frame(image, dataset_config);
    size_t fid = frame->id();
    get_imu(frame.get());
    if (initializer) {
        // initializer->append_frame(std::move(frame));
        track_frame(feature_tracking_map.get(), std::move(frame));
        initializer->mirror_keyframe_map(feature_tracking_map.get(), fid);
        if (sliding_window_tracker = initializer->init()) {
            f_initialized = true;
            auto [pose, motion] = sliding_window_tracker->get_latest_state();
            latest_state = {fid, pose, motion};
            initializer.reset();
            log_info("[fastba]: feed_image, init success");
            // getchar();
        } else {
            log_error("[initializer]: frame {} is initialized failedly", fid);
            return;
        }
    } else if (sliding_window_tracker) {
        track_frame(feature_tracking_map.get(), std::move(frame));
        sliding_window_tracker->mirror_frame(feature_tracking_map.get(), fid);
        if (sliding_window_tracker->track()) {
            auto [pose, motion] = sliding_window_tracker->get_latest_state();
            latest_state = {fid, pose, motion};
        } else {
            log_info("[sliding_window_tracker]: track failed reset");
            getchar();
            initializer = std::make_unique<Initializer>();
            f_initialized = false;
            sliding_window_tracker.reset();
        }
    }
}


void FASTBA::track_frame(Map *map, std::unique_ptr<Frame> frame) {
    if (map->frame_num() > 0) {
        Frame *last_frame = map->get_last_frame();
        log_info("[feature map]: last_frame_id: {}", last_frame->id());
        frame->preintegration.integrate(
            frame->image->t, last_frame->motion.bg, last_frame->motion.ba, true, false);
        log_info("[feature map]: imu intagrated");
        last_frame->track_keypoints(frame.get());
    }
    frame->detect_keypoints();
    size_t fid = frame->id();
    map->append_frame(std::move(frame));
    log_info("[feature map]: frame {} is appended to feature map successfully", fid);
    if (map->frame_num() > 1) {
        std::vector<Eigen::Vector2d> frame_i_keypoints;
        std::vector<Eigen::Vector2d> frame_j_keypoints;

        Frame *frame_i = map->get_second_to_last_frame();
        Frame *frame_j = map->get_last_frame();

        for (size_t ki = 0; ki < frame_i->keypoint_num(); ++ki) {
            Feature *feature = frame_i->get_feature(ki);
            if (!feature)
                continue;
            size_t kj = feature->get_observation_index(frame_j);
            if (kj == nil())
                continue;
            frame_i_keypoints.push_back(frame_i->get_keypoint_normalized(ki));
            frame_j_keypoints.push_back(frame_j->get_keypoint_normalized(kj));
        }

        const int32_t rows = frame_i->image->image.rows;
        const int32_t cols = frame_i->image->image.cols;
        cv::Mat img1 = frame_i->image->image;
        cv::Mat img2 = frame_j->image->image;
        cv::Mat combined(rows * 2, cols, CV_8UC1);
        img1.copyTo(combined.rowRange(0, rows));
        img2.copyTo(combined.rowRange(rows, rows * 2));
        cv::cvtColor(combined, combined, cv::COLOR_GRAY2RGBA);
        for (int i = 0; i < frame_i_keypoints.size(); i++) {
            Eigen::Vector2d pi = frame_i->apply_k(frame_i_keypoints[i]);
            cv::Point2d cv_pi = {pi.x(), pi.y()};
            // std::cout << "pi: " << pi << std::endl;
            cv::circle(combined, cv_pi, 5, cv::Scalar(255, 0, 0));
            Eigen::Vector2d pj = frame_j->apply_k(frame_j_keypoints[i]);
            cv::Point2d cv_pj = {pj.x(), pj.y()};
            // std::cout << "pj: " << pj << std::endl;
            cv::circle(combined, cv_pj + cv::Point2d(0, rows), 5, cv::Scalar(0, 255, 0));
            cv::line(combined, cv_pi, cv_pj + cv::Point2d(0, rows), cv::Scalar(0, 0, 255));
        }
        cv::imshow("continue optical flow", combined);
        cv::waitKey(1);
    }
}

void FASTBA::feed_gt_camera_pose(const Pose &pose) {
    Frame *curr_frame = feature_tracking_map->get_frame(feature_tracking_map->frame_num() - 1);
    curr_frame->gt_pose = pose;
    // compute_essential();
}

void FASTBA::compute_essential() {
    if (feature_tracking_map->frame_num() >= 2) {
        Frame *curr_frame = feature_tracking_map->get_frame(feature_tracking_map->frame_num() - 1);
        Frame *last_frame = feature_tracking_map->get_frame(feature_tracking_map->frame_num() - 2);
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
    feature_tracking_map = std::make_unique<Map>();
    initializer = std::make_unique<Initializer>();
    f_initialized = false;
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
    log_info(
        "[fastba]: frame {} is associated to imu, imu size: {}", frame->id(),
        frame->preintegration.data.size());
}

std::tuple<size_t, Pose, MotionState> FASTBA::get_lastest_state() const {
    return latest_state;
}
