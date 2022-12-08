#include "frame.h"
#include "../../demo/config.h"
#include "dataset/keypoint_filter.h"
#include "../optimizer/preintegrator.h"
#include "feature.h"
#include "map.h"

Frame::~Frame() = default;

Frame::Frame() = default;

Frame::Frame(size_t id) : Identifiable(id), map(nullptr) {}

Eigen::Vector2d Frame::remove_k(const Eigen::Vector2d &p) {
    return {(p(0) - K(0, 2)) / K(0, 0), (p(1) - K(1, 2)) / K(1, 1)};
}

Eigen::Vector2d Frame::apply_k(const Eigen::Vector2d &p) {
    return {p(0) * K(0, 0) + K(0, 2), p(1) * K(1, 1) + K(1, 2)};
}

std::unique_ptr<Frame> Frame::clone() const {
    std::unique_ptr<Frame> frame = std::make_unique<Frame>(id());
    frame->K = K;
    frame->sqrt_inv_cov = sqrt_inv_cov;
    frame->image = image;
    frame->pose = pose;
    frame->motion = motion;
    frame->camera_extri = camera_extri;
    frame->imu_extri = imu_extri;
    frame->keypoints = keypoints;
    frame->keypoints_normalized = keypoints_normalized;
    frame->features = std::vector<Feature *>(keypoints.size(), nullptr);
    frame->reprojection_factors = std::vector<std::unique_ptr<Factor>>(keypoints.size());
    frame->preintegration = preintegration;
    frame->map = nullptr;
    return frame;
}

Pose Frame::get_camera_pose() const {
    Pose ret;
    ret.q = this->pose.q * camera_extri.q;
    ret.p = this->pose.p + this->pose.q * camera_extri.p;
    return ret;
}

Pose Frame::get_imu_pose() const {
    Pose ret;
    ret.q = this->pose.q * imu_extri.q;
    ret.p = this->pose.p + this->pose.q * imu_extri.p;
    return ret;
}

Pose Frame::get_body_pose() const {
    return pose;
}

void Frame::set_camera_pose(const Pose &camera_pose) {
    this->pose.q = camera_pose.q * camera_extri.q.conjugate();
    this->pose.p = camera_pose.p - this->pose.q * camera_extri.p;
}

void Frame::set_imu_pose(const Pose &imu_pose) {
    this->pose.q = imu_pose.q * imu_extri.q.conjugate();
    this->pose.p = imu_pose.p - this->pose.q * imu_extri.p;
}

void Frame::append_keypoint(const Eigen::Vector2d &keypoint) {
    keypoints.emplace_back(keypoint);
    keypoints_normalized.emplace_back(remove_k(keypoint));
    reprojection_factors.emplace_back(nullptr);
}

Feature *Frame::get_feature_if_empty_create(size_t keypoint_id) {
    if (features[keypoint_id] == nullptr) {
        Feature *feature = map->create_feature();
        feature->add_observation(this, keypoint_id);
    }
    return features[keypoint_id];
}

void Frame::detect_keypoints(Config *config) {
    std::vector<Eigen::Vector2d> pkeypoints(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        pkeypoints[i] = keypoints[i];
    }

    image->detect_keypoints(
        pkeypoints, config->get_feature_tracker_max_keypoint_detection(),
        config->get_feature_tracker_min_keypoint_distance());
    size_t old_keypoint_num = keypoints.size();
    keypoints.resize(pkeypoints.size());
    keypoints_normalized.resize(pkeypoints.size());
    features.resize(pkeypoints.size(), nullptr);
    reprojection_factors.resize(pkeypoints.size());
    for (size_t i = old_keypoint_num; i < pkeypoints.size(); ++i) {
        keypoints[i] = pkeypoints[i];
        keypoints_normalized[i] = remove_k(pkeypoints[i]);
    }
    log_info(
        "[feature map]: frame detect_keypoints, tracked keypoints: {}, now all keypoints: {}",
        old_keypoint_num, keypoints.size());
}


void Frame::track_keypoints(Config *config, Frame *next_frame) {
    std::vector<Eigen::Vector2d> curr_keypoints(keypoints.size());
    std::vector<Eigen::Vector2d> next_keypoints;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        curr_keypoints[i] = keypoints[i];
    }

    if (config->is_feature_tracker_predict_keypoints()) { //
        Eigen::Quaternion delta_key_q =
            (camera_extri.q.conjugate() * imu_extri.q * next_frame->preintegration.delta.q
             * next_frame->imu_extri.q.conjugate() * next_frame->camera_extri.q)
                .conjugate();
        next_keypoints.resize(curr_keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            next_keypoints[i] =
                apply_k((delta_key_q * keypoints_normalized[i].homogeneous()).hnormalized());
        }
        log_debug(
            "[feature map]: frame track_keypoints, predicted imu dq: {}",
            next_frame->preintegration.delta.q.coeffs().transpose());
        log_debug(
            "[feature map]: frame track_keypoints, predicted image dq: {}",
            delta_key_q.coeffs().transpose());
    }
    std::vector<char> status;
    image->track_keypoints(next_frame->image.get(), curr_keypoints, next_keypoints, status);

    // filter keypoints based on track length
    std::vector<std::pair<size_t, size_t>> keypoint_index_track_length;
    keypoint_index_track_length.reserve(curr_keypoints.size());

    for (size_t i = 0; i < curr_keypoints.size(); ++i) {
        if (status[i] == 0)
            continue;
        Feature *feature = get_feature(i);
        if (feature == nullptr)
            continue;
        keypoint_index_track_length.emplace_back(i, feature->observation_num());
    }

    std::sort(
        keypoint_index_track_length.begin(), keypoint_index_track_length.end(),
        [](const auto &a, const auto &b) { return a.second > b.second; });

    PoissonKeypointFilter<2> filter(config->get_feature_tracker_min_keypoint_distance());
    for (auto &[keypoint_index, track_length] : keypoint_index_track_length) {
        Eigen::Vector2d pt = next_keypoints[keypoint_index];
        if (filter.permit_point(pt)) {
            filter.preset_point(pt);
        } else {
            status[keypoint_index] = 0;
        }
    }

    int keypoint_tracked_num = 0;
    for (size_t curr_keypoint_index = 0; curr_keypoint_index < curr_keypoints.size();
         ++curr_keypoint_index) {
        if (status[curr_keypoint_index]) {
            size_t next_keypoint_index = next_frame->keypoints.size();
            next_frame->keypoints.emplace_back((next_keypoints[curr_keypoint_index]));
            next_frame->keypoints_normalized.emplace_back(
                remove_k(next_keypoints[curr_keypoint_index]));
            next_frame->features.emplace_back(nullptr);
            next_frame->reprojection_factors.emplace_back(nullptr);
            get_feature_if_empty_create(curr_keypoint_index)
                ->add_observation(next_frame, next_keypoint_index);
            keypoint_tracked_num++;
        }
    }
    log_info(
        "[feature map]: frame track_keypoints, last frame keypoints: {}, success tracked: {}",
        keypoints.size(), keypoint_tracked_num);
}

std::unique_lock<std::mutex> Frame::lock() const {
    if (map) {
        return map->lock();
    } else {
        return {};
    }
}