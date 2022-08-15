#include "feature.h"
#include "frame.h"

#include "map.h"
#include "sliding_window.h"

SlidingWindow::SlidingWindow() = default;
SlidingWindow::~SlidingWindow() = default;


void SlidingWindow::clear() {
    frames.clear();
    features.clear();
}

void SlidingWindow::append_frame(std::unique_ptr<Frame> frame, size_t pos) {
    frame->sw = this;
    if (pos == nil()) {
        frames.emplace_back(std::move(frame));
        pos = frames.size() - 1;
    } else {
        frames.emplace(frames.begin() + pos, std::move(frame));
    }
    if (pos > 0) {
        Frame *frame_i = frames[pos - 1].get();
        Frame *frame_j = frames[pos].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
        // integration
    }
    if (pos < frames.size() - 1) {
        Frame *frame_i = frames[pos].get();
        Frame *frame_j = frames[pos + 1].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
    log_debug("put frame success");
}

Feature *SlidingWindow::create_feature() {
    std::unique_ptr<Feature> feature = std::make_unique<Feature>();
    feature->index_in_sw = features.size();
    feature->sw = this;
    features.emplace_back(std::move(feature));
    // log_debug("Sliding window create success");
    return features.back().get();
}

void SlidingWindow::erase_feature(Feature *feature) {
    while (feature->observation_map().size() > 0) {
        feature->remove_observation(feature->observation_map().begin()->first, false);
    }
    recycle_feature(feature);
}

void SlidingWindow::prune_features(const std::function<bool(const Feature *)> &condition) {
    std::vector<Feature *> features_to_prune;
    for (size_t i = 0; i < feature_num(); ++i) {
        Feature *feature = get_feature(i);
        if (condition(feature)) {
            features_to_prune.push_back(feature);
        }
    }

    for (Feature *feature : features_to_prune) {
        erase_feature(feature);
    }
}

void SlidingWindow::recycle_feature(Feature *feature) {
    if (feature->index_in_sw != features.back()->index_in_sw) {
        features[feature->index_in_sw].swap(features.back());
        features[feature->index_in_sw]->index_in_sw = feature->index_in_sw;
    }
    features.pop_back();
}

void SlidingWindow::compute_reprojections() {
    for (size_t i = 0; i < feature_num(); ++i) {
        Feature *feature = get_feature(i);
        if (!feature->flag(FeatureFlag::FF_VALID))
            continue;
        const Eigen::Vector3d &x = feature->p_in_G;
        double quality = 0.0;
        double quality_num = 0.0;
        for (const auto &k : feature->observation_map()) {
            Frame *frame = k.first;
            size_t keypoint_id = k.second;
            Pose pose = frame->get_camera_pose();
            Eigen::Vector3d y = pose.q.conjugate() * (x - pose.p);
            if (y.z() <= 1.0e-3 || y.z() > 50) {
                feature->flag(FeatureFlag::FF_VALID) = false;
                break;
            }
            quality += (frame->apply_k(y.hnormalized()) - frame->get_keypoint(keypoint_id)).norm();
            quality_num += 1.0;
        }
        if (!feature->flag(FeatureFlag::FF_VALID))
            continue;
        feature->reprojection_error = quality / std::max(quality_num, 1.0);
    }
}

void SlidingWindow::log_feature_reprojections() {
    int feature_cnt = 0;
    double sum_error = 0;
    for (int i = 0; i < feature_num(); ++i) {
        Feature *feature = get_feature(i);
        if (!feature->flag(FeatureFlag::FF_VALID))
            continue;
        feature_cnt++;
        sum_error += feature->reprojection_error;
        log_info(
            "[feature]: i: {},reprojection_error: {}", feature_cnt, feature->reprojection_error);
    }
    log_info("sum error: {}", sum_error);
}
