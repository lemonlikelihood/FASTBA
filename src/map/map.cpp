#include "feature.h"
#include "frame.h"

#include "map.h"

#include "../optimizer/bundle_adjustor.h"

Map::Map() = default;
Map::~Map() = default;

void Map::clear() {
    frames.clear();
    features.clear();
}

void Map::append_frame(std::unique_ptr<Frame> frame, size_t pos) {
    frame->map = this;
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
}

void Map::erase_frame(size_t index) {
    Frame *frame = frames[index].get();
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        if (Feature *feature = frame->get_feature(i); feature != nullptr) {
            feature->remove_observation(frame);
        }
    }
    frames.erase(frames.begin() + index);
    if (index > 0 && index < frames.size()) {
        Frame *frame_i = frames[index - 1].get();
        Frame *frame_j = frames[index].get();
        frame_j->preintegration_factor =
            Factor::create_preintegration_error(frame_i, frame_j); // imu 数据没有融合
    }
}

void Map::marginalize_frame(size_t index) {
    log_info("[map]: marginalize_frame {} begin ...", index);
    BundleAdjustor().marginalize_frame(this, index);
    Frame *frame = frames[index].get();
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        if (Feature *feature = frame->get_feature(i); feature != nullptr) {
            feature->remove_observation(frame);
        }
    }
    frames.erase(frames.begin() + index);
    if (index > 0 && index < frames.size()) {
        frames[index]->preintegration_factor.reset(); // imu 数据没有融合
    }
}

void Map::set_marginalization_factor(std::unique_ptr<Factor> factor) {
    marginalization_factor = std::move(factor);
}

size_t Map::get_frame_index_by_id(size_t id) const {
    struct FrameID {
        FrameID(const std::unique_ptr<Frame> &frame) : id(frame->id()) {}
        FrameID(size_t id) : id(id) {}
        bool operator<(const FrameID &fi) const { return id < fi.id; }
        size_t id;
    };

    auto it = std::lower_bound(frames.begin(), frames.end(), id, std::less<FrameID>());
    if (it == frames.end())
        return nil();
    if (id < (*it)->id())
        return nil();
    return std::distance(frames.begin(), it);
}

void Map::update_feature_state() {
    Frame *first_frame = get_first_frame();
    Frame *last_frame = get_last_frame();
    for (size_t i = 0; i < feature_num(); ++i) {
        Feature *feature = get_feature(i);
        if (!feature->flag(FeatureFlag::FF_VALID) || !feature->flag(FeatureFlag::FF_TRIANGULATED))
            continue;
        if (feature->has_observation(first_frame) && feature->has_observation(last_frame)) {
            feature->flag(FeatureFlag::FF_STABLE) = true;
            feature->flag(FeatureFlag::FF_GOOD) = false;
            feature->flag(FeatureFlag::FF_DEAD) = false;
            feature->flag(FeatureFlag::FF_LOST) = false;
        } else if (feature->has_observation(first_frame) && !feature->has_observation(last_frame)) {
            feature->flag(FeatureFlag::FF_STABLE) = false;
            feature->flag(FeatureFlag::FF_GOOD) = false;
            feature->flag(FeatureFlag::FF_DEAD) = true;
            feature->flag(FeatureFlag::FF_LOST) = false;
        } else if (!feature->has_observation(first_frame) && feature->has_observation(last_frame)) {
            feature->flag(FeatureFlag::FF_STABLE) = false;
            feature->flag(FeatureFlag::FF_GOOD) = true;
            feature->flag(FeatureFlag::FF_DEAD) = false;
            feature->flag(FeatureFlag::FF_LOST) = false;
        } else if (
            !feature->has_observation(first_frame) && !feature->has_observation(last_frame)) {
            feature->flag(FeatureFlag::FF_STABLE) = false;
            feature->flag(FeatureFlag::FF_GOOD) = false;
            feature->flag(FeatureFlag::FF_DEAD) = false;
            feature->flag(FeatureFlag::FF_LOST) = true;
        }
        // if (feature->flag(FeatureFlag::FF_LOST) && feature->has_observation(first_frame)) {
        //     feature->flag(FeatureFlag::FF_LOST) = false;
        //     feature->flag(FeatureFlag::FF_DEAD) = true;
        // } else if (feature->flag(FeatureFlag::FF_GOOD) && feature->has_observation(first_frame)) {
        //     feature->flag(FeatureFlag::FF_GOOD) = false;
        //     feature->flag(FeatureFlag::FF_STABLE) = true;
        // } else if (feature->flag(FeatureFlag::FF_STABLE) && !feature->has_observation(last_frame)) {
        //     feature->flag(FeatureFlag::FF_STABLE) = false;
        //     feature->flag(FeatureFlag::FF_DEAD) = true;
        // } else if (feature->flag(FeatureFlag::FF_STABLE) && feature->has_observation(last_frame)) {
        //     feature->flag(FeatureFlag::FF_STABLE) = true;
        // } else if (feature->flag(FeatureFlag::FF_GOOD) && !feature->has_observation(last_frame)) {
        //     feature->flag(FeatureFlag::FF_GOOD) = false;
        //     feature->flag(FeatureFlag::FF_LOST) = true;
        // } else if (feature->flag(FeatureFlag::FF_GOOD) && feature->has_observation(last_frame)) {
        //     feature->flag(FeatureFlag::FF_GOOD) = true;
        // } else if (
        //     !feature->flag(FeatureFlag::FF_STABLE) && !feature->flag(FeatureFlag::FF_GOOD)
        //     && !feature->flag(FeatureFlag::FF_DEAD) && !feature->flag(FeatureFlag::FF_LOST)
        //     && feature->flag(FeatureFlag::FF_TRIANGULATED)) {
        //     feature->flag(FeatureFlag::FF_GOOD) = true;
        // }
    }
}

Feature *Map::create_feature() {
    std::unique_ptr<Feature> feature = std::make_unique<Feature>();
    feature->index_in_map = features.size();
    feature->map = this;
    feature_id_map[feature->id()] = feature.get();
    features.emplace_back(std::move(feature));
    // log_debug("Sliding window create success");
    return features.back().get();
}

void Map::erase_feature(Feature *feature) {
    while (feature->observation_num() > 0) {
        feature->remove_observation(feature->observation_map().begin()->first, false);
    }
    recycle_feature(feature);
}

void Map::prune_features(const std::function<bool(const Feature *)> &condition) {
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

Feature *Map::get_feature_by_id(size_t id) const {
    if (feature_id_map.count(id)) {
        return feature_id_map.at(id);
    } else {
        return nullptr;
    }
}

void Map::recycle_feature(Feature *feature) {
    if (feature->index_in_map != features.back()->index_in_map) {
        features[feature->index_in_map].swap(features.back());
        features[feature->index_in_map]->index_in_map = feature->index_in_map;
    }

    feature_id_map.erase(feature->id());
    features.pop_back();
}

std::pair<double, double> Map::compute_reprojections() {
    double reprojection_error = 0;
    int reprojection_num = 0;
    int good_num = 0;
    int stable_num = 0;
    int dead_num = 0;
    int lost_num = 0;

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
        reprojection_error += feature->reprojection_error;
        reprojection_num++;
        if (feature->flag(FeatureFlag::FF_STABLE)) {
            stable_num++;
        } else if (feature->flag(FeatureFlag::FF_GOOD)) {
            good_num++;
        } else if (feature->flag(FeatureFlag::FF_DEAD)) {
            dead_num++;
        } else if (feature->flag(FeatureFlag::FF_LOST)) {
            lost_num++;
        }
        printf(
            "[feature]: %d, %zu, %s: ", reprojection_num, feature->id(), feature->info().c_str());
        for (const auto &k : feature->observation_map()) {
            Frame *frame = k.first;
            printf(" %zu, ", frame->id());
        }
        printf("\n");
    }
    reprojection_error = reprojection_error / std::max(reprojection_num, 1);
    printf("stable_num: %d\n", stable_num);
    printf("good_num: %d\n", good_num);
    printf("dead_num: %d\n", dead_num);
    printf("lost_num: %d\n", lost_num);
    return {reprojection_num, reprojection_error};
}

std::pair<double, double> Map::compute_reprojections_without_last_frame() {
    double reprojection_error = 0;
    int reprojection_num = 0;
    Frame *last_frame = get_last_frame();
    for (size_t i = 0; i < feature_num(); ++i) {
        Feature *feature = get_feature(i);
        if (!feature->flag(FeatureFlag::FF_VALID))
            continue;
        const Eigen::Vector3d &x = feature->p_in_G;
        double quality = 0.0;
        double quality_num = 0.0;
        for (const auto &k : feature->observation_map()) {
            Frame *frame = k.first;
            if (frame == last_frame)
                continue;
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
        reprojection_error += feature->reprojection_error;
        reprojection_num++;
    }
    reprojection_error = reprojection_error / std::max(reprojection_num, 1);
    return {reprojection_num, reprojection_error};
}

void Map::log_feature_reprojections() {
    int feature_cnt = 0;
    double sum_error = 0;
    log_info("feature_num: {}", feature_num());
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
