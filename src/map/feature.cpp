#include "feature.h"
#include "../geometry/stereo.h"
#include "frame.h"

Feature::Feature() = default;
Feature::~Feature() = default;

void Feature::add_observation(Frame *frame, size_t keypoint_index) {
    frame->features[keypoint_index] = this;
    frame->reprojection_factors[keypoint_index] =
        Factor::create_reprojection_error(frame, keypoint_index);
    observation_refs[frame] = keypoint_index;
}

void Feature::remove_observation(Frame *frame, bool suicide_if_empty) {
    size_t keypoint_index = observation_refs.at(frame);
    // if (observation_refs.size() > 1) {
    //     if (frame == first_frame()) {
    //         // todo
    //     }
    // } else {
    //     flag(FeatureFlag::FF_VALID) = false;
    // }
    if (observation_refs.size() == 1) {
        flag(FeatureFlag::FF_VALID) = false;
    }

    frame->features[keypoint_index] = nullptr;
    frame->reprojection_factors[keypoint_index].reset();
    observation_refs.erase(frame);
    if (suicide_if_empty && observation_refs.size() == 0) {
        map->recycle_feature(this);
    }
}

bool Feature::triangulate() {
    std::vector<Eigen::Matrix<double, 3, 4>> Ps;
    std::vector<Eigen::Vector2d> ps;
    for (const auto &it : observation_map()) {
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        Pose pose = it.first->get_camera_pose();
        R = pose.q.conjugate().toRotationMatrix();
        t = -(R * pose.p);
        P << R, t;
        Ps.push_back(P);
        ps.push_back(it.first->get_keypoint_normalized(it.second));
    }

    Eigen::Vector3d p;
    double score = 0;
    if (triangulate_point_scored(Ps, ps, p, score)) {
        p_in_G = p;
        // log_info("index_in_map: {}, score: {}", index_in_map, score);
        flag(FeatureFlag::FF_VALID) = true;
    } else {
        flag(FeatureFlag::FF_VALID) = false;
    }
    flag(FeatureFlag::FF_TRIANGULATED) = true;
    return flag(FeatureFlag::FF_VALID);
}
