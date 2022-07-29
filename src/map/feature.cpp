#include "feature.h"
#include "frame.h"

void Feature::add_observation(Frame *frame, size_t keypoint_id) {
    frame->m_features[keypoint_id] = this;
    m_observation_map[frame] = keypoint_id;
}

void Feature::remove_observation(Frame *frame, bool suicide_if_empty) {
    size_t keypoint_id = m_observation_map.at(frame);
    frame->m_features[keypoint_id] = nullptr;
    m_observation_map.erase(frame);
}

bool Feature::triangulate() {
    std::vector<Eigen::Matrix<double, 3, 4>> Ps;
    std::vector<Eigen::Vector2d> ps;
    for (const auto &it : observation_map()) {
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
    }

    return true;
}
