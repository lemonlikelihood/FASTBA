#include "feature.h"
#include "../geometry/stereo.h"
#include "frame.h"

Feature::Feature() = default;
Feature::~Feature() = default;

void Feature::add_observation(Frame *frame, size_t keypoint_id) {
    frame->m_features[keypoint_id] = this;
    m_observation_map[frame] = keypoint_id;
}

void Feature::remove_observation(Frame *frame, bool suicide_if_empty) {
    size_t keypoint_id = m_observation_map.at(frame);
    frame->m_features[keypoint_id] = nullptr;
    frame->m_reprojection_factors[keypoint_id].reset();
    m_observation_map.erase(frame);
    if (suicide_if_empty && m_observation_map.size() == 0) {
        m_sliding_window->recycle_feature(this);
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
        R = pose.q.toRotationMatrix();
        t = -(R * pose.p);
        P << R, t;
        Ps.push_back(P);
        ps.push_back(it.first->get_keypoint(it.second));
    }

    Eigen::Vector3d p;
    if (triangulate_point_checked(Ps, ps, p)) {
        m_p_FinG = p;
        m_is_triangulated = true;
    } else {
        m_is_triangulated = false;
    }

    return m_is_triangulated;
}
