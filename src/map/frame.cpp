#include "frame.h"
#include "../optimizer/preintegrator.h"
#include "feature.h"
#include "sliding_window.h"

Eigen::Vector2d Frame::remove_k(const Eigen::Vector2d &p) {
    return {(p(0) - m_K(0, 2)) / m_K(0, 0), (p(1) - m_K(1, 2)) / m_K(1, 1)};
}

Eigen::Vector2d Frame::apply_k(const Eigen::Vector2d &p) {
    return {p(0) * m_K(0, 0) + m_K(0, 2), p(1) * m_K(1, 1) + m_K(1, 2)};
}

Pose Frame::get_camera_pose() const {
    Pose pose;
    pose.q = m_pose_body2world.q * m_camera_extri.q_sensor2body;
    pose.p = m_pose_body2world.p + m_pose_body2world.q * m_camera_extri.p_sensor2body;
    return pose;
}

Pose Frame::get_imu_pose() const {
    Pose pose;
    pose.q = m_pose_body2world.q * m_imu_extri.q_sensor2body;
    pose.p = m_pose_body2world.p + m_pose_body2world.q * m_imu_extri.p_sensor2body;
    return pose;
}

Pose Frame::get_body_pose() const {
    Pose pose;
    pose.q = m_pose_body2world.q;
    pose.p = m_pose_body2world.p;
    return pose;
}

void Frame::set_camera_pose(const Pose &pose) {
    this->m_pose_body2world.q = pose.q * m_camera_extri.q_sensor2body.conjugate();
    this->m_pose_body2world.p = pose.p - pose.q * m_camera_extri.p_sensor2body;
}

void Frame::set_imu_pose(const Pose &pose) {
    this->m_pose_body2world.q = pose.q * m_imu_extri.q_sensor2body.conjugate();
    this->m_pose_body2world.p = pose.p - pose.q * m_imu_extri.p_sensor2body;
}

void Frame::append_keypoint(const Eigen::Vector2d &keypoint) {
    m_keypoints.emplace_back(keypoint);
    m_keypoints_normalized.emplace_back(remove_k(keypoint));
    m_reprojection_factors.emplace_back(nullptr);
}

Feature *Frame::get_feature_if_empty_create(size_t keypoint_id) {
    if (m_features[keypoint_id] == nullptr) {
        Feature *feature = m_sliding_window->create_feature();
        feature->add_observation(this, keypoint_id);
    }
    return m_features[keypoint_id];
}