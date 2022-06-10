#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>
#include <vector>


class Observation {
public:
    Observation();
    ~Observation();
    Observation(int obs_id_, int frame_id_, int track_id_, double x_, double y_);
    int m_obs_id;
    int m_frame_id;
    int m_feature_id;
    Eigen::Vector2d uv_normalized;
    Eigen::Vector2d uv;
};

class Feature {
public:
    size_t m_feature_id;
    /// What camera ID our pose is anchored in!! By default the first measurement is the anchor.  局部帧相机
    int m_anchor_cam_id = -1;

    /// Timestamp of anchor clone           局部帧时间戳
    double m_anchor_clone_timestamp;

    /// Triangulated position of this feature, in the anchor frame            局部帧三角化坐标
    Eigen::Vector3d m_p_FinA;

    /// Triangulated position of this feature, in the global frame            全局标标
    Eigen::Vector3d m_p_FinG;

    std::unordered_map<size_t, Observation> m_observation_map;
};