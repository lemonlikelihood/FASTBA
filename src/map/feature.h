#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "frame.h"
#include "sliding_window.h"

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
    Feature();
    ~Feature();

    const std::map<Frame *, size_t> &observation_map() const { return m_observation_map; }

    bool has_observation(Frame *frame) const { return m_observation_map.count(frame) > 0; }

    size_t get_observation_id(Frame *frame) const {
        if (has_observation(frame)) {
            return m_observation_map.at(frame);
        } else
            return nil();
    }

    const Eigen::Vector2d &get_observation(Frame *frame) const {
        return frame->get_keypoint(m_observation_map.at(frame));
    }

    void add_observation(Frame *frame, size_t keypoint_id);
    void remove_observation(Frame *, bool suicide_if_empty = true);

    bool triangulate();

    size_t m_feature_id_in_sliding_window;
    /// What camera ID our pose is anchored in!! By default the first measurement is the anchor.  局部帧相机
    int m_anchor_cam_id = -1;

    /// Timestamp of anchor clone           局部帧时间戳
    double m_anchor_clone_timestamp;

    /// Triangulated position of this feature, in the anchor frame            局部帧三角化坐标
    Eigen::Vector3d m_p_FinA;

    /// Triangulated position of this feature, in the global frame            全局标标
    Eigen::Vector3d m_p_FinG;

    bool m_is_triangulated;

    SlidingWindow *m_sliding_window;

private:
    std::map<Frame *, size_t> m_observation_map;
};