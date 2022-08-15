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

enum class FeatureFlag { FF_VALID = 0, FF_TRIANGULATED, FLAG_NUM };

class Feature : public Flagged<FeatureFlag> {
public:
    Feature();
    ~Feature();

    const std::map<Frame *, size_t> &observation_map() const { return observation_refs; }

    bool has_observation(Frame *frame) const { return observation_refs.count(frame) > 0; }

    size_t get_observation_index(Frame *frame) const {
        if (has_observation(frame)) {
            return observation_refs.at(frame);
        } else
            return nil();
    }

    const Eigen::Vector2d &get_observation(Frame *frame) const {
        return frame->get_keypoint(observation_refs.at(frame));
    }

    const Eigen::Vector2d &get_observation_normalized(Frame *frame) const {
        return frame->get_keypoint_normalized(observation_refs.at(frame));
    }

    Frame *first_frame() const { return observation_refs.begin()->first; }
    Frame *last_frame() const { return observation_refs.rbegin()->first; }

    void add_observation(Frame *frame, size_t keypoint_index);
    void remove_observation(Frame *, bool suicide_if_empty = true);

    bool triangulate();
    size_t index_in_sw;
    SlidingWindow *sw;
    Eigen::Vector3d p_in_A;
    double inv_depth_in_A;
    Eigen::Vector3d p_in_G;
    double reprojection_error = 0;

private:
    std::map<Frame *, size_t> observation_refs;
};