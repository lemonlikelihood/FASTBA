#pragma once
#include <Eigen/Eigen>
#include <memory>
#include <yaml-cpp/yaml.h>

class Config {
private:
    size_t initializer_keyframe_gap;
    size_t initializer_min_matches;
    size_t initializer_min_parallax;
    size_t initializer_min_triangulation;
    size_t initializer_min_points;

    size_t feature_tracker_min_keypoint_distance;
    size_t feature_tracker_max_init_frames;
    size_t feature_tracker_max_keypoint_detection;
    size_t feature_tracker_max_frames;
    bool feature_tracker_predict_keypoints;

    size_t sliding_window_size;
    size_t solver_iteration_limit;
    size_t solver_time_limit;

public:
    Config(const std::string &config_path);
    void load_config(const std::string &config_path);
    size_t get_initializer_keyframe_gap() const { return initializer_keyframe_gap; }
    size_t get_initializer_min_matches() const { return initializer_min_matches; }
    size_t get_initializer_min_parallax() const { return initializer_min_parallax; }
    size_t get_initializer_min_triangulation() const { return initializer_min_triangulation; }
    size_t get_initializer_min_points() const { return initializer_min_points; }

    size_t get_feature_tracker_min_keypoint_distance() const {
        return feature_tracker_min_keypoint_distance;
    }
    size_t get_feature_tracker_max_init_frames() const { return feature_tracker_max_init_frames; }
    size_t get_feature_tracker_max_keypoint_detection() const {
        return feature_tracker_max_keypoint_detection;
    }
    size_t get_feature_tracker_max_frames() const { return feature_tracker_max_frames; }
    bool is_feature_tracker_predict_keypoints() const { return feature_tracker_predict_keypoints; }
    ~Config();
};
