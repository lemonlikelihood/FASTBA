#pragma once
#include "config.h"
#include <Eigen/Eigen>
#include <memory>
#include <yaml-cpp/yaml.h>

Config::Config(const std::string &config_path) {
    load_config(config_path);
}

Config::~Config() {}

void Config::load_config(const std::string &config_path) {
    try {
        YAML::Node yaml_node = YAML::LoadFile(config_path);

        YAML::Node feature_tracker_node = yaml_node["feature_tracker"]; // 相机内参
        feature_tracker_min_keypoint_distance =
            feature_tracker_node["min_keypoint_distance"].as<double>();
        feature_tracker_max_keypoint_detection =
            feature_tracker_node["max_keypoint_detection"].as<double>();
        feature_tracker_max_init_frames = feature_tracker_node["max_init_frames"].as<double>();
        feature_tracker_max_frames = feature_tracker_node["max_frames"].as<double>();

        YAML::Node initializer_node = yaml_node["initializer"]; // 畸变系数
        initializer_keyframe_gap = initializer_node["keyframe_gap"].as<double>();
        initializer_min_matches = initializer_node["min_matches"].as<double>();
        initializer_min_parallax = initializer_node["min_parallax"].as<double>();
        initializer_min_triangulation = initializer_node["min_triangulation"].as<double>();
        initializer_min_points = initializer_node["min_points"].as<double>();

        sliding_window_size = yaml_node["sliding_window_size"].as<double>();

        YAML::Node solver_node = yaml_node["solver"];
        solver_iteration_limit = solver_node["iteration_limit"].as<double>();
        solver_time_limit = solver_node["time_limit"].as<double>();

    } catch (...) { // for the sake of example, we don't really handle the errors.
        throw;
    }
}
