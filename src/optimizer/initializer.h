#pragma once
#include "../utils/common.h"
#include "sliding_window_tracker.h"

class Frame;
class Map;

class Initializer {
public:
    Initializer();
    ~Initializer();
    void append_frame(std::unique_ptr<Frame> frame);
    std::unique_ptr<SlidingWindowTracker> init();
    std::unique_ptr<Map> map;

    void mirror_keyframe_map(Map *feature_tracking_map, size_t init_frame_id);

private:
    void solve_gyro_bias();
    void solve_gravity_scale_velocity();
    void refine_scale_velocity_via_gravity();
    bool apply_init();

    bool init_sfm();
    bool init_imu();

    void reset_states();
    void preintegrate();

    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Eigen::Vector3d gravity;
    double scale;
    std::vector<Eigen::Vector3d> velocities;
};