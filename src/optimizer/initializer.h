#pragma once
#include "../utils/common.h"

class Frame;
class SlidingWindow;

class Initializer {
public:
    Initializer();
    ~Initializer();
    void append_frame(std::unique_ptr<Frame> frame);
    std::unique_ptr<SlidingWindow> init();
    std::unique_ptr<SlidingWindow> sw;
    std::unique_ptr<SlidingWindow> map;

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