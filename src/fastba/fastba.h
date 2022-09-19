#pragma once
#include "../../dataset/configurator.h"
#include "../map/frame.h"
#include "../map/map.h"
// #include "../map/sliding_window.h"
#include "../map/tracker.h"
#include "../optimizer/initializer.h"
#include "../optimizer/sliding_window_tracker.h"
#include <opencv2/opencv.hpp>


class FASTBA {
public:
    FASTBA();
    ~FASTBA();
    void feed_imu(const IMUData &imu);
    void get_imu(Frame *frame);
    void feed_image(std::shared_ptr<Image> image, DatasetConfigurator *dataset_config);
    void track_frame(Map *map, std::unique_ptr<Frame> frame);
    void feed_gt_camera_pose(const Pose &pose);
    bool feed_monocular(Frame *frame);
    std::unique_ptr<Frame>
    create_frame(std::shared_ptr<Image> image, DatasetConfigurator *dataset_config);

    std::tuple<size_t, Pose, MotionState> get_lastest_state() const;

    void compute_essential();
    std::unique_ptr<Frame> last_frame;
    bool f_initialized;
    std::unique_ptr<TrackerBase> tracker;
    std::unique_ptr<Map> map;
    std::unique_ptr<Initializer> initializer;
    std::deque<IMUData> imu_buff;
    std::unique_ptr<Map> feature_tracking_map;
    std::unique_ptr<SlidingWindowTracker> sliding_window_tracker;

    std::tuple<size_t, Pose, MotionState> latest_state;
};