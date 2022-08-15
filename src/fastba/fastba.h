#pragma once
#include "../../dataset/configurator.h"
#include "../map/frame.h"
#include "../map/map.h"
#include "../map/sliding_window.h"
#include "../map/tracker.h"
#include "../optimizer/initializer.h"
#include <opencv2/opencv.hpp>


class FASTBA {
public:
    FASTBA();
    ~FASTBA();
    void feed_imu(const IMUData &imu);
    void get_imu(Frame *frame);
    void feed_image(std::shared_ptr<Image> image, DatasetConfigurator *dataset_config);
    void feed_gt_camera_pose(const Pose &pose);
    bool feed_monocular(Frame* frame);
    std::unique_ptr<Frame>
    create_frame(std::shared_ptr<Image> image, DatasetConfigurator *dataset_config);

    void compute_essential();
    std::unique_ptr<Frame> last_frame;
    bool f_initialized;
    std::unique_ptr<TrackerBase> tracker;
    std::unique_ptr<SlidingWindow> sw;
    std::unique_ptr<Initializer> initializer;
    std::deque<IMUData> imu_buff;
};