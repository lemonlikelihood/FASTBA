#pragma once
#include "../../dataset/configurator.h"
#include "../map/frame.h"
#include "../map/map.h"
#include "../map/tracker.h"
#include <opencv2/opencv.hpp>

class FASTBA {
public:
    FASTBA();
    ~FASTBA();
    void feed_imu();
    void feed_image(std::shared_ptr<ImageData> image, DatasetConfigurator *dataset_config);
    bool track_monocular(Frame *frame);
    std::unique_ptr<Frame>
    create_frame(std::shared_ptr<ImageData> image, DatasetConfigurator *dataset_config);
    std::unique_ptr<Frame> last_frame;
    bool m_initialized;
    std::unique_ptr<TrackerBase> tracker;
};