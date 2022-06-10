#pragma once
#include "../map/map.h"
#include <opencv2/opencv.hpp>


class FASTBA {
public:
    FASTBA();
    ~FASTBA() {}
    void feed_imu();
    void feed_image(double t, cv::Mat image);
    bool track_monocular(Frame *frame);
    Frame *last_frame;
}