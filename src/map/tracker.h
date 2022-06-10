#pragma once
#include "frame.h"
class TrackParams {};

class TrackerBase {
public:
    virtual void detect_keypoints(TrackParams *config) = 0;

    virtual void
    track_keypoints(TrackParams *config, Frame *next_frame, std::vector<uint8_t> &status) = 0;

    /// Last set of images (use map so all trackers render in the same order)
    std::map<size_t, cv::Mat> img_last; // 上一帧图像数据

    /// Last set of tracked points
    std::vector<cv::KeyPoint> pts_last; // size_t 表示camera_id 上一帧track到的关键点数据,图像坐标系

    int num_features;
};

class KLTTracker : public TrackerBase {
public:
    void detect_keypoints(TrackParams *config) override;
    void
    track_keypoints(TrackParams *config, Frame *next_frame, std::vector<uint8_t> &status) override;
    bool track_monocular(Frame *frame);

    // How many pyramid levels to track on and the window size to reduce by
    int pyr_levels = 3;                   // 金字塔层数
    cv::Size win_size = cv::Size(15, 15); // 光流法窗口大小
                                          // Parameters for our FAST grid detector
    int threshold;                        // Fast 角点检测阈值
    int grid_x;                           // x 方向的划分格子数
    int grid_y;                           // y 方向的划分格子数
    // Minimum pixel distance to be "far away enough" to be a different extracted feature
    int min_px_dist; // 相邻两个特征点之间的最小距离

    // Last set of image pyramids
    std::map<size_t, std::vector<cv::Mat>> img_pyramid_last; // 上一张图片的金字塔数据
};