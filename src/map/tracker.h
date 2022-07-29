#pragma once

#include "frame.h"
#include <opencv2/opencv.hpp>

class TrackParams {};

class TrackerBase {
public:
    // virtual void detect_keypoints(TrackParams *config) = 0;

    // virtual void
    // track_keypoints(TrackParams *config, Frame *next_frame, std::vector<uint8_t> &status) = 0;

    // virtual void
    // track_monocular(TrackParams *config, Frame *next_frame, std::vector<uint8_t> &status) = 0;

    virtual ~TrackerBase() = default;
    TrackerBase();

    virtual bool track_monocular(Frame *frame) = 0;

    virtual void track_keypoints(
        const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr,
        std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1,
        std::vector<uchar> &mask_out) = 0;

    /// Last set of images (use map so all trackers render in the same order)

    /// Last set of tracked points
    std::vector<cv::KeyPoint>
        m_pts_last; // size_t 表示camera_id 上一帧track到的关键点数据,图像坐标系

    bool m_camera_fisheye;

    cv::Matx33d m_camera_k_opencv;

    cv::Vec4d m_camera_d_opencv;

    Frame *m_last_frame;

    int m_num_features;
};

class KLTTracker : public TrackerBase {
public:
    KLTTracker();
    ~KLTTracker() = default;
    // void detect_keypoints(TrackParams *config) override;
    void detect_keypoints(const std::vector<cv::Mat> &imgpyr, std::vector<cv::KeyPoint> &pts);
    // void
    void track_keypoints(
        const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr,
        std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1,
        std::vector<uchar> &mask_out) override;
    bool track_monocular(Frame *frame) override;

    // void correct_distortion(const );

    // How many pyramid levels to track on and the window size to reduce by
    int m_pyr_levels = 3;                   // 金字塔层数
    cv::Size m_win_size = cv::Size(15, 15); // 光流法窗口大小
                                            // Parameters for our FAST grid detector
    int m_threshold;                        // Fast 角点检测阈值
    int m_grid_x;                           // x 方向的划分格子数
    int m_grid_y;                           // y 方向的划分格子数
    // Minimum pixel distance to be "far away enough" to be a different extracted feature
    int m_min_px_dist; // 相邻两个特征点之间的最小距离

    // Last set of image pyramids
    std::vector<cv::Mat> m_img_pyramid_last; // 上一张图片的金字塔数据
};