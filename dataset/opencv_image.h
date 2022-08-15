
#pragma once

#include "dataset.h"
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class OpenCvImage : public Image {
public:
    OpenCvImage();
    OpenCvImage(const double &t, const std::string &filename);
    void detect_keypoints(
        std::vector<Eigen::Vector2d> &keypoints, size_t max_points = 0,
        double keypoint_distance = 0.5) const override;
    void track_keypoints(
        const Image *next_image, const std::vector<Eigen::Vector2d> &curr_keypoints,
        std::vector<Eigen::Vector2d> &next_keypoints,
        std::vector<char> &result_status) const override;
    // void detect_segments(
    //     std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d>> &segments,
    //     size_t max_segments = 0) const override;

    void preprocess();
    void correct_distortion(const Eigen::Matrix3d &intrinsics, const Eigen::Vector4d &coeffs);

private:
    static cv::CLAHE *clahe();
    static cv::line_descriptor::LSDDetector *lsd();
    static cv::GFTTDetector *gftt();
    static cv::FastFeatureDetector *fast();
    static cv::ORB *orb();
};