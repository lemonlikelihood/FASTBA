#pragma once
#include "../../dataset/dataset.h"
#include "../utils/identifiable.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "../optimizer/factor.h"
#include "../optimizer/preintegrator.h"

class Feature;
class SlidingWindow;
class Map;

struct create_if_empty_t {};
extern create_if_empty_t create_if_empty;

enum class FrameFlag {
    FF_KEYFRAME = 0,
    FF_HAVE_ATTI, // attitude
    // FF_HAVE_GYR,  // gyroscope
    // FF_HAVE_GRA,  // gravity
    FF_HAVE_ACC, // accelerometer
    FF_HAVE_V,   // velocity
    FLAG_NUM
};

class Frame : public Flagged<FrameFlag>, public Identifiable<Frame> {
    friend class Feature;
    friend class SlidingWindow;

public:
    Eigen::Vector2d remove_k(const Eigen::Vector2d &p);
    Eigen::Vector2d apply_k(const Eigen::Vector2d &p);

    size_t keypoint_num() const { return m_keypoints.size(); }

    const Eigen::Vector2d &get_keypoint(size_t keypoint_id) const {
        return m_keypoints[keypoint_id];
    }

    void append_keypoint(const Eigen::Vector2d &keypoint);

    const Eigen::Vector2d &get_keypoint_normalized(size_t keypoint_id) const {
        return m_keypoints_normalized[keypoint_id];
    }

    Feature *get_feature(size_t keypoint_id) const { return m_features[keypoint_id]; }

    Feature *get_feature_if_empty_create(size_t keypoint_id);

    Factor *get_preintegration_factor() { return m_preintegration_factor.get(); }

    Factor *get_reprojection_factor(size_t keypoint_index) {
        return m_reprojection_factors[keypoint_index].get();
    }

    Pose get_camera_pose() const;
    Pose get_imu_pose() const;
    Pose get_body_pose() const;

    void set_camera_pose(const Pose &pose);
    void set_imu_pose(const Pose &pose);

    int m_index_in_map;
    double m_timestamp;
    std::string m_camera_model;
    std::string m_image_name;
    int m_camera_id;

    Eigen::Matrix3d m_K;
    Eigen::Matrix2d m_sqrt_inv_cov;
    std::shared_ptr<ImageData> m_image;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> m_keypoints;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> m_keypoints_normalized;

    std::vector<Feature *> m_features;

    Pose m_pose_body2world; // pose_{world_center}
    bool m_fix_pose;

    ExtrinsicParams m_camera_extri;
    ExtrinsicParams m_imu_extri;
    MotionState motion;

    PreIntegrator m_preintegration;

    SlidingWindow *m_sliding_window;

    Map *m_map;

    std::vector<std::unique_ptr<Factor>> m_reprojection_factors;
    std::unique_ptr<Factor> m_preintegration_factor;
};