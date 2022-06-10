#include "type.h"
#include <../utils/identifiable.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

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
public:
    size_t m_frame_id;
    int m_index_in_map;
    double m_timestamp;
    std::string m_camera_model;
    std::string m_image_name;
    int m_camera_id;

    Eigen::Matrix3d m_K;
    cv::Mat m_image;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> m_keypoints;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> m_keypoints_normalized;

    Pose m_pose; // pose_{world_center}
    ExtrinsicParams m_camera;

    void detect_keypoints(Config *config, KeypointDetectionMode keypoint_detection_mode);

    // void detect_keypoints(Config *config, KeypointDetectionMode keypoint_detection_mode);

    void track_keypoints(Config *config, Frame *next_frame, std::vector<uint8_t> &status);

    TrackBase *trackFEATS = nullptr;
};