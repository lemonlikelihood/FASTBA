#pragma once

#include "common.h"
#include <any>

class ForensicsSupport {
    struct VersionTag;

public:
    enum ForensicsItem {
        FS_RESERVED = 0,
        FS_lvo_info,
        FS_tracker_info,
        FS_stats_info,
        FS_virtual_object_painter_info,
        FS_tracker_landmarks_info,
        FS_accelerometer_info,
        FS_map_info,
        FS_ITEM_COUNT
    };

    ForensicsSupport(const VersionTag &tag);
    ~ForensicsSupport();

    static std::pair<std::any &, std::unique_lock<std::mutex>> get(ForensicsItem item);

private:
    static ForensicsSupport &support();
    std::vector<std::pair<std::any, std::mutex>> storage;
};

#if PROJECTION_ENABLE_FORENSICS
#define forensics(item, var)                                                                       \
    if constexpr (auto [var, var##_lock] = ::ForensicsSupport::get(::ForensicsSupport::FS_##item); \
                  true)
#else
#define forensics(item, var) if constexpr (std::any var; false)
#endif

#define critical_forensics(item, var)                                                              \
    if constexpr (auto [var, var##_lock] = ::ForensicsSupport::get(::ForensicsSupport::FS_##item); \
                  true)

struct output_lvo_info {
    double io_lag;
    double data_fps;
    double real_fps;
};

struct output_tracker_info {
    double web_image_preprocess_time; // web only

    double image_preprocess_time;

    double feature_detection_time;
    double optical_flow_time;
    double optimization_time;
    double frame_track_time;

    double total_track_time;

    double tracked_keypoint_num;
    double rot_diff_from_attitude;

    Eigen::Vector3d velocity_dir;
};

struct output_stats_info {
    double avg_web_image_preprocess_time; // web only

    double avg_image_preprocess_time;

    double avg_feature_detection_time;
    double avg_optical_flow_time;
    double avg_optimization_time;
    double avg_frame_track_time; // sum of above 3 steps and misc operations

    double avg_total_track_time; // sum of all steps
};


struct output_tracker_landmarks_info {
    std::vector<int32_t> flags;
    std::vector<Eigen::Vector3d> ps;
};

struct output_accelerometer_info {
    std::vector<double> acc_norm_raw_data;
    std::vector<double> acc_norm_resampled_data;
    std::vector<double> acc_norm_low_pass_filter;
    std::vector<double> acc_norm_flags;
    std::deque<float> acc_velocity;
};

struct output_map_info {
    std::vector<Pose> poses;
    std::vector<Eigen::Vector3d> ps;
};