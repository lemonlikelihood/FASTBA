#pragma once

#include <Eigen/Eigen>
#include <deque>
#include <vector>

#include "../../dataset/dataset.h"
#include "../utils/common.h"

class Frame;
class Map;

class SlidingWindowTracker {
public:
    SlidingWindowTracker(std::unique_ptr<Map> keyframe_map);
    ~SlidingWindowTracker();

    void mirror_frame(Map *feature_tracking_map, size_t frame_id);

    std::tuple<Pose, MotionState> get_latest_state() const;

    bool track();

    std::unique_ptr<Map> map;

private:
    void keyframe_check(Frame *frame);
    std::unique_ptr<Frame> frame;
    size_t skipped_frames;
};