#pragma once
#include <Eigen/Eigen>
#include <deque>
#include <vector>

#include "../utils/common.h"

class Frame;
class Feature;


class SlidingWindow {
    friend class Feature;

public:
    SlidingWindow();
    ~SlidingWindow();

    void clear();

    size_t frame_num() const { return m_frames.size(); }

    Frame *get_frame(size_t id) const { return m_frames[id].get(); }

    void put_frame(std::unique_ptr<Frame> frame, size_t pos = nil());

    void erase_frame(size_t id);

    size_t feature_num() const { return m_features.size(); }

    Feature *get_feature(size_t id) const { return m_features[id].get(); }

    void erase_feature(Feature *feature);

    Feature *create_feature();

    void prune_features(const std::function<bool(const Feature *)> &condition);

private:
    void recycle_feature(Feature *feature);
    std::deque<std::unique_ptr<Frame>> m_frames;
    std::vector<std::unique_ptr<Feature>> m_features;
};