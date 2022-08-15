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

    size_t frame_num() const { return frames.size(); }

    Frame *get_frame(size_t id) const { return frames[id].get(); }

    Frame *get_last_frame() const { return frames[frames.size() - 1].get(); }
    Frame *get_second_to_last_frame() const { return frames[frames.size() - 2].get(); }

    void append_frame(std::unique_ptr<Frame> frame, size_t pos = nil());

    void erase_frame(size_t id);

    size_t feature_num() const { return features.size(); }

    Feature *get_feature(size_t id) const { return features[id].get(); }

    void erase_feature(Feature *feature);

    Feature *create_feature();

    void prune_features(const std::function<bool(const Feature *)> &condition);

    void compute_reprojections();

    void log_feature_reprojections();

private:
    void recycle_feature(Feature *feature);
    std::deque<std::unique_ptr<Frame>> frames;
    std::vector<std::unique_ptr<Feature>> features;
};