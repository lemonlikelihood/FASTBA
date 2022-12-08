#pragma once
#include <Eigen/Eigen>
#include <deque>
#include <vector>

#include "../utils/common.h"

class Frame;
class Feature;


class Map {
    friend class Feature;

public:
    Map();
    virtual ~Map();

    void clear();

    size_t frame_num() const { return frames.size(); }

    Frame *get_frame(size_t index) const { return frames[index].get(); }

    Frame *get_first_frame() const { return frames[0].get(); }

    Frame *get_last_frame() const { return frames[frames.size() - 1].get(); }
    Frame *get_second_to_last_frame() const { return frames[frames.size() - 2].get(); }

    void append_frame(std::unique_ptr<Frame> frame, size_t pos = nil());

    void erase_frame(size_t index);

    void marginalize_frame(size_t index);

    size_t get_frame_index_by_id(size_t id) const;

    size_t feature_num() const { return features.size(); }

    Feature *get_feature(size_t id) const { return features[id].get(); }

    Feature *get_feature_by_id(size_t id) const;

    Feature *create_feature();

    void update_feature_state();

    void erase_feature(Feature *feature);

    void prune_features(const std::function<bool(const Feature *)> &condition);

    void set_marginalization_factor(std::unique_ptr<Factor> factor);

    Factor *get_marginalization_factor() { return marginalization_factor.get(); }

    std::unique_lock<std::mutex> lock() const { return std::unique_lock(map_mutex); }

    std::pair<double, double> compute_reprojections();

    std::pair<double, double> compute_reprojections_without_last_frame();

    void log_feature_reprojections();

private:
    void recycle_feature(Feature *feature);
    std::deque<std::unique_ptr<Frame>> frames;
    std::vector<std::unique_ptr<Feature>> features;
    std::map<size_t, Feature *> feature_id_map;

    std::unique_ptr<Factor> marginalization_factor;
    mutable std::mutex map_mutex;
};