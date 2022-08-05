#include "feature.h"
#include "frame.h"

#include "map.h"
#include "sliding_window.h"

SlidingWindow::SlidingWindow() = default;
SlidingWindow::~SlidingWindow() = default;


void SlidingWindow::clear() {
    m_frames.clear();
    m_features.clear();
}

void SlidingWindow::put_frame(std::unique_ptr<Frame> frame, size_t pos) {
    frame->m_sliding_window = this;
    if (pos == nil()) {
        m_frames.emplace_back(std::move(frame));
        pos = m_frames.size() - 1;
    } else {
        m_frames.emplace(m_frames.begin() + pos, std::move(frame));
    }
    if (pos > 0) {
        Frame *frame_i = m_frames[pos - 1].get();
        Frame *frame_j = m_frames[pos].get();
        frame_j->m_preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
        // integration
    }
    if (pos < m_frames.size() - 1) {
        Frame *frame_i = m_frames[pos].get();
        Frame *frame_j = m_frames[pos + 1].get();
        frame_j->m_preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
}

Feature *SlidingWindow::create_feature() {
    std::unique_ptr<Feature> feature = std::make_unique<Feature>();
    feature->m_feature_id_in_sliding_window = m_features.size();
    feature->m_sliding_window = this;
    m_features.emplace_back(std::move(feature));
    log_debug("Sliding window create success");
    return m_features.back().get();
}

void SlidingWindow::erase_feature(Feature *feature) {
    while (feature->observation_map().size() > 0) {
        feature->remove_observation(feature->observation_map().begin()->first, false);
    }
    recycle_feature(feature);
}

void SlidingWindow::prune_features(const std::function<bool(const Feature *)> &condition) {
    std::vector<Feature *> features_to_prune;
    for (size_t i = 0; i < feature_num(); ++i) {
        Feature *feature = get_feature(i);
        if (condition(feature)) {
            features_to_prune.push_back(feature);
        }
    }

    for (Feature *feature : features_to_prune) {
        erase_feature(feature);
    }
}

void SlidingWindow::recycle_feature(Feature *feature) {
    if (feature->m_feature_id_in_sliding_window
        != m_features.back()->m_feature_id_in_sliding_window) {
        m_features[feature->m_feature_id_in_sliding_window].swap(m_features.back());
        m_features[feature->m_feature_id_in_sliding_window]->m_feature_id_in_sliding_window =
            feature->m_feature_id_in_sliding_window;
    }
    m_features.pop_back();
}
