#include "sliding_window_tracker.h"
#include "../map/feature.h"
#include "../map/frame.h"
#include "../map/map.h"

#include "../optimizer/bundle_adjustor.h"
#include "../optimizer/pnp.h"

SlidingWindowTracker::SlidingWindowTracker(std::unique_ptr<Map> keyframe_map)
    : map(std::move(keyframe_map)), skipped_frames(0) {
    log_info("[Create SlidingWindowTracker Begin]");
    for (size_t j = 1; j < map->frame_num(); ++j) {
        Frame *frame_i = map->get_frame(j - 1);
        Frame *frame_j = map->get_frame(j);
        frame_j->preintegration.integrate(
            frame_j->image->t, frame_i->motion.bg, frame_i->motion.ba, true, true);
    }
    log_info("[Create SlidingWindowTracker Over]");
}

SlidingWindowTracker::~SlidingWindowTracker() = default;

void SlidingWindowTracker::mirror_frame(Map *feature_tracking_map, size_t frame_id) {
    Frame *new_frame_i = map->get_last_frame();
    size_t frame_index_i = feature_tracking_map->get_frame_index_by_id(new_frame_i->id());
    size_t frame_index_j = feature_tracking_map->get_frame_index_by_id(frame_id);

    if (frame_index_i == nil() || frame_index_j == nil())
        return;

    Frame *old_frame_i = feature_tracking_map->get_frame(frame_index_i);
    Frame *old_frame_j = feature_tracking_map->get_frame(frame_index_j);

    frame = old_frame_j->clone();
    Frame *new_frame_j = frame.get();

    for (size_t ki = 0; ki < old_frame_i->keypoint_num(); ++ki) {
        if (Feature *feature = old_frame_i->get_feature(ki)) {
            if (size_t kj = feature->get_observation_index(old_frame_j); kj != nil()) {
                new_frame_i->get_feature_if_empty_create(ki)->add_observation(new_frame_j, kj);
            }
        }
    }
}

bool SlidingWindowTracker::track() {
    log_info("[SlidingWindowTracker]: track frame {} begin ...", frame->id());
    for (size_t i = 0; i < map->frame_num(); ++i) {
        log_info(
            "[SlidingWindowTracker]: map, {} : {}, keyframe: {}", i, map->get_frame(i)->id(),
            map->get_frame(i)->flag(FrameFlag::FF_KEYFRAME));
    }

    size_t fid = frame->id();
    Frame *last_frame = map->get_last_frame();

    log_info(
        "[SlidingWindowTracker]: last_frame {} is keyframe {}", last_frame->id(),
        last_frame->flag(FrameFlag::FF_KEYFRAME));
    log_info(
        "[SlidingWindowTracker]: preintegration ds: {}, bg: {}, ba: {}",
        frame->preintegration.data.size(), last_frame->motion.bg.transpose(),
        last_frame->motion.ba.transpose());

    frame->preintegration.integrate(
        frame->image->t, last_frame->motion.bg, last_frame->motion.ba, true, true);
    // log_info("[SlidingWindowTracker]: preintegration over");
    frame->preintegration.predict(last_frame, frame.get());

    {
        auto [point_num, reprojection_error] = map->compute_reprojections();
        log_info(
            "[SlidingWindowTracker]: frame {} before pnp, point_num: {}, reprojection_error: {}",
            frame->id(), point_num, reprojection_error);
        auto [point_num1, reprojection_error1] = map->compute_reprojections_without_last_frame();
        log_info(
            "[SlidingWindowTracker]: frame {} before pnp without lastframe, point_num: {}, "
            "reprojection_error: {}",
            frame->id(), point_num1, reprojection_error1);
    }

    visual_inertial_pnp(map.get(), frame.get(), true);

    {
        auto [point_num, reprojection_error] = map->compute_reprojections();
        log_info(
            "[SlidingWindowTracker]: frame {} visual_inertial_pnp, point_num: {}, "
            "reprojection_error: {}",
            frame->id(), point_num, reprojection_error);
        if (reprojection_error > 3.0) {
            log_error(
                "[SlidingWindowTracker]: pnp reprojection_error is bigger {}", reprojection_error);
            tracking_state = TRACKING_FAILURE;
            getchar();
        }
    }

    keyframe_check(frame.get());

    int new_triangulated_feature = 0;
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        Feature *feature = frame->get_feature(i);
        if (!feature)
            continue;
        if (feature->flag(FeatureFlag::FF_VALID)) {
            feature->flag(FeatureFlag::FF_FIXED) = false;
            continue;
        }
        if (feature->triangulate()) {
            new_triangulated_feature++;
            feature->flag(FeatureFlag::FF_VALID) = true;
        }
    }
    log_info("[SlidingWindowTracker]: new_triangulated_feature: {}\n", new_triangulated_feature);

    if (last_frame->flag(FrameFlag::FF_KEYFRAME)) {
        // log_info("[SlidingWindowTracker]: last frame {} is keyframe", last_frame->id());
        while (map->frame_num() >= 8 + 1) {
            log_info("[SlidingWindowTracker]: need marginalization");
            map->marginalize_frame(0);
            map->update_feature_state();
            log_info("[marginalize_frame end]");
            {
                auto [point_num, reprojection_error] = map->compute_reprojections();
                log_info(
                    "[SlidingWindowTracker]: after margin_frame 0, point_num: {}, "
                    "reprojection_error: "
                    "{}",
                    point_num, reprojection_error);
            }
        }
        map->append_frame(std::move(frame));
        map->update_feature_state();

        if (!map->get_marginalization_factor()) {
            std::vector<Frame *> init_frames;
            for (size_t i = 1; i < map->frame_num(); ++i) {
                init_frames.push_back(map->get_frame(i - 1));
            }

            Eigen::MatrixXd init_info_mat(
                ES_SIZE * (map->frame_num() - 1), ES_SIZE * (map->frame_num() - 1));
            Eigen::VectorXd init_info_vec(ES_SIZE * (map->frame_num() - 1));

            init_info_mat.setZero();
            init_info_vec.setZero();

            init_info_mat.block<3, 3>(ES_P, ES_P) = 1.0e15 * Eigen::Matrix3d::Identity();
            init_info_mat.block<3, 3>(ES_Q, ES_Q) = 1.0e15 * Eigen::Matrix3d::Identity();
            map->set_marginalization_factor(Factor::create_marginalization_error(
                init_info_mat, init_info_vec, std::move(init_frames)));
            // log_info("[set_marginalization_factor] \n");
            // log_info("[set_marginalization_factor]: init_info_mat: \n{}", init_info_mat);
            // log_info(
            //     "[set_marginalization_factor]: init_info_vec: \n{}", init_info_vec.transpose());
        }

        BundleAdjustor().solve(map.get(), true, 50, 1e6);
        auto [point_num, reprojection_error] = map->compute_reprojections();
        log_info(
            "[SlidingWindowTracker]: last frame is keyframe, BundleAdjustor, point_num: {}, "
            "reprojection_error: {}",
            point_num, reprojection_error);
    } else {
        // log_info("[SlidingWindowTracker]: last_frame {} is not keyframe", last_frame->id());
        log_info(
            "[SlidingWindowTracker]: append last_frame {} imu data to frame {}", last_frame->id(),
            fid);
        const std::vector<IMUData> &data = last_frame->preintegration.data;
        frame->preintegration.data.insert(
            frame->preintegration.data.begin(), data.begin(), data.end());
        log_info(
            "[SlidingWindowTracker]: preintegration begin, data size: {}, bg: {}, ba: {}",
            frame->preintegration.data.size(), last_frame->motion.bg.transpose(),
            last_frame->motion.ba.transpose());
        frame->preintegration.integrate(
            frame->image->t, last_frame->motion.bg, last_frame->motion.ba, true, true);

        map->erase_frame(map->frame_num() - 1);
        map->update_feature_state();
        log_info("[SlidingWindowTracker]: erase last_frame {}", last_frame->id());

        {
            auto [point_num, reprojection_error] = map->compute_reprojections();
            log_info(
                "[SlidingWindowTracker]: after erase_frame {}, point_num: {}, reprojection_error: "
                "{}",
                last_frame->id(), point_num, reprojection_error);
        }

        map->append_frame(std::move(frame));
        log_info("[SlidingWindowTracker]: append frame {}", fid);
        {
            auto [point_num, reprojection_error] = map->compute_reprojections();
            log_info(
                "[SlidingWindowTracker]: after append frame {}, point_num: {}, reprojection_error: "
                "{}",
                fid, point_num, reprojection_error);
        }
        // BundleAdjustor().solve(map.get(), true, 50, 1e6);
    }

    map->prune_features([](const Feature *feature) {
        return (!feature->flag(FeatureFlag::FF_VALID) || feature->reprojection_error > 3.0);
    });

    {
        auto [point_num, reprojection_error] = map->compute_reprojections();
        log_info(
            "[SlidingWindowTracker]: frame {} after prune feature, point_num: {}, "
            "reprojection_error: {}",
            fid, point_num, reprojection_error);
    }

    log_info("[SlidingWindowTracker]: track frame {} over", fid);
    log_info("");
    tracking_state = TRACKING_SUCCESS;
    return true;
}


void SlidingWindowTracker::keyframe_check(Frame *frame) {
    Frame *last_keyframe = nullptr;
    for (size_t i = 0; i < map->frame_num(); ++i) {
        if (map->get_frame(map->frame_num() - i - 1)->flag(FrameFlag::FF_KEYFRAME)) {
            last_keyframe = map->get_frame(map->frame_num() - i - 1);
            break;
        }
    }

    if (!last_keyframe) {
        log_info("[SlidingWindowTracker]: keyframe check, last_keyframe is nullptr");
        frame->flag(FrameFlag::FF_KEYFRAME) = true;
    } else {
        Eigen::Quaterniond qij = (last_keyframe->camera_extri.q.conjugate()
                                  * last_keyframe->imu_extri.q * frame->preintegration.delta.q
                                  * frame->imu_extri.q.conjugate() * frame->camera_extri.q)
                                     .conjugate();
        std::vector<double> parallax_list;
        for (size_t kj = 0; kj < frame->keypoint_num(); ++kj) {
            Feature *feature = frame->get_feature(kj);
            if (!feature)
                continue;
            size_t ki = feature->get_observation_index(last_keyframe);
            if (ki == nil())
                continue;
            Eigen::Vector2d pi = last_keyframe->apply_k(
                (qij * last_keyframe->get_keypoint_normalized(ki).homogeneous()).hnormalized());
            Eigen::Vector2d pj = frame->get_keypoint(kj);
            parallax_list.push_back((pi - pj).norm());
        }
        if (parallax_list.size() < 50) {
            log_info(
                "[SlidingWindowTracker]: keyframe check, parallax_list.size: {} < 50",
                parallax_list.size());
            frame->flag(FrameFlag::FF_KEYFRAME) = true;
        } else {
            std::sort(parallax_list.begin(), parallax_list.end());
            double parallax = parallax_list[parallax_list.size() * 4 / 5];
            if (parallax > 50) {
                log_info("[SlidingWindowTracker]: keyframe check, parallax: {} > 50", parallax);
                frame->flag(FrameFlag::FF_KEYFRAME) = true;
            } else {
                skipped_frames++;
            }
        }
    }
    if (skipped_frames > 10) {
        log_info("[SlidingWindowTracker]: keyframe check, skipped_frames >10");
        frame->flag(FrameFlag::FF_KEYFRAME) = true;
    }
    if (frame->flag(FrameFlag::FF_KEYFRAME)) {
        skipped_frames = 0;
    }

    log_info(
        "[SlidingWindowTracker]: keyframe check, frame {} keyframe {}", frame->id(),
        frame->flag(FrameFlag::FF_KEYFRAME));
}

std::tuple<TrackingState, Pose, MotionState> SlidingWindowTracker::get_latest_state() const {
    const Frame *frame = map->get_last_frame();
    return {tracking_state, frame->pose, frame->motion};
}