#include "initializer.h"
#include "../geometry/essential.h"
#include "../geometry/homography.h"
#include "../geometry/lie_algebra.h"
#include "../geometry/stereo.h"
#include "../map/feature.h"
#include "../map/frame.h"
#include "../map/map.h"
#include "bundle_adjustor.h"
#include "pnp.h"

#include "../../dataset/dataset.h"
#include "../utils/euler_angle.h"


Initializer::Initializer() {}

Initializer::~Initializer() = default;

void Initializer::mirror_keyframe_map(Map *feature_tracking_map, size_t init_frame_id) {
    size_t init_frame_index_last = feature_tracking_map->get_frame_index_by_id(init_frame_id);
    size_t init_frame_index_gap = 5;
    size_t init_frame_index_distance = init_frame_index_gap * (8 - 1);

    init_frame_id = nil();
    if (init_frame_index_last < init_frame_index_distance) {
        log_error(
            "[initializer]: mirror_keyframe_map, init_frame_index_last {} < "
            "init_frame_index_distance {}",
            init_frame_index_last, init_frame_index_distance);
        map.reset();
        return;
    }

    size_t init_frame_index_first = init_frame_index_last - init_frame_index_distance;

    std::vector<size_t> init_keyframe_indices;

    for (size_t i = 0; i < 8; ++i) {
        init_keyframe_indices.push_back(init_frame_index_first + i * init_frame_index_gap);
    }

    map = std::make_unique<Map>();
    for (size_t index : init_keyframe_indices) {
        map->append_frame(feature_tracking_map->get_frame(index)->clone());
    }

    for (size_t j = 1; j < map->frame_num(); ++j) {
        Frame *old_frame_i = feature_tracking_map->get_frame(init_keyframe_indices[j - 1]);
        Frame *old_frame_j = feature_tracking_map->get_frame(init_keyframe_indices[j]);
        Frame *new_frame_i = map->get_frame(j - 1);
        Frame *new_frame_j = map->get_frame(j);

        for (size_t ki = 0; ki < old_frame_i->keypoint_num(); ++ki) {
            if (Feature *feature = old_frame_i->get_feature(ki)) {
                if (size_t kj = feature->get_observation_index(old_frame_j); kj != nil()) {
                    new_frame_i->get_feature_if_empty_create(ki)->add_observation(new_frame_j, kj);
                }
            }
        }

        new_frame_j->preintegration.data.clear();
        for (size_t f = init_keyframe_indices[j - 1]; f < init_keyframe_indices[j]; ++f) {
            Frame *old_frame = feature_tracking_map->get_frame(f + 1);
            std::vector<IMUData> &old_data = old_frame->preintegration.data;
            std::vector<IMUData> &new_data = new_frame_j->preintegration.data;
            new_data.insert(new_data.end(), old_data.begin(), old_data.end());
        }
    }
}

bool Initializer::init_sfm() {

    log_info("[initializer]: init_sfm begin...");

    Frame *init_frame_i = map->get_frame(0);
    Frame *init_frame_j = map->get_last_frame();

    Eigen::Matrix3d init_R;
    Eigen::Vector3d init_T;
    double init_score;

    std::vector<Eigen::Vector2d> frame_i_keypoints;
    std::vector<Eigen::Vector2d> frame_j_keypoints;

    std::vector<Eigen::Vector3d> init_points;
    std::vector<char> init_point_status;
    std::vector<std::pair<size_t, size_t>> init_matches;

    double total_parallax = 0;
    int common_track_num = 0;

    for (size_t ki = 0; ki < init_frame_i->keypoint_num(); ++ki) {
        Feature *feature = init_frame_i->get_feature(ki);
        if (!feature)
            continue;
        size_t kj = feature->get_observation_index(init_frame_j);
        if (kj == nil())
            continue;
        frame_i_keypoints.push_back(init_frame_i->get_keypoint_normalized(ki));
        frame_j_keypoints.push_back(init_frame_j->get_keypoint_normalized(kj));
        init_matches.emplace_back(ki, kj);
        total_parallax += (init_frame_i->get_keypoint(ki) - init_frame_j->get_keypoint(kj)).norm();
        common_track_num++;
    }

    if (common_track_num < 50) {
        log_error("[initializer]: init_sfm common_track_num: {} < 50", common_track_num);
        return false;
    }

    log_debug("[initializer]: init_sfm init common_track_num : {}", common_track_num);
    total_parallax /= std::max(common_track_num, 1);
    log_debug("[initializer]: init_sfm total_parallax : {}", total_parallax);
    if (total_parallax < 10) {
        log_error("[initializer]: init_sfm total_parallax : {} < 10", total_parallax);
        return false;
    }

    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> Ts;

    Eigen::Matrix3d RH1, RH2;
    Eigen::Vector3d TH1, TH2, nH1, nH2;
    Eigen::Matrix3d H = find_homography_matrix(
        frame_i_keypoints, frame_j_keypoints, 0.7 / init_frame_i->K(0, 0), 0.999, 1000, 648);


    if (!decompose_homography(
            H, RH1, RH2, TH1, TH2, nH1, nH2)) { // 计算单应矩阵，如果是纯旋转，跳到下一帧
        log_error("[initializer]: init_sfm pure rotation");
        return false; // is pure rotation
    }
    TH1 = TH1.normalized();
    TH2 = TH2.normalized();
    Rs.insert(Rs.end(), {RH1, RH1, RH2, RH2});
    Ts.insert(Ts.end(), {TH1, -TH1, TH2, -TH2});

    Eigen::Matrix3d RE1, RE2;
    Eigen::Vector3d TE;
    Eigen::Matrix3d E = find_essential_matrix(
        frame_i_keypoints, frame_j_keypoints, 0.7 / init_frame_i->K(0, 0), 0.999, 1000, 648);
    decompose_essential(E, RE1, RE2, TE);
    TE = TE.normalized();
    Rs.insert(Rs.end(), {RE1, RE1, RE2, RE2});
    Ts.insert(Ts.end(), {TE, -TE, TE, -TE});

    // 对单应矩阵和本质矩阵分解得到的8组R，t分别三角化求最好的三角化个数和最小的重投影误差，最小的重投影误差优先级别高
    size_t triangulated_num = triangulate_from_rt_scored(
        frame_i_keypoints, frame_j_keypoints, Rs, Ts, 20, init_points, init_R, init_T,
        init_point_status, init_score);

    if (triangulated_num < 20) { // 如果三角化最好的个数不满足阈值，跳到下一帧
        log_error(
            "[initializer]: init_sfm failed: triangulation num is {} and less than 20",
            triangulated_num);
        return false;
    }

    log_info("[initializer]: init_sfm init triangulated_num: {}", triangulated_num);

    // {
    //     const int32_t rows = init_frame_i->image->image.rows;
    //     const int32_t cols = init_frame_i->image->image.cols;
    //     cv::Mat img1 = init_frame_i->image->image;
    //     cv::Mat img2 = init_frame_j->image->image;
    //     cv::Mat combined(rows * 2, cols, CV_8UC1);
    //     img1.copyTo(combined.rowRange(0, rows));
    //     img2.copyTo(combined.rowRange(rows, rows * 2));
    //     cv::cvtColor(combined, combined, cv::COLOR_GRAY2RGBA);
    //     for (int i = 0; i < frame_i_keypoints.size(); i++) {
    //         Eigen::Vector2d pi = init_frame_i->apply_k(frame_i_keypoints[i]);
    //         cv::Point2d cv_pi = {pi.x(), pi.y()};
    //         cv::circle(combined, cv_pi, 5, cv::Scalar(255, 0, 0));
    //         Eigen::Vector2d pj = init_frame_j->apply_k(frame_j_keypoints[i]);
    //         cv::Point2d cv_pj = {pj.x(), pj.y()};
    //         log_info("[init pair]: i: {}, pi: {}, pj: {}", i, pi.transpose(), pj.transpose());
    //         cv::circle(combined, cv_pj + cv::Point2d(0, rows), 5, cv::Scalar(0, 255, 0));
    //         cv::line(combined, cv_pi, cv_pj + cv::Point2d(0, rows), cv::Scalar(0, 0, 255));
    //     }
    //     cv::imshow("init pair", combined);
    //     cv::waitKey(1);
    // }

    // [2.3] set init states
    Pose pose; // 设置第一帧和最后一帧的camera位姿
    pose.q.setIdentity();
    pose.p.setZero();
    init_frame_i->set_camera_pose(pose);
    pose.q = init_R.transpose();
    pose.p = -(init_R.transpose() * init_T);
    init_frame_j->set_camera_pose(pose);

    Pose pose_i = init_frame_i->get_camera_pose();
    log_debug(
        "[initializer]: init_sfm init pose_i: {} {}", pose_i.q.coeffs().transpose(),
        pose_i.p.transpose());

    Pose pose_j = init_frame_j->get_camera_pose();
    log_debug(
        "[initializer]: init_sfm pose_j: {} {}", pose_j.q.coeffs().transpose(),
        pose_j.p.transpose());

    int init_points_num = 0;
    for (size_t k = 0; k < init_points.size();
         ++k) { // 对于成功三角化的点，设置第一帧作为参考帧，也就是作为世界坐标系，固定不变
        if (init_point_status[k] == 0)
            continue;
        Feature *feature = init_frame_i->get_feature(init_matches[k].first);
        feature->p_in_G = init_points[k]; // 3D点信息与track绑定，就是与每一帧绑定
        feature->flag(FeatureFlag::FF_VALID) = true;
        init_points_num++;
        log_debug(
            "[initializer]: init feature i:{} p: {}", init_points_num, feature->p_in_G.transpose());
    }

    log_info("[initializer]: init frame i,j reprojection error {}", init_score);
    // sw->compute_reprojections();
    // sw->log_feature_reprojections();

    // [2.4] solve other frames via pnp    // pnp 通过前一帧的位姿和初始化的3D点来求解后一帧的位姿
    for (size_t j = 1; j + 1 < map->frame_num();
         ++j) { // 这个过程不会增加新的3D点，也不会改变初始化3D点的坐标
        Frame *frame_i = map->get_frame(j - 1);
        Frame *frame_j = map->get_frame(j);
        frame_j->set_camera_pose(frame_i->get_camera_pose()); // 后一帧的位姿初始值为前一帧
        visual_inertial_pnp(map.get(), frame_j, false);       // ceres求解位姿
    }

    {
        double error = map->compute_reprojections();
        // map->log_feature_reprojections();
        log_info("[initializer]: pnp reprojection error: {}", error);
    }

    // [2.5] triangulate more points
    for (size_t i = 0; i < map->feature_num();
         ++i) { // 对map中其他的track也进行三角化，增加更多的3D点
        Feature *feature = map->get_feature(i);
        if (feature->flag(FeatureFlag::FF_VALID))
            continue;           // 如果是初始三角化的点，不可以改变
        feature->triangulate(); // 给track设置landmark值，并固定不变
    }

    // [3] sfm

    // [3.1] bundle adjustment  // sfm固定第一帧的位姿，固定初始化的3D点
    map->get_frame(0)->flag(FrameFlag::FF_FIX_POSE) =
        true; // 利用所有的观测和相机位姿最小化重投影误差估计，优化其中非固定的变量，包括相机位姿和3D点
    if (!BundleAdjustor().solve(map.get(), false, 50, 1e6)) {
        return false;
    }

    // map->log_feature_reprojections();
    {
        double error = map->compute_reprojections();
        // map->log_feature_reprojections();
        log_info("[initializer]: BundleAdjustor reprojection error: {}", error);
    }

    // // [3.2] cleanup invalid points                                   //  从map中清除没有成功三角化的track，或者三角化误差（平均重投影误差）大于1的track
    map->prune_features([](const Feature *feature) {
        return !feature->flag(FeatureFlag::FF_VALID) || feature->reprojection_error > 0.5;
    });

    {
        double error = map->compute_reprojections();
        // map->log_feature_reprojections();
        log_info("[initializer]: BundleAdjustor prune_features reprojection error: {}", error);
    }

    log_info("[initializer]: init_sfm successful");

    return true;
}


// bool Initializer::init_sfm() {

//     log_info("[init_sfm]: begin...");

//     Frame *init_frame_i = nullptr;
//     Frame *init_frame_j = map->get_last_frame();

//     log_info("[init_sfm]: init_frame_j: {}", init_frame_j->id());

//     size_t init_frame_i_id = nil();

//     Eigen::Matrix3d init_R;
//     Eigen::Vector3d init_T;
//     double init_score;

//     std::vector<Eigen::Vector2d> frame_i_keypoints;
//     std::vector<Eigen::Vector2d> frame_j_keypoints;

//     std::vector<Eigen::Vector3d> init_points;
//     std::vector<char> init_point_status;
//     std::vector<std::pair<size_t, size_t>> init_matches;

//     Frame *frame_j = init_frame_j;
//     for (size_t frame_i_id = 0; frame_i_id + 15 < sw->frame_num(); ++frame_i_id) {
//         double total_parallax = 0;
//         int common_track_num = 0;
//         frame_i_keypoints.clear();
//         frame_j_keypoints.clear();
//         init_matches.clear();

//         Frame *frame_i = sw->get_frame(frame_i_id);
//         log_info("[init_sfm]: test init_frame_i : {}", frame_i->id());
//         for (size_t ki = 0; ki < frame_i->keypoint_num(); ++ki) {
//             Feature *feature = frame_i->get_feature(ki);
//             if (!feature)
//                 continue;
//             size_t kj = feature->get_observation_index(frame_j);
//             if (kj == nil())
//                 continue;
//             frame_i_keypoints.push_back(frame_i->get_keypoint_normalized(ki));
//             frame_j_keypoints.push_back(frame_j->get_keypoint_normalized(kj));
//             init_matches.emplace_back(ki, kj);
//             total_parallax += (frame_i->get_keypoint(ki) - frame_j->get_keypoint(kj)).norm();
//             common_track_num++;
//         }

//         log_info("[init_sfm]: get common observation : {}", common_track_num);

//         if (common_track_num < 50) {
//             log_error("common_track_num: {} < 50", common_track_num);
//             continue;
//         }
//         total_parallax /= std::max(common_track_num, 1);
//         log_info("[init_sfm]: get total_parallax : {}", total_parallax);
//         if (total_parallax < 10) {
//             log_error("total_parallax: {} < 10", total_parallax);
//             continue;
//         }

//         std::vector<Eigen::Matrix3d> Rs;
//         std::vector<Eigen::Vector3d> Ts;

//         Eigen::Matrix3d RH1, RH2;
//         Eigen::Vector3d TH1, TH2, nH1, nH2;
//         Eigen::Matrix3d H = find_homography_matrix(
//             frame_i_keypoints, frame_j_keypoints, 0.7 / frame_i->K(0, 0), 0.999, 1000, 648);


//         if (!decompose_homography(
//                 H, RH1, RH2, TH1, TH2, nH1, nH2)) { // 计算单应矩阵，如果是纯旋转，跳到下一帧
//             log_info("pure rotation");
//             continue; // is pure rotation
//         }
//         TH1 = TH1.normalized();
//         TH2 = TH2.normalized();
//         Rs.insert(Rs.end(), {RH1, RH1, RH2, RH2});
//         Ts.insert(Ts.end(), {TH1, -TH1, TH2, -TH2});

//         Eigen::Matrix3d RE1, RE2;
//         Eigen::Vector3d TE;
//         Eigen::Matrix3d E = find_essential_matrix(
//             frame_i_keypoints, frame_j_keypoints, 0.7 / frame_i->K(0, 0), 0.999, 1000, 648);
//         decompose_essential(E, RE1, RE2, TE);
//         TE = TE.normalized();
//         Rs.insert(Rs.end(), {RE1, RE1, RE2, RE2});
//         Ts.insert(Ts.end(), {TE, -TE, TE, -TE});

//         // 对单应矩阵和本质矩阵分解得到的8组R，t分别三角化求最好的三角化个数和最小的重投影误差，最小的重投影误差优先级别高
//         size_t triangulated_num = triangulate_from_rt_scored(
//             frame_i_keypoints, frame_j_keypoints, Rs, Ts, 20, init_points, init_R, init_T,
//             init_point_status, init_score);

//         if (triangulated_num < 20) { // 如果三角化最好的个数不满足阈值，跳到下一帧
//             log_info("SFM Init Failed: triangulation num is {} and less than 20", triangulated_num);
//             continue;
//         }

//         log_info("[triangulated_num]: {}", triangulated_num);
//         init_frame_i = frame_i;
//         init_frame_i_id = frame_i_id;
//         break;
//     }

//     log_info("[init_sfm]:get init_frame_i ...");

//     if (!init_frame_i)
//         return false;

//     log_info("init_frame_id: {}", init_frame_i_id);

//     const int32_t rows = init_frame_i->image->image.rows;
//     const int32_t cols = init_frame_i->image->image.cols;
//     cv::Mat img1 = init_frame_i->image->image;
//     cv::Mat img2 = init_frame_j->image->image;
//     cv::Mat combined(rows * 2, cols, CV_8UC1);
//     img1.copyTo(combined.rowRange(0, rows));
//     img2.copyTo(combined.rowRange(rows, rows * 2));
//     cv::cvtColor(combined, combined, cv::COLOR_GRAY2RGBA);
//     for (int i = 0; i < frame_i_keypoints.size(); i++) {
//         Eigen::Vector2d pi = init_frame_i->apply_k(frame_i_keypoints[i]);
//         cv::Point2d cv_pi = {pi.x(), pi.y()};
//         cv::circle(combined, cv_pi, 5, cv::Scalar(255, 0, 0));
//         Eigen::Vector2d pj = init_frame_j->apply_k(frame_j_keypoints[i]);
//         cv::Point2d cv_pj = {pj.x(), pj.y()};
//         log_info("[init pair]: i: {}, pi: {}, pj: {}", i, pi.transpose(), pj.transpose());
//         cv::circle(combined, cv_pj + cv::Point2d(0, rows), 5, cv::Scalar(0, 255, 0));
//         cv::line(combined, cv_pi, cv_pj + cv::Point2d(0, rows), cv::Scalar(0, 0, 255));
//     }
//     cv::imshow("init pair", combined);
//     // cv::waitKey(0);

//     std::vector<size_t> init_keyframe_indices;
//     size_t init_map_frames = 8;
//     double keyframe_id_gap = static_cast<double>(map->frame_num() - 1 - init_frame_i_id)
//                              / static_cast<double>(init_map_frames - 1);
//     for (size_t i = 0; i < init_map_frames; ++i) {
//         init_keyframe_indices.push_back((size_t)round(init_frame_i_id + keyframe_id_gap * i));
//         log_info("[init_keyframe_indices]: i: {}->{}", i, init_keyframe_indices.back());
//     }

//     for (size_t i = 0; i < init_keyframe_indices.size(); ++i) {
//         map->append_frame(sw->get_frame(init_keyframe_indices[i])->clone());
//     }

//     for (size_t j = 1; j < init_keyframe_indices.size(); ++j) {
//         Frame *old_frame_i = sw->get_frame(init_keyframe_indices[j - 1]);
//         Frame *old_frame_j = sw->get_frame(init_keyframe_indices[j]);
//         Frame *new_frame_i = map->get_frame(j - 1);
//         Frame *new_frame_j = map->get_frame(j);

//         for (size_t ki = 0; ki < old_frame_i->keypoint_num(); ++ki) {
//             Feature *feature = old_frame_i->get_feature(ki);
//             if (feature == nullptr)
//                 continue;
//             size_t kj = feature->get_observation_index(old_frame_j);
//             if (kj == nil())
//                 continue;
//             new_frame_i->get_feature_if_empty_create(ki)->add_observation(new_frame_j, kj);
//         }

//         new_frame_j->preintegration.data.clear();
//         for (size_t f = init_keyframe_indices[j - 1]; f < init_keyframe_indices[j]; ++f) {
//             Frame *old_frame = sw->get_frame(f + 1);
//             std::vector<IMUData> &old_data = old_frame->preintegration.data;
//             std::vector<IMUData> &new_data = new_frame_j->preintegration.data;
//             new_data.insert(new_data.end(), old_data.begin(), old_data.end());
//         }
//     }

//     log_info("create new map over");
//     Frame *new_init_frame_i = map->get_frame(0);
//     Frame *new_init_frame_j = map->get_last_frame();

//     // [2.3] set init states
//     Pose pose; // 设置第一帧和最后一帧的camera位姿
//     pose.q.setIdentity();
//     pose.p.setZero();
//     new_init_frame_i->set_camera_pose(pose);
//     pose.q = init_R.transpose();
//     pose.p = -(init_R.transpose() * init_T);
//     new_init_frame_j->set_camera_pose(pose);

//     Pose pose_i = new_init_frame_i->get_camera_pose();
//     log_error("pose_i: {} {}", pose_i.q.coeffs().transpose(), pose_i.p.transpose());

//     Pose pose_j = new_init_frame_j->get_camera_pose();
//     log_error("pose_j: {} {}", pose_j.q.coeffs().transpose(), pose_j.p.transpose());

//     int init_points_num = 0;
//     for (size_t k = 0; k < init_points.size();
//          ++k) { // 对于成功三角化的点，设置第一帧作为参考帧，也就是作为世界坐标系，固定不变
//         if (init_point_status[k] == 0)
//             continue;
//         Feature *feature = new_init_frame_i->get_feature(init_matches[k].first);
//         feature->p_in_G = init_points[k]; // 3D点信息与track绑定，就是与每一帧绑定
//         feature->flag(FeatureFlag::FF_VALID) = true;
//         init_points_num++;
//         log_info("[init feature]: i:{} p: {}", init_points_num, feature->p_in_G.transpose());
//     }

//     log_info("[init reprojection error]: {}", init_score);
//     // sw->compute_reprojections();
//     // sw->log_feature_reprojections();

//     // [2.4] solve other frames via pnp    // pnp 通过前一帧的位姿和初始化的3D点来求解后一帧的位姿
//     for (size_t j = 1; j + 1 < map->frame_num();
//          ++j) { // 这个过程不会增加新的3D点，也不会改变初始化3D点的坐标
//         Frame *frame_i = map->get_frame(j - 1);
//         Frame *frame_j = map->get_frame(j);
//         frame_j->set_camera_pose(frame_i->get_camera_pose()); // 后一帧的位姿初始值为前一帧
//         visual_inertial_pnp(map.get(), frame_j, false);       // ceres求解位姿
//     }

//     log_debug("[pnp reprojection error]");
//     map->compute_reprojections();
//     map->log_feature_reprojections();
//     log_debug("[pnp reprojection error over]");


//     // [2.5] triangulate more points
//     for (size_t i = 0; i < map->feature_num();
//          ++i) { // 对map中其他的track也进行三角化，增加更多的3D点
//         Feature *feature = map->get_feature(i);
//         if (feature->flag(FeatureFlag::FF_VALID))
//             continue;           // 如果是初始三角化的点，不可以改变
//         feature->triangulate(); // 给track设置landmark值，并固定不变
//     }

//     // [3] sfm

//     // [3.1] bundle adjustment  // sfm固定第一帧的位姿，固定初始化的3D点
//     map->get_frame(0)->flag(FrameFlag::FF_FIX_POSE) =
//         true; // 利用所有的观测和相机位姿最小化重投影误差估计，优化其中非固定的变量，包括相机位姿和3D点
//     if (!BundleAdjustor().solve(map.get(), false, 50, 1e6)) {
//         return false;
//     }

//     map->log_feature_reprojections();

//     // // [3.2] cleanup invalid points                                   //  从map中清除没有成功三角化的track，或者三角化误差（平均重投影误差）大于1的track
//     map->prune_features([](const Feature *feature) {
//         return !feature->flag(FeatureFlag::FF_VALID) || feature->reprojection_error > 1.0;
//     });

//     log_info("after prune_features");
//     map->log_feature_reprojections();

//     int result_point_num = 0;
//     for (size_t i = 0; i < map->feature_num(); ++i) {
//         if (map->get_feature(i)->triangulate()) {
//             result_point_num++;
//         }
//     }
//     log_info("[init sfm] :  all triangulated ");
//     map->compute_reprojections();
//     map->log_feature_reprojections();

//     return true;
// }

void Initializer::reset_states() {
    bg.setZero();
    ba.setZero();
    gravity.setZero();
    scale = 1;
    velocities.resize(map->frame_num(), Eigen::Vector3d::Zero());
}

void Initializer::preintegrate() {
    for (int i = 1; i < map->frame_num(); ++i) {
        Frame *frame_i = map->get_frame(i);
        // auto &data = frame_i->preintegration.data;
        // for (IMUData &imu : data) {
        //     log_info(
        //         "[Frame ImuData]: fid: {}, ft: {}, t:{},w:{},a:{}", frame_i->id(),
        //         frame_i->image->t, imu.t, imu.w.transpose(), imu.a.transpose());
        // }
        frame_i->preintegration.integrate(frame_i->image->t, bg, ba, true, false);
    }
    log_info("[initializer]: preintegrate map, bg: {}, ba: {}", bg.transpose(), ba.transpose());
}

void Initializer::solve_gyro_bias() {
    log_info("[initializer]: solve_gyro_bias, begin ...");
    preintegrate();
    Eigen::Matrix3d A1 = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b1 = Eigen::Vector3d::Zero();

    Eigen::Matrix3d A2 = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b2 = Eigen::Vector3d::Zero();

    for (int j = 1; j < map->frame_num(); ++j) {
        size_t i = j - 1;
        Frame *frame_i = map->get_frame(i);
        Frame *frame_j = map->get_frame(j);

        Pose pose_i = frame_i->get_imu_pose();
        Pose pose_j = frame_j->get_imu_pose();

        Eigen::Quaterniond dq = frame_j->preintegration.delta.q;
        Eigen::Matrix3d dq_dbg = frame_j->preintegration.jacobian.dq_dbg;

        // log_info("dq imu: \n{}", dq.toRotationMatrix());
        // log_info("dq image: \n{}", (pose_i.q.conjugate() * pose_j.q).toRotationMatrix());

        Eigen::Quaterniond r_q = (pose_i.q * dq).conjugate() * pose_j.q;

        Eigen::Vector3d r = logmap(r_q);
        Eigen::Matrix3d dr_dq = -right_jacobian_inv(r) * r_q.conjugate();

        Eigen::Matrix3d J = dq_dbg;

        A1 += J.transpose() * J; // A1 和 A2 计算出来的bg没有明显差距
        b1 += J.transpose() * r;

        J = dr_dq * dq_dbg;
        A2 += J.transpose() * J;
        b2 += J.transpose() * r;
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd1(A1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d bg1 = svd1.solve(b1);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd2(A2, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d bg2 = -svd2.solve(b2);

    bg = bg2;
    Eigen::Vector3d bg3 = {-0.002229, 0.0207, 0.07635};
    bg3 = {-0.003172, 0.021267, 0.078502};

    double sum_e1 = 0;
    double sum_e2 = 0;

    for (int j = 1; j < map->frame_num(); ++j) {
        size_t i = j - 1;
        Frame *frame_i = map->get_frame(i);
        Frame *frame_j = map->get_frame(j);

        Pose pose_i = frame_i->get_imu_pose();
        Pose pose_j = frame_j->get_imu_pose();

        Eigen::Quaterniond dq = frame_j->preintegration.delta.q;
        Eigen::Matrix3d dq_dbg = frame_j->preintegration.jacobian.dq_dbg;

        Eigen::Quaterniond e1 = (pose_i.q * dq * expmap(dq_dbg * bg1)).conjugate() * pose_j.q;
        // log_info("est bg1: {}", bg1.transpose());
        // log_info("est bg2: {}", bg2.transpose());
        // log_info("error: {},{}", logmap(e1).transpose(), logmap(e1).norm());
        sum_e1 += (logmap(e1)).squaredNorm();

        Eigen::Quaterniond e2 = (pose_i.q * dq * expmap(dq_dbg * bg3)).conjugate() * pose_j.q;
        // log_debug("gt bg: {}", bg3.transpose());
        // log_debug("gt error: {},{}", logmap(e2).transpose(), logmap(e2).norm());
        sum_e2 += (logmap(e2)).squaredNorm();
    }

    log_info("[initializer]: solve_gyro_bias, bg: {}", bg.transpose());
    log_debug("[initializer]: solve_gyro_bias, sume1: {}", sum_e1);
    log_debug("[initializer]: solve_gyro_bias, sume2: {}", sum_e2);
    log_info("[initializer]: solve_gyro_bias, successful");
    // getchar();
}


void Initializer::solve_gravity_scale_velocity() {
    log_info("[initializer]: solve_gravity_scale_velocity, begin ...");

    /*
                g
                s
          A  *  v1  = b
                v2
                .
                .
                .
                vn               

      vj - vi - g∆tij = C(qi)∆vij
      s(pj - pi)-  vi∆tij -0.5 * g∆tij^2 = C(qi)∆pij    
    */
    preintegrate();
    int N = static_cast<int>(map->frame_num());
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    A.resize((N - 1) * 6, 3 + 1 + 3 * N);
    b.resize((N - 1) * 6);
    A.setZero();
    b.setZero();

    Frame *frame0 = map->get_frame(0);
    ExtrinsicParams camera_extri = frame0->camera_extri;

    for (int j = 1; j < map->frame_num(); ++j) {
        size_t i = j - 1;
        Frame *frame_i = map->get_frame(i);
        Frame *frame_j = map->get_frame(j);

        Pose pose_i = frame_i->get_camera_pose();
        Pose pose_j = frame_j->get_camera_pose();

        const PreIntegrator::Delta &delta = frame_j->preintegration.delta;

        A.block<3, 3>(i * 6, 0) = -0.5 * delta.t * delta.t * Eigen::Matrix3d::Identity();
        A.block<3, 1>(i * 6, 3) = pose_j.p - pose_i.p;
        A.block<3, 3>(i * 6, 4 + i * 3) = -delta.t * Eigen::Matrix3d::Identity();
        b.segment<3>(i * 6) = frame_i->pose.q * delta.p + frame_j->pose.q * camera_extri.p
                              - frame_i->pose.q * camera_extri.p;

        A.block<3, 3>(i * 6 + 3, 0) = -delta.t * Eigen::Matrix3d::Identity();
        A.block<3, 3>(i * 6 + 3, 4 + i * 3) = -Eigen::Matrix3d::Identity();
        A.block<3, 3>(i * 6 + 3, 4 + j * 3) = Eigen::Matrix3d::Identity();
        b.segment<3>(i * 6 + 3) = frame_i->pose.q * delta.v;
    }

    Eigen::VectorXd x = A.fullPivHouseholderQr().solve(b);
    gravity = x.segment<3>(0).normalized() * GRAVITY_NORM;
    scale = x(3);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(4 + 3 * i);
    }

    log_info("[initializer]: solve_gravity_scale_velocity, gravity: {}", gravity.transpose());
    log_info("[initializer]: solve_gravity_scale_velocity, scale: {}", scale);
    log_info("[initializer]: solve_gravity_scale_velocity, successful");
}

void Initializer::refine_scale_velocity_via_gravity() {
    log_info("[initializer]: refine_scale_velocity_via_gravity, begin ...");
    /*
                dg1
                dg2
                s
          A  *  v1  = b
                v2
                .
                .
                .
                vn               

      vj - vi - g∆tij = C(qi)∆vij
      s(pj - pi)-  vi∆tij -0.5 * g∆tij^2 = C(qi)∆pij    
    */

    static const double damp = 0.1;
    preintegrate();
    int N = static_cast<int>(map->frame_num());
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd x;
    A.resize((N - 1) * 6, 2 + 1 + 3 * N);
    b.resize((N - 1) * 6);
    x.resize(2 + 1 + 3 * N);

    Frame *frame0 = map->get_frame(0);
    ExtrinsicParams camera_extri = frame0->camera_extri;

    for (size_t iter = 0; iter < 4; ++iter) {
        A.setZero();
        b.setZero();
        Eigen::Matrix<double, 3, 2> Tg = s2_tangential_basis(gravity);
        for (int j = 1; j < map->frame_num(); ++j) {
            size_t i = j - 1;
            Frame *frame_i = map->get_frame(i);
            Frame *frame_j = map->get_frame(j);

            Pose pose_i = frame_i->get_camera_pose();
            Pose pose_j = frame_j->get_camera_pose();

            const PreIntegrator::Delta &delta = frame_j->preintegration.delta;

            A.block<3, 2>(i * 6, 0) = -0.5 * delta.t * delta.t * Tg;
            A.block<3, 1>(i * 6, 2) = pose_j.p - pose_i.p;
            A.block<3, 3>(i * 6, 3 + i * 3) = -delta.t * Eigen::Matrix3d::Identity();
            b.segment<3>(i * 6) = frame_i->pose.q * delta.p + frame_j->pose.q * camera_extri.p
                                  - frame_i->pose.q * camera_extri.p
                                  + 0.5 * delta.t * delta.t * gravity;

            A.block<3, 2>(i * 6 + 3, 0) = -delta.t * Tg;
            A.block<3, 3>(i * 6 + 3, 3 + i * 3) = -Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6 + 3, 3 + j * 3) = Eigen::Matrix3d::Identity();
            b.segment<3>(i * 6 + 3) = frame_i->pose.q * delta.v + delta.t * gravity;
        }

        x = A.fullPivHouseholderQr().solve(b);
        Eigen::Vector2d dg = x.segment<2>(0);
        gravity = (gravity + damp * Tg * dg).normalized() * GRAVITY_NORM;
    }

    scale = x(2);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(3 + 3 * i);
    }
    log_info("[initializer]: refine_scale_velocity_via_gravity, gravity: {}", gravity.transpose());
    log_info("[initializer]: refine_scale_velocity_via_gravity, scale: {}", scale);
    log_info("[initializer]: refine_scale_velocity_via_gravity, successful");
}


bool Initializer::apply_init() {
    log_info("[initializer]: apply_init, begin ...");
    {
        double error = map->compute_reprojections();
        log_info("[initializer]: apply_init, before reprojection error: {}", error);
        // map->log_feature_reprojections();
    }

    const Eigen::Vector3d gravity_world {0, 0, -GRAVITY_NORM};
    Frame *frame0 = map->get_frame(0);
    Pose camera_pose0 = frame0->get_camera_pose();
    ExtrinsicParams camera_extri = frame0->camera_extri;

    Pose imu_pose0;
    imu_pose0.q = camera_pose0.q * camera_extri.q.conjugate();
    imu_pose0.p = (scale * camera_pose0.p - imu_pose0.q * camera_extri.p);
    // Eigen::Vector3d g_in_b0 = frame_world->camera_extri.q * gravity;
    Eigen::Matrix3d R0 = g2R(gravity);
    double yaw = R2ypr(R0 * imu_pose0.q).x();
    R0 = ypr2R(Eigen::Vector3d {-yaw, 0, 0}) * R0;
    gravity = R0 * gravity;
    log_info("[initializer]: apply_init, gravity: {}", gravity.transpose());

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        Pose camera_pose = frame->get_camera_pose();
        Pose imu_pose;
        imu_pose.q = camera_pose.q * camera_extri.q.conjugate();
        imu_pose.p = (scale * camera_pose.p - imu_pose.q * camera_extri.p) - imu_pose0.p;
        imu_pose.q = R0 * imu_pose.q;
        imu_pose.p = R0 * imu_pose.p;
        frame->set_imu_pose(imu_pose);
        frame->motion.v = R0 * velocities[i];
        frame->motion.bg = bg;
        frame->motion.ba.setZero();
    }

    // for (size_t i = 0; i < map->feature_num(); ++i) {
    //     Feature *feature = map->get_feature(i);
    //     feature->p_in_G = scale * (R0 * feature->p_in_G);
    // }

    // log_info("[apply init] : set new pose ");
    // map->compute_reprojections();
    // map->log_feature_reprojections();

    int result_point_num = 0;
    for (size_t i = 0; i < map->feature_num(); ++i) {
        if (map->get_feature(i)->triangulate()) {
            result_point_num++;
        }
    }

    {
        double error = map->compute_reprojections();
        log_info("[initializer]: apply_init, after reprojection error: {}", error);
        // map->log_feature_reprojections();
    }

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        log_debug(
            "[initializer]: apply_init, frame {}, imu pose q: {}", frame->id(),
            frame->pose.q.coeffs().transpose());
        log_debug(
            "[initializer]: apply_init, frame {}, imu pose q ypr: {}", frame->id(),
            R2ypr(frame->pose.q.toRotationMatrix()).transpose());
        log_debug(
            "[initializer]: apply_init, frame {}, imu pose p: {}", frame->id(),
            frame->pose.p.transpose());
    }
    log_info("[initializer]: apply_init, over");
    return result_point_num >= 30;
}

bool Initializer::init_imu() {
    log_info("[initializer]: init_imu begin ...");
    reset_states();
    solve_gyro_bias();
    solve_gravity_scale_velocity();
    if (scale < 0.001 || scale > 1.0) {
        log_error("[initializer]: scale is not valid, scale: {}", scale);
        return false;
    }
    refine_scale_velocity_via_gravity();
    if (scale < 0.001 || scale > 1.0) {
        log_error("[initializer]: scale is not valid, scale: {}", scale);
        return false;
    }
    bool result = apply_init();
    log_info("[initializer]: init_imu successful");
    return result;
}

std::unique_ptr<SlidingWindowTracker> Initializer::init() {
    if (!map)
        return nullptr;
    if (!init_sfm())
        return nullptr;
    if (!init_imu())
        return nullptr;

    map->get_frame(0)->flag(FrameFlag::FF_FIX_POSE) = true;
    BundleAdjustor().solve(map.get(), true, 50, 1e6);

    for (size_t i = 0; i < map->frame_num(); ++i) {
        map->get_frame(i)->flag(FrameFlag::FF_KEYFRAME) = true;
    }

    {
        double error = map->compute_reprojections();
        log_info("[initializer]: sfm-imu initialized, reprojection error: {}", error);
        // map->log_feature_reprojections();
    }

    map->prune_features([](const Feature *feature) {
        return !feature->flag(FeatureFlag::FF_VALID) || feature->reprojection_error > 1.0;
    });

    {
        double error = map->compute_reprojections();
        log_info("[initializer]: sfm-imu initialized, reprojection error: {}", error);
        // map->log_feature_reprojections();
    }

    Eigen::Vector3d gt_bg = {-0.003172, 0.021267, 0.078502};
    Eigen::Vector3d gt_ba = {-0.025266, 0.136696, 0.075593};
    log_info("[initializer]: initialized bg : {} ", map->get_frame(0)->motion.bg.transpose());
    // log_info("gt bg: {} ", gt_bg.transpose());
    log_info("[initializer]: initialized ba : {} ", map->get_frame(0)->motion.ba.transpose());
    // log_info("gt ba: {} ", gt_ba.transpose());

    std::unique_ptr<SlidingWindowTracker> sw_tracker =
        std::make_unique<SlidingWindowTracker>(std::move(map));

    log_info("[initializer]: sfm-imu initialized successful");

    return sw_tracker;
}
