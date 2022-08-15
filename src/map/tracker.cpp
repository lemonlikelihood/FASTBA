#include "tracker.h"
#include "../utils/common.h"
#include "feature.h"
#include "frame.h"
#include "grider_fast.h"

TrackerBase::TrackerBase() : m_num_features(200) {}

KLTTracker::KLTTracker()
    : TrackerBase(), m_threshold(10), m_grid_x(8), m_grid_y(5), m_min_px_dist(30) {}

void KLTTracker::detect_keypoints(
    const std::vector<cv::Mat> &imgpyr, std::vector<cv::KeyPoint> &pts) {
    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    Eigen::MatrixXi grid_2d_current = Eigen::MatrixXi::Zero(
        static_cast<int>(imgpyr.at(0).cols / m_min_px_dist) + 10,
        static_cast<int>(imgpyr.at(0).rows / m_min_px_dist) + 10);
    auto it_pts = pts.begin();
    while (it_pts != pts.end()) {
        // Get current left keypoint
        cv::KeyPoint kpt = *it_pts;
        // Check if this keypoint is near another point
        if (grid_2d_current(
                static_cast<int>(kpt.pt.x / m_min_px_dist),
                static_cast<int>(kpt.pt.y / m_min_px_dist))
            == 1) { // 检测每个最小的grid中是否有特征点，保证相邻的两个特征点之间的距离大于min_px_dist
            it_pts = pts.erase(it_pts); // pts和ids已经剔除相距太近的冗余特征点
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_current(
            static_cast<int>(kpt.pt.x / m_min_px_dist),
            static_cast<int>(kpt.pt.y / m_min_px_dist)) = 1;
        it_pts++;
    }

    // // First compute how many more features we need to extract from this image
    int num_featsneeded = m_num_features - static_cast<int>(pts.size());
    log_info("[tracker] num_features: {}", m_num_features);
    log_info("[tracker] pts.size(): {}", pts.size());
    log_info("[tracker] num_featsneeded: {}", num_featsneeded);

    // // If we don't need any features, just return
    if (num_featsneeded < 1)
        return;

    // Extract our features (use fast with griding)
    std::vector<cv::KeyPoint> pts_ext;
    Grider_FAST::perform_griding(
        imgpyr.at(0), pts_ext, num_featsneeded, m_grid_x, m_grid_y, m_threshold,
        true); // 在每一个grid中检测fast角点

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    Eigen::MatrixXi grid_2d = Eigen::MatrixXi::Zero(
        static_cast<int>(imgpyr.at(0).cols / m_min_px_dist) + 10,
        static_cast<int>(imgpyr.at(0).rows / m_min_px_dist) + 10);
    for (auto &kpt : pts) {
        grid_2d(
            static_cast<int>(kpt.pt.x / m_min_px_dist),
            static_cast<int>(kpt.pt.y / m_min_px_dist)) = 1; // 对原来的特征点设置occupancy_grid
    }

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts_new;
    std::vector<cv::Point2f> pts_new;
    for (auto &kpt : pts_ext) {
        // See if there is a point at this location
        if (grid_2d(
                static_cast<int>(kpt.pt.x / m_min_px_dist),
                static_cast<int>(kpt.pt.y / m_min_px_dist))
            == 1) // 将新检测到的特征点和原来的特征点融合到同一个occupancy_grid 中
            continue;
        // Else lets add it!
        kpts_new.push_back(
            kpt); // 如果新加入的特征点和已存在的特征点以及当前的特征点都没有冲突的话，直接加入
        pts_new.push_back(kpt.pt);
        grid_2d(
            static_cast<int>(kpt.pt.x / m_min_px_dist),
            static_cast<int>(kpt.pt.y / m_min_px_dist)) = 1;
    }

    // Loop through and record only ones that are valid
    for (size_t i = 0; i < pts_new.size(); i++) {
        // update the uv coordinates
        kpts_new.at(i).pt = pts_new.at(i);
        // append the new uv coordinate
        pts.push_back(kpts_new.at(i));
    }
}


bool KLTTracker::track_monocular(Frame *frame) {
    // 1. 对收到的图片已经去畸变和直方图均衡化
    cv::Mat &img = frame->image->image;

    // 2. 对均衡化后的图像提取金字塔（按传入的窗口大小和金字塔层数来提取）
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, m_win_size, m_pyr_levels);

    if (m_pts_last.empty()) {
        detect_keypoints(imgpyr, m_pts_last);
        m_img_pyramid_last = imgpyr;
        m_last_frame = frame;
        return true;
    }

    detect_keypoints(
        m_img_pyramid_last, m_pts_last); // m_pts_last 先对上一帧图像提取新的特征点，再跟踪
    // Our return success masks, and predicted new features

    m_last_frame->features.resize(m_pts_last.size(), nullptr);
    m_last_frame->keypoints.resize(m_pts_last.size());
    m_last_frame->keypoints_normalized.resize(m_pts_last.size());
    m_last_frame->reprojection_factors.resize(m_pts_last.size());
    for (int i = 0; i < m_pts_last.size(); i++) {
        Eigen::Vector2d keypoint {m_pts_last[i].pt.x, m_pts_last[i].pt.y};
        m_last_frame->keypoints[i] = keypoint;
        m_last_frame->keypoints_normalized[i] = m_last_frame->remove_k(keypoint);
    }

    std::vector<uchar> mask_ll;
    std::vector<cv::KeyPoint> pts_left_new = m_pts_last;

    // Lets track temporally         // 上一帧金字塔  // 当前帧金字塔 // 上一帧关键点(原图) // 当前帧关键点(原图)
    track_keypoints(
        m_img_pyramid_last, imgpyr, m_pts_last, pts_left_new,
        mask_ll); // mask_ll 表明track是否成功

    log_info("track success");

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty()) {
        m_img_pyramid_last = imgpyr;
        m_last_frame = frame;
        m_pts_last.clear();
        log_error("[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....");
        return false;
    }

    std::vector<cv::KeyPoint> good_tracked_keypoints;

    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++) {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_left_new[i].pt.x < 0 || pts_left_new[i].pt.y < 0)
            continue;
        // If it is a good track, and also tracked from left to right
        if (mask_ll[i]) {
            size_t next_keypoint_id = frame->keypoints.size();
            good_tracked_keypoints.push_back(pts_left_new[i]);
            Eigen::Vector2d keypoint {pts_left_new[i].pt.x, pts_left_new[i].pt.y};
            frame->keypoints.emplace_back(keypoint);
            frame->keypoints_normalized.emplace_back(frame->remove_k(keypoint));
            frame->features.emplace_back(nullptr);
            frame->reprojection_factors.emplace_back(nullptr);
            m_last_frame->get_feature_if_empty_create(i)->add_observation(frame, next_keypoint_id);
        }
    }
    auto frame_i = m_last_frame;
    auto frame_j = frame;
    const int32_t rows = frame_i->image->image.rows;
    const int32_t cols = frame_i->image->image.cols;
    cv::Mat img1 = frame_i->image->image;
    cv::Mat img2 = frame_j->image->image;
    cv::Mat combined(rows * 2, cols, CV_8UC1);
    img1.copyTo(combined.rowRange(0, rows));
    img2.copyTo(combined.rowRange(rows, rows * 2));
    cv::cvtColor(combined, combined, cv::COLOR_GRAY2RGBA);

    std::vector<Eigen::Vector2d> frame_i_keypoints;
    std::vector<Eigen::Vector2d> frame_j_keypoints;

    frame_i_keypoints.clear();
    frame_j_keypoints.clear();

    for (size_t ki = 0; ki < frame_i->keypoint_num(); ++ki) {
        Feature *feature = frame_i->get_feature(ki);
        if (!feature)
            continue;
        size_t kj = feature->get_observation_index(frame_j);
        if (kj == nil())
            continue;
        frame_i_keypoints.push_back(frame_i->get_keypoint_normalized(ki));
        frame_j_keypoints.push_back(frame_j->get_keypoint_normalized(kj));
    }

    for (int i = 0; i < frame_i_keypoints.size(); i++) {
        Eigen::Vector2d pi = frame_i->apply_k(frame_i_keypoints[i]);
        cv::Point2d cv_pi = {pi.x(), pi.y()};
        cv::circle(combined, cv_pi, 5, cv::Scalar(255, 0, 0));
        Eigen::Vector2d pj = frame_j->apply_k(frame_j_keypoints[i]);
        cv::Point2d cv_pj = {pj.x(), pj.y()};
        log_info("i: {}, pi: {}, pj: {}", i, pi.transpose(), pj.transpose());
        cv::circle(combined, cv_pj + cv::Point2d(0, rows), 5, cv::Scalar(0, 255, 0));
        cv::line(combined, cv_pi, cv_pj + cv::Point2d(0, rows), cv::Scalar(0, 0, 255));
    }
    cv::imshow("track combined", combined);
    // cv::waitKey(0);

    if (true) {
        auto frame_i = frame->sw->get_frame(0);
        auto frame_j = frame;
        const int32_t rows = frame_i->image->image.rows;
        const int32_t cols = frame_i->image->image.cols;
        cv::Mat img1 = frame_i->image->image;
        cv::Mat img2 = frame_j->image->image;
        cv::Mat combined(rows * 2, cols, CV_8UC1);
        img1.copyTo(combined.rowRange(0, rows));
        img2.copyTo(combined.rowRange(rows, rows * 2));
        cv::cvtColor(combined, combined, cv::COLOR_GRAY2RGBA);

        std::vector<Eigen::Vector2d> frame_i_keypoints;
        std::vector<Eigen::Vector2d> frame_j_keypoints;

        frame_i_keypoints.clear();
        frame_j_keypoints.clear();

        for (size_t ki = 0; ki < frame_i->keypoint_num(); ++ki) {
            Feature *feature = frame_i->get_feature(ki);
            if (!feature)
                continue;
            size_t kj = feature->get_observation_index(frame_j);
            if (kj == nil())
                continue;
            frame_i_keypoints.push_back(frame_i->get_keypoint_normalized(ki));
            frame_j_keypoints.push_back(frame_j->get_keypoint_normalized(kj));
        }

        for (int i = 0; i < frame_i_keypoints.size(); i++) {
            Eigen::Vector2d pi = frame_i->apply_k(frame_i_keypoints[i]);
            cv::Point2d cv_pi = {pi.x(), pi.y()};
            cv::circle(combined, cv_pi, 5, cv::Scalar(255, 0, 0));
            Eigen::Vector2d pj = frame_j->apply_k(frame_j_keypoints[i]);
            cv::Point2d cv_pj = {pj.x(), pj.y()};
            log_info("i: {}, pi: {}, pj: {}", i, pi.transpose(), pj.transpose());
            cv::circle(combined, cv_pj + cv::Point2d(0, rows), 5, cv::Scalar(0, 255, 0));
            cv::line(combined, cv_pi, cv_pj + cv::Point2d(0, rows), cv::Scalar(0, 0, 255));
        }
        cv::imshow("track init combined", combined);
        // cv::waitKey(0);
    }

    m_img_pyramid_last = imgpyr;
    m_pts_last = good_tracked_keypoints;
    m_last_frame = frame;

    return true;
}

void KLTTracker::track_keypoints(
    const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr,
    std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1,
    std::vector<uchar> &mask_out) {

    assert(kpts0.size() == kpts1.size());

    // Return if we don't have any points
    if (kpts0.empty() || kpts1.empty())
        return;

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::Point2f> pts0, pts1;
    for (size_t i = 0; i < kpts0.size(); i++) {
        pts0.push_back(kpts0.at(i).pt);
        pts1.push_back(kpts1.at(i).pt);
    }

    // If we don't have enough points for ransac just return empty
    // We set the mask to be all zeros since all points failed RANSAC
    if (pts0.size() < 10) {
        for (size_t i = 0; i < pts0.size(); i++)
            mask_out.push_back((uchar)0);
        return;
    }

    // Now do KLT tracking to get the valid new points
    std::vector<uchar> mask_klt;
    std::vector<float> error;
    cv::TermCriteria term_crit =
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01);
    cv::calcOpticalFlowPyrLK(
        img0pyr, img1pyr, pts0, pts1, mask_klt, error, m_win_size, m_pyr_levels, term_crit,
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // 对原始图像上的特征点进行光流跟踪，pts1现在变成光流跟踪后的坐标

    // // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    cv::findFundamentalMat(
        pts0, pts1, cv::FM_RANSAC, 1.0, 0.999,
        mask_rsc); // 通过OpenCV FindFundamentalMat Ransac来剔除噪声点，输入是去畸变后的归一化平面坐标

    // // Loop through and record only ones that are valid
    for (size_t i = 0; i < mask_klt.size(); i++) {
        auto mask =
            (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0); // 如果光流跟踪和ransac都是内点才认为该特征点成功匹配
        mask_out.push_back(mask);
    }

    // // Copy back the updated positions
    for (size_t i = 0; i < pts0.size(); i++) { // 返回成功匹配的特征点
        kpts0.at(i).pt = pts0.at(i);
        kpts1.at(i).pt = pts1.at(i);
    }
}
