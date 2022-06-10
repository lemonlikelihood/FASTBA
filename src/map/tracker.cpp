#include "tracker.h"
#include "grider_fast.h"

void KLTTracker::detect_keypoints(
    const std::vector<cv::Mat> &imgpyr, std::vector<cv::KeyPoint> &pts) {
    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    Eigen::MatrixXi grid_2d_current = Eigen::MatrixXi::Zero(
        (int)(imgpyr.at(0).cols / min_px_dist) + 10, (int)(imgpyr.at(0).rows / min_px_dist) + 10);
    auto it0 = pts.begin();
    // auto it2 = ids.begin();
    while (it0 != pts.end()) {
        // Get current left keypoint
        cv::KeyPoint kpt = *it0;
        // Check if this keypoint is near another point
        if (grid_2d_current((int)(kpt.pt.x / min_px_dist), (int)(kpt.pt.y / min_px_dist))
            == 1) { // 检测每个最小的grid中是否有特征点，保证相邻的两个特征点之间的距离大于min_px_dist
            it0 = pts.erase(it0); // pts和ids已经剔除相距太近的冗余特征点
            // it2 = ids.erase(it2);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_current((int)(kpt.pt.x / min_px_dist), (int)(kpt.pt.y / min_px_dist)) = 1;
        it0++;
        // it2++;
    }

    // First compute how many more features we need to extract from this image
    int num_featsneeded = num_features - (int)pts.size();

    // If we don't need any features, just return
    if (num_featsneeded < 1)
        return;

    // Extract our features (use fast with griding)
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_FAST::perform_griding(
        imgpyr.at(0), pts0_ext, num_featsneeded, grid_x, grid_y, threshold,
        true); // 在每一个grid中检测fast角点

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    Eigen::MatrixXi grid_2d = Eigen::MatrixXi::Zero(
        (int)(imgpyr.at(0).cols / min_px_dist) + 10, (int)(imgpyr.at(0).rows / min_px_dist) + 10);
    for (auto &kpt : pts) {
        grid_2d((int)(kpt.pt.x / min_px_dist), (int)(kpt.pt.y / min_px_dist)) =
            1; // 对原来的特征点设置occupancy_grid
    }

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
        // See if there is a point at this location
        if (grid_2d((int)(kpt.pt.x / min_px_dist), (int)(kpt.pt.y / min_px_dist))
            == 1) // 将新检测到的特征点和原来的特征点融合到同一个occupancy_grid 中
            continue;
        // Else lets add it!
        kpts0_new.push_back(
            kpt); // 如果新加入的特征点和已存在的特征点以及当前的特征点都没有冲突的话，直接加入
        pts0_new.push_back(kpt.pt);
        grid_2d((int)(kpt.pt.x / min_px_dist), (int)(kpt.pt.y / min_px_dist)) = 1;
    }

    // Loop through and record only ones that are valid
    for (size_t i = 0; i < pts0_new.size(); i++) {
        // update the uv coordinates
        kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts.push_back(kpts0_new.at(i));
        // move id foward and append this new point
        size_t temp = ++currid; // 给新检测到的特征点取一个id,并融合到原来的特征点中
        ids.push_back(temp);
    }
}


bool KLTTracker::track_monocular(Frame *frame) {
    // 1. 对收到的图片首先做一个直方图均衡化
    cv::Mat img = frame->m_image;
    cv::equalizeHist(img, img);

    // 2. 对均衡化后的图像提取金字塔（按传入的窗口大小和金字塔层数来提取）
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

    if (pts_last.empty()) {
        detect_keypoints(imgpyr, pts_last);
    }
}