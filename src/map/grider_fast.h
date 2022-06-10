#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

/**
     * @brief Extracts FAST features in a grid pattern.
     *
     * As compared to just extracting fast features over the entire image,
     * we want to have as uniform of extractions as possible over the image plane.
     * Thus we split the image into a bunch of small grids, and extract points in each.
     * We then pick enough top points in each grid so that we have the total number of desired points.
     */
class Grider_FAST {

public:
    /**
         * @brief Compare keypoints based on their response value.
         * @param first First keypoint
         * @param second Second keypoint
         *
         * We want to have the keypoints with the highest values!
         * See: https://stackoverflow.com/a/10910921
         */
    static bool compare_response(cv::KeyPoint first, cv::KeyPoint second) {
        return first.response > second.response;
    }


    /**
         * @brief This function will perform grid extraction using FAST.
         * @param img Image we will do FAST extraction on
         * @param pts vector of extracted points we will return
         * @param num_features max number of features we want to extract
         * @param grid_x size of grid in the x-direction / u-direction
         * @param grid_y size of grid in the y-direction / v-direction
         * @param threshold FAST threshold paramter (10 is a good value normally)
         * @param nonmaxSuppression if FAST should perform non-max suppression (true normally)
         *
         * Given a specified grid size, this will try to extract fast features from each grid.
         * It will then return the best from each grid in the return vector.
         */

    // 在 grid_x 和 grid_y 大小的网格中 ，在每一个网格中提取特定数量的特征点
    static void perform_griding(
        const cv::Mat &img, std::vector<cv::KeyPoint> &pts, int num_features, int grid_x,
        int grid_y, int threshold, bool nonmaxSuppression) {

        // Calculate the size our extraction boxes should be
        int size_x = img.cols / grid_x;
        int size_y = img.rows / grid_y;

        // Make sure our sizes are not zero
        assert(size_x > 0);
        assert(size_y > 0);

        // We want to have equally distributed features
        auto num_features_grid =
            (int)(num_features / (grid_x * grid_y))
            + 1; // 将所有的特征点平均分配到所有网格中，每一个grid中至少需要的特征点数量

        // Parallelize our 2d grid extraction!!
        int ct_cols = std::floor(img.cols / size_x); // 实际的列数
        int ct_rows = std::floor(img.rows / size_y); // 实际的行数
        std::vector<std::vector<cv::KeyPoint>> collection(
            ct_cols * ct_rows); // 每一个网格里面都有一个vector
        parallel_for_(cv::Range(0, ct_cols * ct_rows), [&](const cv::Range &range) {
            for (int r = range.start; r < range.end; r++) {

                // Calculate what cell xy value we are in
                int x = r % ct_cols * size_x; // 第r个网格所对应的x坐标
                int y = r / ct_cols * size_y; // 第r个网格所对应的y坐标

                // Skip if we are out of bounds
                if (x + size_x > img.cols || y + size_y > img.rows) // 判断是否越界
                    continue;

                // Calculate where we should be extracting from
                cv::Rect img_roi = cv::Rect(x, y, size_x, size_y); // 裁剪图片

                // Extract FAST features for this part of the image
                std::vector<cv::KeyPoint> pts_new; //
                cv::FAST(
                    img(img_roi), pts_new, threshold, nonmaxSuppression); // 调用opencv提取fast角点

                // Now lets get the top number from this
                std::sort(
                    pts_new.begin(), pts_new.end(),
                    Grider_FAST::compare_response); // 按照响应值进行特征点排序

                // Append the "best" ones to our vector
                // Note that we need to "correct" the point u,v since we extracted it in a ROI
                // So we should append the location of that ROI in the image
                for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size();
                     i++) { // 按指定个数的特征点加入vector中
                    cv::KeyPoint pt_cor = pts_new.at(i);
                    pt_cor.pt.x += x;
                    pt_cor.pt.y += y;
                    collection.at(r).push_back(pt_cor);
                }
            }
        });

        // Combine all the collections into our single vector
        for (size_t r = 0; r < collection.size(); r++) {
            pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
        }
    }
};