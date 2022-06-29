#include "../src/utils/common.h"
#include "../src/utils/debug.h"
#include "../src/utils/forensics.h"

int main() {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    int a = 3;
    // log_debug("{}\n", R);
    log_info("{}", a);
    // log_trace("{}\n", R);
    Pose pose;
    pose.p = Eigen::Vector3d(1, 2, 3);
    pose.q = Eigen::Quaterniond(0.2, 0.3, 0.4, 0.5);

    forensics(map_info, info) {
        output_map_info data;
        data.poses.push_back(pose);
        data.poses.push_back(pose);
        info = std::move(data);
    }

    std::vector<Pose> pose_vec;
    forensics(map_info, info) {
        if (info.has_value()) {
            const output_map_info &painter = std::any_cast<output_map_info>(info);
            for (int i = 0; i < painter.poses.size(); i++) {
                pose_vec.push_back(painter.poses[i]);
            }
        }
    }

    for (int i = 0; i < pose_vec.size(); i++) {
        log_debug("{}: {}", i, pose_vec[i].q.coeffs().transpose());
    }

    return 0;
}