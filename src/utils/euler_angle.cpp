#include "euler_angle.h"

//  -180<y<180, -90<p<<90, 180<r<180  vins-fusion method
Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R) {
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

Eigen::Matrix3d ypr2R(const Eigen::Vector3d &ypr) {

    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

    Eigen::Matrix3d Ry;
    Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

    Eigen::Matrix3d Rx;
    Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

Eigen::Matrix3d g2R(const Eigen::Vector3d &g) {
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2 {0, 0, -1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = R2ypr(R0).x();
    R0 = ypr2R(Eigen::Vector3d {-yaw, 0, 0}) * R0;
    return R0;
}

Eigen::Matrix3d Rxy(const Eigen::Matrix3d &R) {
    Eigen::Vector3d gw {0, 0, -1.0};
    Eigen::Vector3d gc = R.inverse() * gw;
    return g2R(gc);
}

Eigen::Matrix3d Rz(const Eigen::Matrix3d &R) {
    return R * Rxy(R).inverse();
}

Eigen::Vector2d get_gravity_direction(const Eigen::Matrix3d &R) {
    Eigen::Vector3d ypr = R2ypr(Rxy(R));
    return Eigen::Vector2d {ypr[1], ypr[2]};
}

// void compute_yaw_gravity(const std::vector<PoseData> &pose_vec, std::vector<YprData> &ypr_vec) {
//     ypr_vec.clear();
//     Eigen::Matrix3d pose_init_R = pose_vec[0].q.matrix();

//     Eigen::Matrix3d Rz_init = Rz(pose_init_R);

//     Eigen::AngleAxisd AA_Rz_init(Rz_init);
//     if (AA_Rz_init.axis().z() < 0) {
//         AA_Rz_init.axis() *= -1.0;
//         AA_Rz_init.angle() *= -1.0;
//     }

//     for (int i = 0; i < pose_vec.size(); i++) {
//         Eigen::Matrix3d pose_curr_R = pose_vec[i].q.matrix();
//         Eigen::Matrix3d Rz_curr = Rz(pose_curr_R);

//         Eigen::AngleAxisd AA_Rz_curr(Rz_curr);
//         if (AA_Rz_curr.axis().z() < 0) {
//             AA_Rz_curr.axis() *= -1.0;
//             AA_Rz_curr.angle() *= -1.0;
//         }

//         Eigen::Vector2d gravity_direction = get_gravity_direction(pose_curr_R);
//         YprData tmp;
//         tmp.t = pose_vec[i].t;
//         tmp.ypr[0] = (AA_Rz_curr.angle() - AA_Rz_init.angle()) * 180 / M_PI;
//         tmp.ypr[1] = gravity_direction[0];
//         tmp.ypr[2] = gravity_direction[1];
//         ypr_vec.push_back(tmp);
//     }
// }

// void compute_euler(const std::vector<PoseData> &pose_vec, std::vector<YprData> &ypr_vec) {
//     ypr_vec.clear();
//     for (int i = 0; i < pose_vec.size(); i++) {
//         YprData tmp;
//         tmp.t = pose_vec[i].t;
//         tmp.ypr = R2ypr(pose_vec[i].q.toRotationMatrix());
//         ypr_vec.push_back(tmp);
//     }
// }
