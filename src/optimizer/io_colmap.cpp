//
// Created by lemon on 2020/9/22.
//

#include "io_colmap.h"

namespace colmap {
    void read_cameras(const std::string &cameras_file, Map &map) {
        std::ifstream f(cameras_file);
        if (!f.is_open()) {
            std::cout << "camera.txt is not existed!";
            return;
        }
        std::cout << "Loading " << cameras_file << " ..." << std::endl;
        std::string s;

        int line_num = 0;
        while (getline(f, s)) {
            if (line_num > 2) {
                std::stringstream ss;
                CameraModel camera;
                ss << s;
//            std::cout<<"line_num: s"<<line_num<<" "<<s<<std::endl;
                ss >> camera.id >> camera.camera_model >> camera.width >> camera.height;
                ss >> camera.camera_params[0] >> camera.camera_params[1] >> camera.camera_params[2]
                   >> camera.camera_params[3];
                map.m_camera_models[camera.id] = camera;
//            std::cout << camera.id << std::endl;
            }
            line_num++;
        }
    }

    void read_images(const std::string &images_file, Map &map) {
        std::ifstream f(images_file);
        if (!f.is_open()) {
            std::cout << "images.txt is not existed!";
            return;
        }
        std::cout << "Loading " << images_file << " ..." << std::endl;
        std::string s;

//    Eigen::Quaterniond qcw;
//    Eigen::Vector3d tcw;
        Eigen::Vector2d keypoint;
        int point3d_index = 0;
        int line_num = 0;
        while (getline(f, s)) {
            if (line_num > 3) {
                std::stringstream ss;
                Frame frame;
                ss << s;
                ss >> frame.id;
                ss >> frame.qcw.w() >> frame.qcw.x() >> frame.qcw.y() >> frame.qcw.z();
                ss >> frame.tcw.x() >> frame.tcw.y() >> frame.tcw.z();
                ss >> frame.camera_id;
                ss >> frame.image_name;
                getline(f, s);
                ss.clear();
                ss << s;
                while (ss) {
                    ss >> keypoint(0) >> keypoint(1);
                    ss >> point3d_index;
                    frame.keypoints.emplace_back(keypoint);
                    frame.track_ids.emplace_back(point3d_index);
                }
                map.m_frames[frame.id] = frame;
            }
            line_num++;
        }
    }

    void read_points3D(const std::string &points3D_file, Map &map) {
        std::ifstream f(points3D_file);
        if (!f.is_open()) {
            std::cout << "points3D.txt is not existed!";
            return;
        }
        std::cout << "Loading " << points3D_file << " ..." << std::endl;
        std::string s;

        int image_id;
        int point2d_index = 0;
        int line_num = 0;
        while (getline(f, s)) {
            if (line_num > 2) {
                std::stringstream ss;
                Track track;
                ss << s;
                ss >> track.id;
                ss >> track.point3D.x() >> track.point3D.y() >> track.point3D.z();
                ss >> track.color[0] >> track.color[1] >> track.color[2];
                ss >> track.error;
                while (ss) {
                    ss >> image_id;
                    ss >> point2d_index;
                    track.observations[image_id] = point2d_index;
                }
                map.m_tracks[track.id] = track;
//            std::cout<<track.id<<std::endl;
            }
            line_num++;
        }
//    std::cout<<line_num<<std::endl;
    }

    void read_map(std::string &path, Map &map) {
        std::cout << "Loading Map..." << std::endl;

        TicToc ticToc;
        read_cameras(path + "/cameras.txt", map);
        double t1 = ticToc.toc();

        ticToc.tic();
        read_images(path + "/images.txt", map);
        double t2 = ticToc.toc();

        ticToc.tic();
        read_points3D(path + "/points3D.txt", map);
        double t3 = ticToc.toc();

        for (auto &f:map.m_frames) {
            map.m_image_name_to_camera_ids[f.second.image_name] = f.first;
            f.second.camera_model = map.m_camera_models[f.first];
        }

        for (auto &f:map.m_frames) {
            const int id = f.first;   // f_id
            auto &frame = f.second;   // frame
            for (int i = 0; i < frame.track_ids.size(); i++) {
                auto &track_id = frame.track_ids[i];
                if (track_id == -1)continue;
                if (map.m_tracks[track_id].observations[id] != i) {
                    frame.track_ids[i] = -1;
                }
            }
        }

//    std::cout << "j:" << j << std::endl;

        std::cout << "camera model number: " << map.m_camera_models.size() << std::endl;
        std::cout << "Frame number: " << map.m_frames.size() << std::endl;
        std::cout << "Point number: " << map.m_tracks.size() << std::endl;
//    std::cout << "observation number: " << .size() << std::endl;
        std::cout << "Done!" << std::endl;
    }

    void associate_OBS(Map &map) {
        std::cout << "associate_OBS Begin!" << std::endl;
        for (auto &it:map.m_frames) {
            auto &frame = it.second;
            frame.qcw = frame.qcw.normalized();
            frame.camera_model = map.m_camera_models[frame.camera_id];
            frame.keypoints_normalized.resize(frame.keypoints.size());
            for (int i = 0; i < frame.keypoints.size(); ++i) {
                auto &pt = frame.keypoints[i];
                Eigen::Vector2d point_normalized;
                frame.normalize_point(pt, point_normalized);
                frame.keypoints_normalized[i] = point_normalized;
            }


            Eigen::VectorXd cam(7);
            cam << frame.qcw.coeffs(), frame.tcw;
            map.cam_vec.push_back(cam);

            for (int i = 0; i < frame.keypoints.size(); ++i) {
                int track_id = frame.track_ids[i];
                if (track_id == -1)continue;
                Track &track = map.m_tracks[track_id];

                int num_obs = 0;
                for (int k = 1; k <= map.m_frames.size(); ++k) {
                    if (track.observations.count(k) != 0)num_obs++;
                }
                if (num_obs < 2)continue;

                if (map.point_vec.count(track_id) == 0)
                    map.point_vec[track_id] = track.point3D.data();

                OBS obs{frame.id - 1, track_id, frame.keypoints_normalized[i]};
                map.obs_vec.emplace_back(obs);
            }
        }
        std::cout << "associate_OBS Done!" << std::endl;
    }

    double calculate_reprojection_error(Map &map) {
        int count = 0;
        double mse = 0;
        for (auto &obs:map.obs_vec) {
            Eigen::Map<Eigen::Quaterniond> qcw(map.cam_vec[obs.camera_id].data());
            Eigen::Map<Eigen::Vector3d> tcw(map.cam_vec[obs.camera_id].data() + 4);
            Eigen::Map<Eigen::Vector3d> pw(map.point_vec[obs.point_id]);

            Eigen::Matrix3d Rcw = qcw.toRotationMatrix();
            Eigen::Vector3d Pc = Rcw * pw + tcw;
            Eigen::Vector2d Pn = Pc.head<2>() / Pc.z();
            Eigen::Vector2d res = Pn - obs.uv;

            mse += res.norm();
            count++;
        }
        return mse / count;
    }
}