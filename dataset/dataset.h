#pragma once

#include <Eigen/Eigen>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum NextDataType { AGAIN, GYROSCOPE, ACCELEROMETER, ATTITUDE, GRAVITY, IMAGE, IMU, END };

struct Pose {
    double t = -1.0;
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

struct AttitudeData {
    double t = -1.0;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

struct GyroscopeData {
    double t = -1.0;
    Eigen::Vector3d w = Eigen::Vector3d::Zero();
};

struct GravityData {
    double t = -1.0;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
};

struct IMUData {
    double t = -1.0;
    Eigen::Vector3d w = Eigen::Vector3d::Zero();
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
};

struct AccelerometerData {
    double t = -1.0;
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
};

struct VelocityData {
    double t = -1.0;
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d cov = Eigen::Vector3d::Zero();
};

struct ExtrinsicParams {
    Eigen::Quaterniond q_bs = Eigen::Quaterniond::Identity();
    Eigen::Vector3d p_bs = Eigen::Vector3d::Zero();
};

struct CameraCsvData {
    double t = -1.0;
    std::string filename;
};

struct ImageData {
    double t = -1.0;
    cv::Mat image;

    ImageData() = default;
    ImageData(const double &t, const std::string &filename) {
        // std::cout << "image filename: " << filename << std::endl;
        this->t = t;
        image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    }
};

struct CameraCsv {

    std::deque<CameraCsvData> items;

    void load(const std::string &filename) {
        items.clear();
        if (FILE *csv = fopen(filename.c_str(), "r")) {
            fscanf(csv, "%*[^\r\n]");
            char filename_buffer[2048];
            CameraCsvData item;
            while (not feof(csv)) {
                memset(filename_buffer, 0, 2048);
                if (fscanf(csv, "%lf,%2047[^\r\n]%*[\r\n]", &item.t, filename_buffer) != 2) {
                    break;
                }
                item.filename = std::string(filename_buffer);
                items.emplace_back(std::move(item));
            }
            fclose(csv);
            std::cout << "load " << filename << " successfully and " << items.size() << " images."
                      << std::endl;
        }
    }

    void save(const std::string &filename) const {
        if (FILE *csv = fopen(filename.c_str(), "w")) {
            fputs("#t[s:double],filename[string]\n", csv);
            for (auto item : items) {
                fprintf(csv, "%.9lf,%s\n", item.t, item.filename.c_str());
            }
            fclose(csv);
        }
    }
};

struct ImuCsv {
    std::deque<IMUData> items;
    void load(const std::string &filename) {
        items.clear();
        if (FILE *csv = fopen(filename.c_str(), "r")) {
            fscanf(csv, "%*[^\r\n]");
            IMUData item;
            while (not feof(csv)
                   && fscanf(
                          csv, "%lf,%lf,%lf,%lf,%lf,%lf,%lf%*[\r\n]", &item.t, &item.w.x(),
                          &item.w.y(), &item.w.z(), &item.a.x(), &item.a.y(), &item.a.z())
                          == 7) {
                items.emplace_back(std::move(item));
            }
            fclose(csv);
            std::cout << "load " << filename << " successfully and " << items.size() << " imus."
                      << std::endl;
        }
    }

    void save(const std::string &filename) const {
        if (FILE *csv = fopen(filename.c_str(), "w")) {
            fputs(
                "#t[s:double],w.x[rad/s:double],w.y[rad/s:double],w.z[rad/s:double],a.x[m/"
                "s^2:double],a.y[m/s^2:double],a.z[m/s^2:double]\n",
                csv);
            for (auto item : items) {
                fprintf(
                    csv, "%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf\n", item.t, item.w.x(),
                    item.w.y(), item.w.z(), item.a.x(), item.a.y(), item.a.z());
            }
            fclose(csv);
        }
    }
};

struct AttitudeCsv {

    std::deque<AttitudeData> items;

    void load(const std::string &filename) {
        items.clear();
        if (FILE *csv = fopen(filename.c_str(), "r")) {
            fscanf(csv, "%*[^\r\n]");
            AttitudeData item;
            while (not feof(csv)
                   && fscanf(
                          csv, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf%*[\r\n]", &item.t, &item.g.x(),
                          &item.g.y(), &item.g.z(), &item.q.x(), &item.q.y(), &item.q.z(),
                          &item.q.w())
                          == 8) {
                items.emplace_back(std::move(item));
            }
            fclose(csv);
        }
    }

    void save(const std::string &filename) const {
        if (FILE *csv = fopen(filename.c_str(), "w")) {
            fputs(
                "#t[s:double],g.x[m/s^2:double],g.y[m/s^2:double],g.z[m/"
                "s^2:double],atti.x[double],atti.y[double],atti.z[double],atti.w[double]\n",
                csv);
            for (auto item : items) {
                fprintf(
                    csv, "%.9e,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf\n", item.t, item.g.x(),
                    item.g.y(), item.g.z(), item.q.x(), item.q.y(), item.q.z(), item.q.w());
            }
            fclose(csv);
        }
    }
};
