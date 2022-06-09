#pragma once

#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

struct CameraCsv {
    struct CameraData {
        double t;
        std::string filename;
    };

    std::deque<CameraData> items;

    void load(const std::string &filename) {
        items.clear();
        if (FILE *csv = fopen(filename.c_str(), "r")) {
            fscanf(csv, "%*[^\r\n]");
            char filename_buffer[2048];
            CameraData item;
            while (not feof(csv)) {
                memset(filename_buffer, 0, 2048);
                if (fscanf(csv, "%lf,%2047[^\r\n]%*[\r\n]", &item.t, filename_buffer) != 2) {
                    break;
                }
                item.filename = std::string(filename_buffer);
                items.emplace_back(std::move(item));
            }
            fclose(csv);
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
    struct ImuData {
        double t;
        struct {
            double x;
            double y;
            double z;
        } w;
        struct {
            double x;
            double y;
            double z;
        } a;
    };

    std::deque<ImuData> items;

    void load(const std::string &filename) {
        items.clear();
        if (FILE *csv = fopen(filename.c_str(), "r")) {
            fscanf(csv, "%*[^\r\n]");
            ImuData item;
            while (not feof(csv)
                   && fscanf(
                          csv, "%lf,%lf,%lf,%lf,%lf,%lf,%lf%*[\r\n]", &item.t, &item.w.x, &item.w.y,
                          &item.w.z, &item.a.x, &item.a.y, &item.a.z)
                          == 7) {
                items.emplace_back(std::move(item));
            }
            fclose(csv);
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
                    csv, "%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf\n", item.t, item.w.x, item.w.y,
                    item.w.z, item.a.x, item.a.y, item.a.z);
            }
            fclose(csv);
        }
    }
};

struct AttitudeCsv {
    struct AttitudeData {
        double t;
        struct {
            double x;
            double y;
            double z;
        } g;
        struct {
            double x;
            double y;
            double z;
            double w;
        } atti;
    };

    std::deque<AttitudeData> items;

    void load(const std::string &filename) {
        items.clear();
        if (FILE *csv = fopen(filename.c_str(), "r")) {
            fscanf(csv, "%*[^\r\n]");
            AttitudeData item;
            while (not feof(csv)
                   && fscanf(
                          csv, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf%*[\r\n]", &item.t, &item.g.x,
                          &item.g.y, &item.g.z, &item.atti.x, &item.atti.y, &item.atti.z,
                          &item.atti.w)
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
                    csv, "%.9e,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf\n", item.t, item.g.x,
                    item.g.y, item.g.z, item.atti.x, item.atti.y, item.atti.z, item.atti.w);
            }
            fclose(csv);
        }
    }
};
