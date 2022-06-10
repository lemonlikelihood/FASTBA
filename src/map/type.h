#include <Eigen/Eigen>
#include <bitset>

inline constexpr size_t nil() {
    return size_t(-1);
}

struct Pose {
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

struct Attitude {
    double t = -1.0;
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
};

struct Gyroscope {
    double t = -1.0;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
};

struct Gravity {
    double t = -1.0;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
};

struct Accelerometer {
    double t = -1.0;
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
};

struct Velocity {
    double t = -1.0;
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d cov = Eigen::Vector3d::Zero();
};

struct ExtrinsicParams {
    Eigen::Quaterniond q_cs = Eigen::Quaterniond::Identity();
    Eigen::Vector3d p_cs = Eigen::Vector3d::Zero();
};

template<class FlagEnum>
struct Flagged {
    static const size_t flag_num = static_cast<size_t>(FlagEnum::FLAG_NUM);

    Flagged() { flags.reset(); }

    bool operator==(const Flagged &rhs) const { return (flags == rhs.flags); }

    bool flag(FlagEnum f) const { return flags[static_cast<size_t>(f)]; }

    typename std::bitset<flag_num>::reference flag(FlagEnum f) {
        return flags[static_cast<size_t>(f)];
    }

    bool any_of(std::initializer_list<FlagEnum> flags) const {
        return std::any_of(flags.begin(), flags.end(), [this](FlagEnum f) { return flag(f); });
    }

    bool all_of(std::initializer_list<FlagEnum> flags) const {
        return std::all_of(flags.begin(), flags.end(), [this](FlagEnum f) { return flag(f); });
    }

    bool none_of(std::initializer_list<FlagEnum> flags) const {
        return std::none_of(flags.begin(), flags.end(), [this](FlagEnum f) { return flag(f); });
    }

private:
    std::bitset<flag_num> flags;
};