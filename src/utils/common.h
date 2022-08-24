#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <forward_list>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "debug.h"
#include <Eigen/Eigen>

#define GRAVITY_NORM 9.80665

inline constexpr size_t nil() {
    return size_t(-1);
}

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