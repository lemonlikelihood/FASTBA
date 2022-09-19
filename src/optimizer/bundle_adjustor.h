#pragma once

#include "../utils/common.h"

class Map;
class Frame;

class BundleAdjustor {
    struct BundleAdjustorSolver; // pimpl

public:
    BundleAdjustor();
    virtual ~BundleAdjustor();

    bool
    solve(Map *map, bool use_inertial = true, size_t max_iter = 50, const double &max_time = 1.0e6);
    void marginalize_frame(Map *map,size_t index);

private:
    std::unique_ptr<BundleAdjustorSolver> solver;
};