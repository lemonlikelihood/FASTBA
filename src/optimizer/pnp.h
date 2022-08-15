#pragma once

#include "../utils/common.h"

class Frame;
class SlidingWindow;

void visual_inertial_pnp(
    SlidingWindow *map, Frame *frame, bool use_inertial = true, size_t max_iter = 50,
    const double &maxtime = 1.0e6);