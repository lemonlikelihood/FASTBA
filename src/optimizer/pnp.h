#pragma once

#include "../utils/common.h"

class Frame;
class Map;

void visual_inertial_pnp(
    Map *map, Frame *frame, bool use_inertial = true, size_t max_iter = 50,
    const double &maxtime = 1.0e6);