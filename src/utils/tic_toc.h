//
// Created by lemon on 2020/9/22.
//

#ifndef BA_COST_TIC_TOC_H
#define BA_COST_TIC_TOC_H

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <ratio>

class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    double toc()
    {
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed_seconds = end - start;
        return elapsed_seconds.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start, end;
};

#endif //BA_COST_TIC_TOC_H
