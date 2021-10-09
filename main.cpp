#include <iostream>
//#include "src/optimizer/lm_optimizer.h"
#include "src/optimizer/lm_optimizer.h"
#include "src/optimizer/problem.h"
#include "src/optimizer/read_bal.h"
#include "src/optimizer/read_colmap.h"
#include <ceres/ceres.h>

int main() {
    //    std::string bal_path = "/Users/lemon/dataset/problem-49-7776-pre.txt";
    //    BalReader balReader(bal_path);
    //
    //    balReader.read_map();
    //    balReader.map->print_map();
    //    fast_ba::Problem problem(balReader.map);
    std::string colmap_path = "/Users/lemon/dataset/test";
    ColmapReader colmapReader(colmap_path);
    colmapReader.read_map();
    double init_error = colmapReader.map->calculate_reprojection_error();
    std::cout << "Init Error: " << init_error << std::endl;
    // std::cout << std::endl;
    fast_ba::Problem *problem = new fast_ba::Problem(colmapReader.map.get());
    SolverSummary *solver_summary = new SolverSummary();
    LMMinimizerOptions lmMinimizerOptions = LMMinimizerOptions();
    fast_ba::LMOptimizer lmOptimizer;
    lmOptimizer.init(lmMinimizerOptions, problem, solver_summary);
    lmOptimizer.solve_problem();

    double final_error = problem->map->calculate_reprojection_error();
    std::cout << "Final Error: " << final_error << std::endl;

    solver_summary->full_information();

    return 0;
}