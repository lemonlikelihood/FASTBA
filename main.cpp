#include "clipp.h"
#include <iostream>
//#include "src/optimizer/lm_optimizer.h"
#include "src/optimizer/lm_optimizer.h"
#include "src/optimizer/problem.h"
#include "src/optimizer/read_bal.h"
#include "src/optimizer/read_colmap.h"
#include <ceres/ceres.h>

// using namespace clipp;

int main(int argc, char *argv[]) {
    // std::string bal_path = "/Users/lemon/dataset/problem-49-7776-pre.txt";
    // BalReader balReader(bal_path);

    // balReader.read_map();
    // balReader.map->print_map();
    // double init_error = balReader.map->calculate_reprojection_error(true);
    // std::cout << "Init Error: " << init_error << std::endl;
    // fast_ba::Problem *problem = new fast_ba::Problem(balReader.map.get());

    // using namespace clipp;
    // auto cli = (required("-c", "--data-type") & value("colmap_path", colmap_path));

    // if (parse(argc, argv, cli)) {
    //     std::cout << "colmap_path: " << colmap_path << std::endl;
    // } else {
    //     std::cerr << make_man_page(cli, argv[0]) << std::endl;
    //     return 0;
    // }

    std::string colmap_path = "/Users/lemon/dataset/test";
    ColmapReader colmapReader(colmap_path);
    colmapReader.read_map();
    double init_error = colmapReader.map->calculate_reprojection_error();
    std::cout << "Init reprojection Error: " << init_error << "\n" << std::endl;
    // std::cout << std::endl;
    fast_ba::Problem *problem = new fast_ba::Problem(
        colmapReader.map.get(), fast_ba::PROJECTION_TYPE::CLASSICAL_PROJECTION);

    // fast_ba::Problem *problem =
    //     new fast_ba::Problem(colmapReader.map.get(), fast_ba::PROJECTION_TYPE::SYM_PROJECTION);

    SolverSummary *solver_summary = new SolverSummary();
    LMMinimizerOptions lmMinimizerOptions = LMMinimizerOptions();
    fast_ba::LMOptimizer lmOptimizer;
    lmOptimizer.init(lmMinimizerOptions, problem, solver_summary);
    lmOptimizer.solve_problem();

    double final_error = problem->map->calculate_reprojection_error();
    std::cout << "Final reprojection Error: " << final_error << "\n" << std::endl;

    solver_summary->full_information();
    return 0;
}