//
// Created by lemon on 2021/1/20.
//

#include "lm_optimizer.h"
#include "../utils/tic_toc.h"
#include <iostream>

namespace fast_ba {
LMOptimizer::LMOptimizer() {}
LMOptimizer::~LMOptimizer() {}
void LMOptimizer::init(
    const LMMinimizerOptions &options, Problem *problem_, SolverSummary *solver_summary_) {
    lm_minimizer_options = options;
    problem = problem_;
    solver_summary = solver_summary_;
    solver = new Solver(problem);
    solver->max_radius = lm_minimizer_options.max_radius;
    solver->min_radius = lm_minimizer_options.min_radius;
    solver->radius = 1;
    std::cout.precision(15);
}

bool LMOptimizer::IterationZero() {
    TicToc iter_time;
    iteration_summary = IterationSummary();
    //    iteration_summary.iteration =0;
    iteration_summary.trust_region_radius = 1;

    if (!EvaluateGradientAndJacobian()) {
        return false;
    }

    solver->compute_cost(iteration_summary.cost);
    solver_summary->initial_cost = iteration_summary.cost;
    solver_summary->minimum_cost = iteration_summary.cost;
    iteration_summary.step_is_vaild = true;
    iteration_summary.step_is_successful = true;
    iteration_summary.gradient_max_norm = solver->gradient_max_norm;
    iteration_summary.iteration_time = iter_time.toc();
    // iteration_summary.print();
    // solver_summary->iterations.emplace_back(iteration_summary);
    // std::cout << std::endl;
    return true;
}

void LMOptimizer::solve_problem() {
    TicToc total_time;
    IterationZero();
    while (FinalizeIterationAndCheckIfMinimizerCanContinue()) {
        iteration_summary = IterationSummary();
        iteration_summary.iteration = solver_summary->iterations.back().iteration + 1;
        TicToc iteration_time;
        solver->solve_linear_problem();
        solver->compute_relative_decrease();
        // std::cout << std::endl;
        iteration_summary.step_is_vaild =
            solver->model_cost_change > 0.0 && solver->relative_decrease > 1e-3;
        iteration_summary.cost = solver->candidate_cost;
        iteration_summary.cost_change = solver->cost_change;
        if (!iteration_summary.step_is_vaild) {
            solver_summary->num_consecutive_invalid_steps++;
            if (solver_summary->num_consecutive_invalid_steps
                > lm_minimizer_options.max_num_consecutive_invalid_steps) {
                solver_summary->termination_type = FAILURE;
                std::cout << "failed : " << solver_summary->num_consecutive_invalid_steps << " > "
                          << lm_minimizer_options.max_num_consecutive_invalid_steps << std::endl;
                break;
            }
            StepRejected(0);
            iteration_summary.iteration_time = iteration_time.toc();
            // solver_summary->iterations.emplace_back(iteration_summary);
            continue;
        }
        solver->compute_step_norm_x_norm(iteration_summary.step_norm, iteration_summary.x_norm);
        // std::cout << "iteration_summary.step_norm: " << iteration_summary.step_norm << std::endl;
        if (ParameterToleranceReached()) {
            break;
        }
        if (FunctionToleranceReached()) {
            break;
        }
        StepAccepted(solver->relative_decrease);
        iteration_summary.iteration_time = iteration_time.toc();
        // solver_summary->iterations.emplace_back(iteration_summary);
        // solver_summary->iterations.emplace_back(iteration_summary);
    }

    double total_time_ms = total_time.toc();
    solver->param_recover();
    double final_cost;
    solver->compute_cost(final_cost);
    std::cout << "Final Cost : " << final_cost << std::endl;
    std::cout << "Iterations Size: " << solver_summary->iterations.size() << std::endl;
    std::cout << "Total Time: " << total_time_ms << " ms\n" << std::endl;
}

bool LMOptimizer::MaxSolverIterationsReached() {
    if (iteration_summary.iteration < lm_minimizer_options.max_num_iterations) {
        return false;
    }
    std::cout << "Terminating: Maximum number of iterations reached. Number of iterations: "
              << iteration_summary.iteration << std::endl;
    solver_summary->termination_type = NO_CONVERGENCE;
    return true;
}

bool LMOptimizer::ParameterToleranceReached() {
    const double step_size_tolerance =
        lm_minimizer_options.parameter_tolerance
        * (iteration_summary.x_norm + lm_minimizer_options.parameter_tolerance);
    if (iteration_summary.step_norm > step_size_tolerance) {
        return false;
    }
    std::cout << "ParameterToleranceReached : " << iteration_summary.step_norm << " < "
              << step_size_tolerance << " " << std::endl;
    solver_summary->termination_type = CONVERGENCE;
    return true;
}

bool LMOptimizer::FunctionToleranceReached() {
    //        solver->compute_cost_change();
    double cost_change = solver->cost_change;
    const double absolute_function_tolerance =
        lm_minimizer_options.function_tolerance * solver->current_cost;

    if (std::abs(cost_change) > absolute_function_tolerance) {
        return false;
    }
    std::cout << "Function tolerance reached. |cost_change|/cost: " << std::abs(cost_change)
              << " < " << absolute_function_tolerance << std::endl;
    return true;
}

bool LMOptimizer::GradientToleranceReached() {
    if (!iteration_summary.step_is_vaild
        || iteration_summary.gradient_max_norm > lm_minimizer_options.gradient_tolerance) {
        return false;
    }
    std::cout << "CONVERGENCE, Gradient tolerance reached. Gradient max norm: "
              << iteration_summary.gradient_max_norm
              << " <= " << lm_minimizer_options.gradient_tolerance << std::endl;
    solver_summary->termination_type = CONVERGENCE;
    return true;
}

bool LMOptimizer::MinTrustRegionRadiusReached() {
    if (iteration_summary.trust_region_radius > lm_minimizer_options.min_radius) {
        return false;
    }
    std::cout << "CONVERGENCE, Minimum trust region reached. Trust region radius : "
              << iteration_summary.trust_region_radius << " <= " << lm_minimizer_options.min_radius
              << std::endl;
    solver_summary->termination_type = CONVERGENCE;
    return true;
}

bool LMOptimizer::EvaluateGradientAndJacobian() {
    problem->evaluate();

    solver->compute_gradient_norm();
    //        problem.compute_cost(iteration_summary.cost);
    //        problem.compute_step_norm_x_norm(iteration_summary.step_norm, iteration_summary.x_norm);
    return true;
}

void LMOptimizer::StepAccepted(double step_quality) {
    solver->step_accepted(step_quality);
    solver->param_update();
    solver->compute_gradient_norm();
    iteration_summary.gradient_max_norm = solver->gradient_max_norm;
    iteration_summary.trust_region_radius = solver->radius;
    iteration_summary.relative_decrease = solver->relative_decrease;
    solver_summary->num_consecutive_invalid_steps = 0;
    // solver_summary->iterations.emplace_back(iteration_summary);
    // std::cout << "radius: " << iteration_summary.trust_region_radius << std::endl;
}

void LMOptimizer::StepRejected(double step_quality) {
    solver->step_rejected(0);
    // iteration_summary.cost_change = 0;
    // iteration_summary.step_norm = 0;
    iteration_summary.relative_decrease = solver->relative_decrease;
    iteration_summary.trust_region_radius = solver->radius;
    // solver_summary->iterations.emplace_back(iteration_summary);
    // std::cout << "radius: " << iteration_summary.trust_region_radius << std::endl;
}

bool LMOptimizer::FinalizeIterationAndCheckIfMinimizerCanContinue() {
    if (iteration_summary.step_is_successful) {
        ++solver_summary->num_successful_steps;
    } else {
        ++solver_summary->num_unsuccessful_steps;
    }

    iteration_summary.trust_region_radius = solver->radius;
    solver_summary->iterations.push_back(iteration_summary);
    if (MaxSolverIterationsReached()) {
        return false;
    }
    if (GradientToleranceReached()) {
        return false;
    }
    if (MinTrustRegionRadiusReached()) {
        return false;
    }

    return true;
}

} // namespace fast_ba
