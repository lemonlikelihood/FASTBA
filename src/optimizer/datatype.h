//
// Created by lemon on 2021/1/20.
//

#ifndef FASTBA_DATATYPE_H
#define FASTBA_DATATYPE_H

#include <iostream>
#include <vector>

enum TerminationType {
    // Minimizer terminated because one of the convergence criterion set
    // by the user was satisfied.
    //
    // 1.  (new_cost - old_cost) < function_tolerance * old_cost;
    // 2.  max_i |gradient_i| < gradient_tolerance
    // 3.  |step|_2 <= parameter_tolerance * ( |x|_2 +  parameter_tolerance)
    //
    // The user's parameter blocks will be updated with the solution.
    CONVERGENCE,

    // The solver ran for maximum number of iterations or maximum amount
    // of time specified by the user, but none of the convergence
    // criterion specified by the user were met. The user's parameter
    // blocks will be updated with the solution found so far.
    NO_CONVERGENCE,

    // The minimizer terminated because of an error.  The user's
    // parameter blocks will not be updated.
    FAILURE,

    // Using an IterationCallback object, user code can control the
    // minimizer. The following enums indicate that the user code was
    // responsible for termination.
    //
    // Minimizer terminated successfully because a user
    // IterationCallback returned SOLVER_TERMINATE_SUCCESSFULLY.
    //
    // The user's parameter blocks will be updated with the solution.
    USER_SUCCESS,

    // Minimizer terminated because because a user IterationCallback
    // returned SOLVER_ABORT.
    //
    // The user's parameter blocks will not be updated.
    USER_FAILURE
};

enum LinearSolverTerminationType {
    // Termination criterion was met.
    LINEAR_SOLVER_SUCCESS,

    // Solver ran for max_num_iterations and terminated before the
    // termination tolerance could be satisfied.
    LINEAR_SOLVER_NO_CONVERGENCE,

    // Solver was terminated due to numerical problems, generally due to
    // the linear system being poorly conditioned.
    LINEAR_SOLVER_FAILURE,

    // Solver failed with a fatal error that cannot be recovered from,
    // e.g. CHOLMOD ran out of memory when computing the symbolic or
    // numeric factorization or an underlying library was called with
    // the wrong arguments.
    LINEAR_SOLVER_FATAL_ERROR
};

struct IterationSummary {
    int iteration = 0;
    bool step_is_vaild = false;
    bool step_is_nonmonotonic = false;
    bool step_is_successful = false;

    double cost = 0.0;
    double minimum_cost = 0.0;
    double cost_change = 0.0;
    double gradient_max_norm = 0.0;
    double gradient_norm = 0.0;
    double x_norm = 0.0;
    double step_norm = 0.0;
    double relative_decrease = 0.0;
    double trust_region_radius = 0.0;
    int linear_solver_iterations = 0.0;

    double iteration_time = 0.0;

    void print() {
        std::cout << "Iteration num: " << iteration << std::endl;
        std::cout << "Cost: " << cost << std::endl;
        std::cout << "Cost change: " << cost_change << std::endl;
        std::cout << "|Gradient|: " << gradient_max_norm << std::endl;
        std::cout << "|Step|: " << step_norm << std::endl;
        std::cout << "Tr_ratio: " << relative_decrease << std::endl;
        std::cout << "Tr_radius: " << trust_region_radius << std::endl;
        std::cout << "Successful: " << step_is_successful << std::endl;
        std::cout << "Iter_time: " << iteration_time << " ms" << std::endl;
        std::cout << std::endl;
    }
};

struct SolverSummary {
    TerminationType termination_type;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    double minimum_cost = 0.0;
    double num_consecutive_invalid_steps = 0.0;

    std::vector<IterationSummary> iterations;
    int num_successful_steps;
    int num_unsuccessful_steps;

    void full_information() {
        for (int i = 0; i < iterations.size(); ++i) {
            iterations[i].print();
        }
    }
};

struct LMMinimizerOptions {
    int max_num_iterations = 50;
    double gradient_tolerance = 1e-10;
    double parameter_tolerance = 1e-12;
    double function_tolerance = 1e-12;
    double min_relative_decrease = 0;
    int max_num_consecutive_invalid_steps = 5;
    double max_radius = 1e16;
    double min_radius = 1e-32;
};

#endif //FASTBA_DATATYPE_H
