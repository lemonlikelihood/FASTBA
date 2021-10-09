//
// Created by lemon on 2021/1/20.
//

#ifndef FASTBA_LM_OPTIMIZER_H
#define FASTBA_LM_OPTIMIZER_H

#include "datatype.h"
#include "problem.h"
#include "solver.h"
namespace fast_ba {
class LMOptimizer {
public:
    LMOptimizer();

    ~LMOptimizer();

    void init(const LMMinimizerOptions &options, Problem *problem_, SolverSummary *solve_summary_);

    void solve_problem();

private:
    void StepRejected(double step_quality_);

    void StepAccepted(double step_quality_);

    bool ParameterToleranceReached();
    bool FunctionToleranceReached();
    bool GradientToleranceReached();
    bool MaxSolverIterationsReached();
    bool MinTrustRegionRadiusReached();

    bool IterationZero();

    bool EvaluateGradientAndJacobian();


    bool FinalizeIterationAndCheckIfMinimizerCanContinue();

    LMMinimizerOptions lm_minimizer_options;
    IterationSummary iteration_summary;
    SolverSummary *solver_summary;
    Problem *problem;
    Solver *solver;
};

} // namespace fast_ba

#endif //FASTBA_LM_OPTIMIZER_H
