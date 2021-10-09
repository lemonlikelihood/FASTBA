//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_COST_FUNCTION_H
#define FASTBA_COST_FUNCTION_H
class CostFunction
{
public:
    CostFunction() : num_residuals_(0) {}
    CostFunction(const CostFunction &) = delete;
    void operator=(const CostFunction &) = delete;

    virtual ~CostFunction() {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const = 0;

    const std::vector<int32_t> &parameter_block_sizes() const
    {
        return parameter_block_sizes_;
    }

    int num_residuals() const { return num_residuals_; }

protected:
    std::vector<int32_t> *mutable_parameter_block_sizes()
    {
        return &parameter_block_sizes_;
    }

    void set_num_residuals(int num_residuals) { num_residuals_ = num_residuals; }

private:
    // Cost function signature metadata: number of inputs & their sizes,
    // number of outputs (residuals).
    std::vector<int32_t> parameter_block_sizes_;
    int num_residuals_;
};

template <int kNumResiduals, int... Ns>
class SizedCostFunction : public CostFunction
{
public:
    // static_assert(kNumResiduals > 0 || kNumResiduals == DYNAMIC,
    //               "Cost functions must have at least one residual block.");
    // static_assert(internal::StaticParameterDims<Ns...>::kIsValid,
    //               "Invalid parameter block dimension detected. Each parameter "
    //               "block dimension must be bigger than zero.");

    // using ParameterDims = internal::StaticParameterDims<Ns...>;

    SizedCostFunction()
    {
        set_num_residuals(kNumResiduals);
        *mutable_parameter_block_sizes() = std::vector<int32_t>{Ns...};
    }

    virtual ~SizedCostFunction() {}

    // Subclasses must implement Evaluate().
};
#endif //FASTBA_COST_FUNCTION_H
