# note that when using the binomial loss, the treatment effect must be between -1 and 1

__precompile()__

module CausalEffectMorph

include_dependency("data_structures.jl")
include_dependency("losses.jl")
include_dependency("estimators.jl")

include("data_structures.jl")
include("losses.jl")
include("estimators.jl")

using .Estimators
using .Losses
using .ObsDataStructures

using JuMP
using Ipopt

const TF = [true, false]

predict_counterfactuals(estimator, data) = Counterfactuals(predict(const_estimator, data.X), data.W)

function calculate_residuals(data::ObsData, F::Counterfactuals, tr, loss::Loss)
    Y = data.Y.observed[true][tr]
    F = F.observed[true][tr]
    return evaluate(loss, Y, F)

function fit_const_estimators(data::ObsData, loss::Squared, τ::Real)
    Ȳ = mean(data.Y.observed)
    return Dict(t => Leaf(Ȳ - τ * data.N_treated[t]/data.N) for t in TF)
end

function fit_const_estimators(data::ObsData, loss::Binomial, τ::Real)
    m = Model(solver=IpoptSolver())
    @variable(m, 0 <= πt <= 1)
    @variable(m, 0 <= πf <= 1)
    @constraint(m, πt-πf == τ)
    @NLobjective(m, Max, data.N_treated[true]*log(1-πf) + 
                         data.N_treated[false]*log(πt))
    status = solve(m)
    π = Dict(true=>getvalue(πt), false=>getvalue(πf))
    return Dict(t  => Leaf(log(π[t]/(1-π[t]))) for t in TF)
end

function constrained_boost(data::ObsData, loss::Loss, τ::Real, n_trees::Int; 
                           max_depth=3, learning_rate=0.1, min_samples_leaf=1,
                           training_index=nothing)
    if training_index == nothing
        training_index = 1:data.N
    end

    estimators, F, residuals = [], [], []

    const_estimator = fit_const_estimators(data, loss, τ)
    push!(F, F[end]+predict_counterfactuals(const_estimator, data)) # the full set of predicted counterfactuals, a Vector{Counterfactuals}
    push!(residuals, calculate_residuals(data, F[end], training_index, loss))

end # module
