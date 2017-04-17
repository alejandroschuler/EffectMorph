# note that when using the binomial loss, the treatment effect must be between -1 and 1

push!(LOAD_PATH, "../src")
using Estimators
using Losses
using ObsDataStructures
using JuMP
using Ipopt

const TF = [true, false]

function fit_const_estimators(Data::ObsData, loss::Squared, τ::Real)
    Ȳ = mean(Data.Y.observed)
    return Dict(t => Leaf(Ȳ - τ * Data.N_treated[t]/Data.N) for t in TF)
end

function fit_const_estimators(Data::ObsData, loss::Binomial, τ::Real)
    m = Model(solver=IpoptSolver())
    @variable(m, 0 <= πt <= 1)
    @variable(m, 0 <= πf <= 1)
    @constraint(m, πt-πf == τ)
    @NLobjective(m, Max, Data.N_treated[true]*log(1-πf) + 
                         Data.N_treated[false]*log(πt))
    status = solve(m)
    π = Dict(true=>getvalue(πt), false=>getvalue(πf))
    return Dict(t  => Leaf(log(π[t]/(1-π[t]))) for t in TF)
end

function constrained_boost(Data, loss, τ::Real, n_trees::Int; 
                           max_depth=3, learning_rate=0.1, min_samples_leaf=1,
                           training_index=nothing)
    if training_index == nothing
        training_index = 1:Data.N
    end

    estimators, F_hat, residuals = [], [], []

    const_estimator = fit_const_estimators(Data, loss, τ)