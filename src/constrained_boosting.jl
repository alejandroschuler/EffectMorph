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

function fit_const_pair(data::ObsData, loss::Squared, τ::Real)
    Ȳ = mean(data.Y.observed)
    return Dict(t => Leaf(Ȳ - τ * data.N_treated[t]/data.N) for t in TF)
end

function fit_const_pair(data::ObsData, loss::Binomial, τ::Real)
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

predict_counterfactuals(estimator_pair, data) = Counterfactuals(predict(estimator_pair, data.X), data.W)


function evaluate(loss::Loss, Y::Counterfactuals, F::Counterfactuals; idx=:)
    return evaluate(loss, Y[idx].observed[true], F[idx].observed[true])
end


function gradient(loss::Loss, Y::Counterfactuals, F::Counterfactuals; idx=:)
    return Dict(t => gradient(loss, Y[idx].observed[true][Y.W==t], F[idx].observed[true][Y.W==t]) for t in TF)
end


function step_search(loss::Binomial, data::ObsData, F::Counterfactuals, f::Counterfactuals, tree_pair; tr=:)
    leaf_index = build_leaf_index(tree_pair, data.X)
    m = Model(solver=IpoptSolver())
    @variable(νt[])

function build_leaf_index(tree_pair::Dict{Bool,Node}, X::Matrix)
    assingments = Dict(t=>leaf_assignments(tree_pair[t], X) for t in TF)
    leaves = Dict(t=>Set(leaf_assignments(t)) for t in TF)
    assignment_index = Dict(t=>Dict(l=>find(assingments[t].==l) for l in leaves[t]) for t in TF)
    #leaves = Set(vcat([ls for (t,ls) in assingments]...))
    #assignment_index_matrix = [(t,l,find(derp[t].==l)) for t in TF, l in leaves]
    #assignment_index = Dict((t,l)=>ind for (t,l,ind) in vec(assignment_index_matrix) if length(ind)>0)
    return assignment_index, length(assignment_index)
end

function constrained_boost(data::ObsData, loss::Loss, τ::Real, n_trees::Int; 
                           max_depth=3, learning_rate=0.1, min_samples_leaf=1;
                           tr=:)
    data_tr = data[tr]

    estimators, F, residuals = Vector{Node}, Vector{Counterfactuals}, Vector{Vector{Float64}}

    const_pair = fit_const_pair(data_tr, loss, τ)
    push!(estimators, const_pair)
    push!(F, F[end]+predict_counterfactuals(estimators[end], data)) 
    push!(residuals, gradient(loss, data.Y, F[end], tr))

    for i in 1:n_trees
        tree_pair = Dict(t=>fit_regression_tree(data_tr.X[data_tr.W.==t,:], residuals[t], 
                                                min_samples_leaf=min_samples_leaf, 
                                                max_depth=max_depth) for t in TF)

        nu = step_search(loss, data, tree_pair)
        # integrage nu into the estimators

        push!(estimators, ... )
        push!(F, F[end]+predict_counterfactuals( ... , data)) 
        push!(residuals, gradient(loss, data.Y, F[end], tr))
    end

end # module
