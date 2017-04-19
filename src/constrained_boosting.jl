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

using DataStructures
using JuMP
using Ipopt

const TF = [true, false]

function fit_const_pair(data::ObsData, loss::Squared, τ::Real)
    Ȳ = mean(data.Y.observed[true])
    return Dict(t => Leaf(Ȳ +(t*1) * τ * data.N_treated[t]/data.N) for t in TF)
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

function predict_counterfactuals{T<:Estimators.Estimator}(estimator_pair::Dict{Bool,T}, data::ObsData)
    return Counterfactuals(predict(estimator_pair, data.X.X), data.W)
end

function obs_gradient{T<:Loss}(loss::T, Y::Outcome, F::Counterfactuals; idx=:)
    return Dict(t => grad(loss, Γ(Y[idx],t=t), Γ(F[idx],o=true,t=t)) for t in TF)
end

function build_leaf_index(tree_pair::Dict{Bool,Node}, X::Covariates)
    assignments = Dict(t=>assign_leaves(tree_pair[t], X.X) for t in TF)
    leaves = Set(vcat([ls for (t,ls) in assignments]...))
        assignment_index_matrix = [(t,l,find(assignments[t].==l)) for l in leaves, t in TF]
    assignment_index = OrderedDict((t,l)=>ind for (t,l,ind) in vec(assignment_index_matrix) if length(ind)>0)
    return assignment_index
end

function step_search(loss::Squared, data::ObsData, F::Counterfactuals, tree_pair::Dict{Bool,Node}; tr=:)
    idx = build_leaf_index(tree_pair, data.X)

    squares, coefs, constraint_coefs = Vector{Int}(0), Vector{Float64}(0), Vector{Int}(0)
    for ((t,l), i) in idx
        idx_leaf = i∩tr
        push!(squares, length(idx_leaf))
        push!(constraint_coefs, (2*t-1)*length(i))
        if length(idx_leaf) > 0
            push!(coefs, sum(Γ(data.Y[idx_leaf],t=t) - Γ(F[idx_leaf],o=true,t=t)))
        else 
            push!(coefs, 0.0)
        end
    end

    L = length(coefs)
    m = Model(solver=IpoptSolver())
    @variable(m, ν[1:L])
    @constraint(m, sum(constraint_coefs[i]*ν[i] for i in 1:L) == 0)
    @objective(m, Min, sum(ν[i]^2*squares[i] + coefs[i]*ν[i] for i in 1:L))
    status = solve(m)

    ν_dict = Dict(true=>Dict(), false=>Dict())
    for (νi, (t,l)) in zip(getvalue(ν), keys(idx))
        ν_dict[t][l] = νi
    end
                            
    return ν_dict
end

function step_search(loss::Binomial, data::ObsData, F::Counterfactuals, f::Counterfactuals, tree_pair; tr=:)
    return nothing # TO DO
end

function constrained_boost(data::ObsData, loss::Loss, τ::Float64, n_trees::Int; 
                           max_depth=3, learning_rate=0.1, min_samples_leaf=1;
                           tr=:)
    data_tr = data[tr]

    estimators, F, residuals = Vector{Node}, Vector{Counterfactuals}, Vector{Vector{Float64}}

    const_pair = fit_const_pair(data_tr, loss, τ)
    push!(estimators, const_pair)
    push!(F, F[end]+predict_counterfactuals(estimators[end], data)) 
    push!(residuals, obs_gradient(loss, data.Y, F[end], idx=tr))

    for i in 1:n_trees
        tree_pair = Dict(t=>fit_regression_tree(Γ(data.X[tr],t=t) , residuals[t], 
                                                        min_samples_leaf=min_samples_leaf, 
                                                        max_depth=max_depth) for t in TF);

        nu = step_search(loss, data, tree_pair)
        # integrage nu into the estimators

        push!(estimators, ... )
        push!(F, F[end]+predict_counterfactuals( ... , data)) 
        push!(residuals, obs_gradient(loss, data.Y, F[end], tr))
    end

end # module
