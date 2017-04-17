
module ObsDataStructures
import Base, Base.getindex
export label_treatments, 
       Counterfactuals, ObsData,
       getindex

function label_treatments(W::Vector)
    treatment_levels = sort!([w for w in Set(W)])
    if length(treatment_levels) != 2
        error("Only binary treatments")
    end
    return Dict(label=>t for (label, t) in zip([true, false], treatment_levels))
end

type Counterfactuals
    treated::Dict{Bool,Vector}
    observed::Dict{Bool,Vector}
    W::Vector{Bool}
end

function Counterfactuals(treated::Dict, W::Vector{Bool})
    observed = Dict(c=>Vector{Union{Void, Real}}(length(W)) for c in [true, false])
    observed[true][W] = treated[true][W]
    observed[true][!W] = treated[false][!W]
    observed[false][W] = treated[false][W]
    observed[false][!W] = treated[true][!W]
    return Counterfactuals(treated, observed, W)
end

function Counterfactuals(observed_vec::Vector, W::Vector{Bool})
    observed = Dict(true  => observed_vec, 
                    false => Vector{Union{Void, Real}}([nothing for i in 1:length(W)]))
    treated = Dict(c=>Vector{Union{Void, Real}}(length(W)) for c in [true, false])
    treated[true][W] = observed[true][W]   
    treated[false][!W] = observed[true][!W]  
    treated[false][W] = observed[false][W]  
    treated[true][!W] = observed[false][!W] 
    return Counterfactuals(treated, observed, W)
end

function getindex(Y::Counterfactuals, index)
    return Counterfactuals(Dict(true=>Y.treated[true][index],
                                false=>Y.treated[false][index]),
                           Y.W[index])
end

immutable ObsData
    X::Matrix{Real}
    W::Vector{Bool}
    Y::Counterfactuals
    trts::Dict{Bool,Any}
    N::Int
    N_treated::Dict{Bool,Int}
end

function ObsData(X::Matrix, W::Vector, Y::Vector; trts=nothing)
    trts == nothing ? trts = label_treatments(W) : nothing
    W_bool = Vector{Bool}(length(W))
    for (t,w) in trts
        W_bool[W.==w] = t
    end
    Y = Counterfactuals(Y, W_bool)
    return ObsData(X, W_bool, Y, trts, length(W), Dict(true=>sum(W_bool), false=>sum(!W_bool)))
end

function ObsData(X::Matrix, W::Vector{Bool}, Y::Counterfactuals, trts::Dict{Bool,Any})
    return ObsData(X, W, Y, trts, length(W), sum(W), sum(!W))
end

function getindex(data::ObsData, index)
    Y = Counterfactuals(data.Y.treated, data.W)
    return ObsData(data.X[index,:], data.W[index], Y[index], data.trts)
end

end #module