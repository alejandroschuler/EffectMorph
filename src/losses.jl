__precompile__()

module Losses


export Loss, Squared, Binomial,
       evaluate, grad, effect


logistic(x) = 1./(1+exp(-x))

abstract Loss

type Squared <: Loss
end

function evaluate{T<:Real}(loss::Squared, Y::Vector{T}, F::Vector{T})
    return (Y-F).^2
end

function grad{T<:Real}(loss::Squared, Y::Vector{T}, F::Vector{T})
    return Y-F
end

function effect{T<:Real}(loss::Squared, Ft::Vector{T}, Ff::Vector{T})
    return mean(Ft - Ff)
end


type Binomial <: Loss
end

function evaluate{T<:Real}(loss::Binomial, Y::Vector{T}, F::Vector{T})
    return Y.*F - log(1+exp(F))
end

function grad{T<:Real}(loss::Binomial, Y::Vector{T}, F::Vector{T})
    return Y-logistic(F) 
end

function effect{T<:Real}(loss::Binomial, Ft::Vector{T}, Ff::Vector{T})
    return mean(logistic(Ft) - logistic(Ff))
end

end #module