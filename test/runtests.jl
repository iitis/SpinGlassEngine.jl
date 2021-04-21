using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using TensorCast

disable_logging(LogLevel(1))

using Test

function proj(state, dims::Union{Vector, NTuple})
    P = Matrix{Float64}[]
    for (σ, r) ∈ zip(state, dims)
        v = zeros(r)
        v[idx(σ)...] = 1.
        push!(P, v * v')
    end
    P
end

function tensor(ψ::AbstractMPS, state::State)
    C = I
    for (A, σ) ∈ zip(ψ, state)
        C *= A[:, idx(σ), :]
    end
    tr(C)
end

function tensor(ψ::MPS)
    dims = rank(ψ)
    Θ = Array{eltype(ψ)}(undef, dims)

    for σ ∈ all_states(dims)
        Θ[idx.(σ)...] = tensor(ψ, σ)
    end
    Θ
end


my_tests = []

push!(my_tests,
      "MPS_search.jl",
      "PEPS.jl",
      "contract.jl",
      "search.jl",
      "search_2.jl",
      "search_3.jl"
)

for my_test in my_tests
    include(my_test)
end
