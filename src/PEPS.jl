using SpinGlassTensors
export PEPSNetwork, contract_network
export generate_boundary, peps_tensor, node_from_index


const DEFAULT_CONTROL_PARAMS = Dict(
    "bond_dim" => typemax(Int),
    "var_tol" => 1E-8,
    "sweeps" => 4.,
    "β" => 1.
)


function peps_lattice(m, n)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end



@memoize Dict function _right_env(peps, i::Int, ∂v::Vector{Int})
    W = MPO(peps, i)
    ψ = MPS(peps, i+1)    
    right_env(ψ, W, ∂v)
end

@memoize Dict function _left_env(peps, i::Int, ∂v::Vector{Int})
    ψ = MPS(peps, i+1)
    left_env(ψ, ∂v)
end


struct PEPSNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
    factor_graph::LabelledGraph{T, NTuple{2, Int}} where T
    network_graph::LabelledGraph{S, NTuple{2, Int}} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    β::Real
    bond_dim::Int
    var_tol::Real
    sweeps::Int

    function PEPSNetwork(
        m::Int,
        n::Int,
        factor_graph,
        transformation::LatticeTransformation;
        β::Real,
        bond_dim::Int=typemax(Int),
        var_tol::Real=1E-8,
        sweeps::Int=4
    )
        vmap = vertex_map(transformation, m, n)
        ng = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)
        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end
        

        new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim, var_tol, sweeps)
    end
end


function projectors(network::PEPSNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
end


@memoize Dict function peps_tensor(peps::PEPSNetwork, i::Int, j::Int) where {T <: Number}
    # println("Computing peps_tensor for $(i), $(j)")
    # generate tensors from projectors
    A = build_tensor(peps, (i, j))

    # include energy
    h = build_tensor(peps, (i, j-1), (i, j))
    v = build_tensor(peps, (i-1, j), (i, j))
    # println("Starting @tensor")
    @tensor B[l, u, r, d, σ] := h[l, l̃] * v[u, ũ] * A[l̃, ũ, r, d, σ]
    B
end

#@memoize Dict peps_tensor(peps::PEPSNetwork, i::Int, j::Int) = peps_tensor(Float64, peps, i, j)


function SpinGlassTensors.MPO(::Type{T},
    peps::PEPSNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) where {T <: Number}
    W = MPO(T, peps.ncols)

    for j ∈ 1:peps.ncols
        A = peps_tensor(peps, i, j)
        v = get(states_indices, peps.vertex_map((i, j)), nothing)
        if v !== nothing
       #     @cast B[l, u, r, d] |= A[l, u, r, d, $(v)]
             B = A[:, :, :, :, v]
        else
            B = dropdims(sum(A, dims=5), dims=5)
            #@reduce B[l, u, r, d] |= sum(σ) A[l, u, r, d, σ]
        end
        W[j] = B
    end
    W
end


@memoize Dict SpinGlassTensors.MPO(
    peps::PEPSNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
) = MPO(Float64, peps, i, states_indices)


function compress(
    ψ::AbstractMPS,
    peps::PEPSNetwork;
)
    if bond_dimension(ψ) < peps.bond_dim return ψ end
    SpinGlassTensors.compress(ψ, peps.bond_dim, peps.var_tol, peps.sweeps)
end


@memoize Dict function SpinGlassTensors.MPS(
    peps::PEPSNetwork,
    i::Int,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
)
    if i > peps.nrows return IdentityMPS() end
    W = MPO(peps, i, states_indices)
    ψ = MPS(peps, i+1, states_indices)
    compress(W * ψ, peps)
end


function contract_network(
    peps::PEPSNetwork,
    states_indices::Dict{NTuple{2, Int}, Int} = Dict{NTuple{2, Int}, Int}()
)
    ψ = MPS(peps, 1, states_indices)
    prod(dropindices(ψ))[]
end


node_index(peps::PEPSNetwork, node::NTuple{2, Int}) = peps.ncols * (node[1] - 1) + node[2]

# Below is needed because we are counting fom 1 ¯\_(ツ)_/¯
# Therefore, when computing column from index, we can't just use remainder,
# we need to wrap to m if k % m is zero.
_mod_wo_zero(k, m) = k % m == 0 ? m : k % m


node_from_index(peps::PEPSNetwork, index::Int) =
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))


iteration_order(peps::PEPSNetwork) = [(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols]


function boundary_at_splitting_node(peps::PEPSNetwork, node::NTuple{2, Int})
    i, j = node
    [
        [((i, k), (i+1, k)) for k ∈ 1:j-1]...,
        ((i, j-1), (i, j)),
        [((i-1, k), (i, k)) for k ∈ j:peps.ncols]...
    ]
end


function _normalize_probability(prob::Vector{T}) where {T <: Number}
    # exceptions (negative pdo, etc)
    # will be added here later
    prob / sum(prob)
end


function conditional_probability(peps::PEPSNetwork, v::AbstractVector{Int},
)

    i, j = node_from_index(peps, length(v)+1)
    ∂v = generate_boundary_states(peps, v, (i, j))

    W = MPO(peps, i)
    ψ = MPS(peps, i+1)

    #L = left_env(ψ, ∂v[1:j-1])
    L = _left_env(peps, i, ∂v[1:j-1])
    #R = right_env(ψ, W, ∂v[j+2:peps.ncols+1])
    R = _right_env(peps, i, ∂v[j+2:peps.ncols+1])
    A = peps_tensor(peps, i, j)

    l, u = ∂v[j:j+1]
    M = ψ[j]
    Ã = @view A[l, u, :, :, :]
    @tensor prob[σ] := L[x] * M[x, d, y] *
                       Ã[r, d, σ] * R[y, r] order = (x, d, r, y)


    _normalize_probability(prob)
end

function bond_energy(network, u::NTuple{2, Int}, v::NTuple{2, Int}, σ::Int)
    fg_u, fg_v = network.vertex_map(u), network.vertex_map(v)
    if has_edge(network.factor_graph, fg_u, fg_v)
        pu, en, pv = get_prop.(Ref(network.factor_graph), Ref(fg_u), Ref(fg_v), (:pl, :en, :pr))
        energies = (pu * (en * pv[:, σ:σ]))'
    elseif has_edge(network.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(Ref(network.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr))
        energies = (pv[σ:σ, :] * en) * pu
    else
        energies = zeros(length(local_energy(network, u)))
    end
    vec(energies)
end

function update_energy(network::PEPSNetwork, σ::AbstractVector{Int})
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    local_energy(network, (i, j))
end
