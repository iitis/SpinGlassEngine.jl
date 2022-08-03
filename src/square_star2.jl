export SquareStar2 # update_reduced_env_right

struct SquareStar2{T <: AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)
"""
function SquareStar2(m::Int, n::Int)
    lg = Square2(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j, 1), (i+1, j+1, 1))
        add_edge!(lg, (i, j, 1), (i+1, j+1, 2))
        add_edge!(lg, (i, j, 2), (i+1, j+1, 1))
        add_edge!(lg, (i, j, 2), (i+1, j+1, 2))
        add_edge!(lg, (i+1, j, 1), (i, j+1, 1))
        add_edge!(lg, (i+1, j, 1), (i, j+1, 2))
        add_edge!(lg, (i+1, j, 2), (i, j+1, 1))
        add_edge!(lg, (i+1, j, 2), (i, j+1, 2))
    end
    lg
end

"""
$(TYPEDSIGNATURES)
"""
Virtual2(::Type{Dense}) = :virtual2

"""
$(TYPEDSIGNATURES)
"""
Virtual2(::Type{Sparse}) = :sparse_virtual2

"""
$(TYPEDSIGNATURES)
"""
function tensor_map(
    ::Type{SquareStar2{T}}, ::Type{S}, nrows::Int, ncols::Int
) where {T <: Union{EnergyGauges, GaugesEnergy}, S <: AbstractSparsity}
    map = Dict{PEPSNode, Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site2(S),
            PEPSNode(i, j - 1//2) => Virtual2(S),
            PEPSNode(i + 1//2, j) => :central_v2
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1   # why from 0?
        push!(map, PEPSNode(i + 1//2, j + 1//2) => :central_d2)
    end
    map
end

# """
# $(TYPEDSIGNATURES)
# """
# function tensor_map(
#     ::Type{SquareStar2{T}}, ::Type{S}, nrows::Int, ncols::Int
# ) where {T <: EngGaugesEng, S <: AbstractSparsity}
#     map = Dict{PEPSNode, Symbol}()

#     for i ∈ 1:nrows, j ∈ 1:ncols
#         push!(
#             map,
#             PEPSNode(i, j) => site2(S),
#             PEPSNode(i, j - 1//2) => Virtual2(S),
#             PEPSNode(i + 1//5, j) => :sqrt_up,
#             PEPSNode(i + 4//5, j) => :sqrt_down
#         )
#     end

#     for i ∈ 1:nrows-1, j ∈ 0:ncols-1
#         push!(
#             map,
#             PEPSNode(i + 1//5, j + 1//2) => :sqrt_up_d,
#             PEPSNode(i + 4//5, j + 1//2) => :sqrt_down_d
#         )
#     end
#     map
# end

"""
$(TYPEDSIGNATURES)
"""
function gauges_list(::Type{SquareStar2{T}}, nrows::Int, ncols::Int) where T <: GaugesEnergy
    [
        GaugeInfo(
            (PEPSNode(i + 1//6, j), PEPSNode(i + 2//6, j)),
            PEPSNode(i + 1//2, j),
            1,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

"""
$(TYPEDSIGNATURES)
"""
function gauges_list(::Type{SquareStar2{T}}, nrows::Int, ncols::Int) where T <: EnergyGauges
    [
        GaugeInfo(
            (PEPSNode(i + 4//6, j), PEPSNode(i + 5//6, j)),
            PEPSNode(i + 1//2, j),
            2,
            :gauge_h
        )
        for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

# """
# $(TYPEDSIGNATURES)
# """
# function gauges_list(::Type{SquareStar2{T}}, nrows::Int, ncols::Int) where T <: EngGaugesEng
#     [
#         GaugeInfo(
#             (PEPSNode(i + 2//5, j), PEPSNode(i + 3//5, j)),
#             PEPSNode(i + 1//5, j),
#             2,
#             :gauge_h
#         )
#         for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
#     ]
# end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareStar2 geometry with the EnergyGauges layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar2{EnergyGauges}
    MpoLayers(
        Dict(site(i) => (-1//6, 0, 3//6, 4//6) for i ∈ 1//2:1//2:ncols),  # for now removes gauges
        Dict(site(i) => (3//6, 4//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareStar2 geometry with the GaugesEnergy layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar2{GaugesEnergy}
    MpoLayers(
        Dict(site(i) => (-4//6, -1//2, 0, 1//6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1//6,) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3//6, 0) for i ∈ 1//2:1//2:ncols)
    )
end

# """
# $(TYPEDSIGNATURES)
# Defines the MPO layers for the SquareStar2 geometry with the EngGaugesEng layout.
# """
# function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar2{EngGaugesEng}
#     MpoLayers(
#         Dict(site(i) => (-2//5, -1//5, 0, 1//5, 2//5) for i ∈ 1//2:1//2:ncols),
#         Dict(site(i) => (1//5, 2//5) for i ∈ 1//2:1//2:ncols),
#         Dict(site(i) => (-4//5, -1//5, 0) for i ∈ 1//2:1//2:ncols)
#     )
# end

"""
$(TYPEDSIGNATURES)
"""
# TODO: rewrite this using brodcasting if possible
function conditional_probability(  # TODO
    ::Type{T}, ctr::MpsContractor{S}, ∂v::Vector{Int}
) where {T <: SquareStar2, S}
    indβ, β = length(ctr.betas), last(ctr.betas)
    i, j = ctr.current_node

    L = left_env(ctr, i, ∂v[1:2*j-2], indβ)
    R = right_env(ctr, i, ∂v[(2*j+3):2*ctr.peps.ncols+2], indβ)
    ψ = dressed_mps(ctr, i, indβ)

    MX, M = ψ[j-1//2], ψ[j]
    @tensor LMX[y, z] := L[x] * MX[x, y, z]

    v = ((i, j-1), (i-1, j-1), (i-1, j))
    @nexprs 3 k->(
        en_k = projected_energy(ctr.peps, (i, j), v[k], ∂v[2*j-1+k]);
    )
    probs = probability(local_energy(ctr.peps, (i, j)) .+ en_1 .+ en_2 .+ en_3, β)

    p_rb = projector(ctr.peps, (i, j), (i+1, j-1))
    pr = projector(ctr.peps, (i, j), @ntuple 3 k->(i+2-k, j+1))
    pd = projector(ctr.peps, (i, j), (i+1, j))

    @cast lmx2[b, c, d] := LMX[(b, c), d] (c ∈ 1:maximum(p_rb))

    for σ ∈ 1:length(probs)
        lmx = @inbounds lmx2[∂v[2*j-1], p_rb[σ], :]
        m = @inbounds M[:, pd[σ], :]
        r = @inbounds R[:, pr[σ]]
        @inbounds probs[σ] *= (lmx' * m * r)[]
    end

    push!(ctr.statistics, ((i, j), ∂v) => error_measure(probs))
    normalize_probability(probs)
end


# """
# $(TYPEDSIGNATURES)
# """
# function update_reduced_env_right(  # TODO
#     K::Array{T, 1},
#     RE::Array{T, 2},
#     M::SparseVirtualTensor,
#     B::Array{T, 3}
# ) where T <: Real
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#     @cast K2[t1, t2] := K[(t1, t2)] (t1 ∈ 1:maximum(p_rt))
#     @tensor REB[x, y1, y2, β] := B4[x, y1, y2, α] * RE[α, β]
#     R = zeros(size(B, 1), length(p_l))
#     for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#         @inbounds R[:, l] += (K2[p_rt[r], p_lt[l]] .* h[p_l[l], p_r[r]]) .*
#                               REB[:, p_lb[l], p_rb[r], r]
#     end
#     R
# end

"""
$(TYPEDSIGNATURES)
"""
function nodes_search_order_Mps(peps::PEPSNetwork{T, S}) where {T <: SquareStar2, S}
    ([(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2], (peps.nrows+1, 1, 1))
end

"""
$(TYPEDSIGNATURES)
"""
function boundary(::Type{T}, ctr::MpsContractor{S}, node::Node) where {T <: SquareStar2, S}
    i, j = node  # todo
    vcat(
        [
            [((i, k-1), (i+1, k), (i, k), (i+1, k-1)), ((i, k), (i+1, k))]
            for k ∈ 1:(j-1)
        ]...,
        ((i, j-1), (i+1, j)),
        ((i, j-1), (i, j)),
        ((i-1, j-1), (i, j)),
        ((i-1, j), (i, j)),
        [
            [((i-1, k-1), (i, k), (i-1, k), (i, k-1)), ((i-1, k), (i, k))]
            for k ∈ (j+1):ctr.peps.ncols
        ]...
    )
end

"""
$(TYPEDSIGNATURES)
"""
function update_energy(
    ::Type{T}, ctr::MpsContractor{S}, σ::Vector{Int}
) where {T <: SquareStar2, S}
    net = ctr.peps
    i, j, k = ctr.current_node

    en = local_energy(net, (i, j, k))
    for v ∈ ((i, j-1, 1), (i, j-1, 2), (i-1, j, 1), (i-1, j, 2), (i-1, j-1, 1), (i-1, j-1, 2), (i-1, j+1, 1), (i-1, j+1, 2))
        en += bond_energy(net, (i, j), v, local_state_for_node(ctr, σ, v))
    end
    if k != 2 return en end
    en += bond_energy(net, (i, j, k), (i, j, 1), local_state_for_node(ctr, σ, (i, j, 1)))  # here k=2
    en
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{T, Dense}, node::PEPSNode, β::Real, ::Val{:central_d2}
) where {T <: AbstractGeometry}
    i, j = floor(Int, node.i), floor(Int, node.j)
    T_1 = dense_central_tensor(SparseCentralTensor(net, β, (i, j), (i+1, j+1)))
    T_2 = dense_central_tensor(SparseCentralTensor(net, β, (i, j+1), (i+1, j)))
    @cast A[(u, uu), (d, dd)] := T_1[u, d] * T_2[uu, dd]
    A
end


"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{T, Sparse}, node::PEPSNode, β::Real, ::Val{:central_d2}
) where {T <: AbstractGeometry}
    i, j = floor(Int, node.i), floor(Int, node.j)
    T_1 = SparseCentralTensor(net, β, (i, j), (i+1, j+1))
    T_2 = SparseCentralTensor(net, β, (i, j+1), (i+1, j))
    # ADD new structure to cover this
end

"""
$(TYPEDSIGNATURES)
"""
function Base.size(
    net::PEPSNetwork{SquareStar2{T}, S}, node::PEPSNode, ::Val{:central_d2}
) where {T <: AbstractTensorsLayout, S <: AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    s_1 =  SparseCentralTensor_size(net, (i, j), (i+1, j+1))
    s_2 =  SparseCentralTensor_size(net, (i, j+1), (i+1, j))
    (s_1[1] * s_2[1], s_1[2] * s_2[2])
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(  #TODO
    net::PEPSNetwork{SquareStar2{T}, S}, node::PEPSNode, β::Real, ::Val{:sparse_virtual2}
) where {T <: AbstractTensorsLayout, S <: Union{Sparse, Dense}}
    v = Node(node)
    i, j = node.i, floor(Int, node.j)

    p_l = outer_projector(
        (projector(net, (i, j, k), ((i, j-1, 1), (i, j-1, 2))) for k ∈ 1:2)...)
    p_r = outer_projector(
        (projector(net, (i, j, k), ((i, j+1, 1), (i, j+1, 2))) for k ∈ 1:2)...)
    p_lb = outer_projector(
        (projector(net, (i, j, k), ((i+1, j-1, 1), (i+1, j-1, 2))) for k ∈ 1:2)...)
    p_lt = outer_projector(
        (projector(net, (i, j, k), ((i-1, j-1, 1), (i-1, j-1, 2))) for k ∈ 1:2)...)
    p_rb = outer_projector(
        (projector(net, (i, j, k), ((i+1, j+1, 1), (i+1, j+1, 2))) for k ∈ 1:2)...)
    p_rt = outer_projector(
        (projector(net, (i, j, k), ((i-1, j+1, 1), (i-1, j+1, 2))) for k ∈ 1:2)...)
    p_l = last(fuse_projectors((p_lb, p_l, p_lt)))
    p_r = last(fuse_projectors((p_rb, p_r, p_rt)))

    SparseVirtualTensor(
       SparseCentralTensor(net, β, floor.(Int, v), ceil.(Int, v)),
       vec.((p_l..., p_r...))
    )
end

"""
$(TYPEDSIGNATURES)
"""
function tensor(
    net::PEPSNetwork{T, Dense}, node::PEPSNode, β::Real, ::Val{:virtual2}
) where{T <: AbstractGeometry}
    sp = tensor(net, node, β, Val(:sparse_virtual2))
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = sp.projs

    dense_con = dense_central_tensor(sp.con)

    A = zeros(
        eltype(dense_con),
        length(p_l), maximum.((p_rt, p_lt))..., length(p_r), maximum.((p_lb, p_rb))...
    )
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        @inbounds A[l, p_rt[r], p_lt[l], r, p_lb[l], p_rb[r]] = dense_con[p_l[l], p_r[r]]
    end
    @cast B[l, (uu, u), r, (dd, d)] := A[l, uu, u, r, dd, d]
    B
end



"""
$(TYPEDSIGNATURES)
"""
function projectors_site_tensor(
    net::PEPSNetwork{T, S}, vertex::Node
) where {T <: SquareStar2, S}
    i, j = vertex
    pl = outer_projector(
        (projector(net, (i, j, k), ((i, j-1, 1), (i, j-1, 2))) for k ∈ 1:2)...)
    pt = outer_projector(
        (projector(net, (i, j, k), ((i-1, j, 1), (i-1, j, 2))) for k ∈ 1:2)...)
    pr = outer_projector(
        (projector(net, (i, j, k), ((i, j+1, 1), (i, j+1, 2))) for k ∈ 1:2)...)
    pb = outer_projector(
        (projector(net, (i, j, k), ((i+1, j, 1), (i+1, j, 2))) for k ∈ 1:2)...)
    plb = outer_projector(
        (projector(net, (i, j, k), ((i+1, j-1, 1), (i+1, j-1, 2))) for k ∈ 1:2)...)
    plt = outer_projector(
        (projector(net, (i, j, k), ((i-1, j-1, 1), (i-1, j-1, 2))) for k ∈ 1:2)...)
    prb = outer_projector(
        (projector(net, (i, j, k), ((i+1, j+1, 1), (i+1, j+1, 2))) for k ∈ 1:2)...)
    prt = outer_projector(
        (projector(net, (i, j, k), ((i-1, j+1, 1), (i-1, j+1, 2))) for k ∈ 1:2)...)
    plf = first(fuse_projectors((plb, pl, plt)))
    prf = first(fuse_projectors((prb, pr, prt)))
    (plf, pt, prf, pb)
end
