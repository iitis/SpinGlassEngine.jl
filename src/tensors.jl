export tensor, probability

"""
$(TYPEDSIGNATURES)

"""
function tensor(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real)
    if v ∉ keys(network.tensors_map) return ones(Float64, 1, 1) end
    tensor(network, v, β, Val(network.tensors_map[v]))
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode)
    if v ∉ keys(network.tensors_map) return (1, 1) end
    size(network, v, Val(network.tensors_map[v]))
end

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    net::PEPSNetwork{T, S}, v::PEPSNode, β::Real, ::Val{:sparse_site}
) where {T <: AbstractGeometry, S}
    SiteTensor(
        probability(local_energy(net, Node(v)), β),
        projectors_site_tensor(net, Node(v))
    )
end

# TODO: This is how the engine should work
#=
function (peps::PEPSNetwork)(::Type{SparseCentralTensor}, v::PEPSNode, β::Real)
    # ...
end

ten = peps(SparseCentralTensor, v, β)
=#

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    net::PEPSNetwork{T, Dense}, v::PEPSNode, β::Real, ::Val{:site}
) where T <: AbstractGeometry
    sp = tensor(net, v, β, Val(:sparse_site))
    A = zeros(maximum.(sp.projs))
    for (σ, lexp) ∈ enumerate(sp.loc_exp)
        @inbounds A[getindex.(sp.projs, Ref(σ))...] += lexp
    end
    A
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    network::PEPSNetwork{T, S}, v::PEPSNode, ::Union{Val{:site}, Val{:sparse_site}, Val{:sparse_site_square2}}
) where {T <: AbstractGeometry, S  <: AbstractSparsity}
    maximum.(projectors_site_tensor(network, Node(v)))
end



"""
$(TYPEDSIGNATURES)

"""
function tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, β::Real, ::Val{:central_v}
)
    i = floor(Int, node.i)
    connecting_tensor(net, (i, node.j), (i+1, node.j), β)
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:central_v}
)
    i = floor(Int, node.i)
    size(interaction_energy(network, (i, node.j), (i+1, node.j)))
end

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, β::Real, ::Val{:central_h}
)
    j = floor(Int, node.j)
    connecting_tensor(net, (node.i, j), (node.i, j+1), β)
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:central_h}
)
    j = floor(Int, node.j)
    size(interaction_energy(network, (node.i, j), (node.i, j+1)))
end

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:gauge_h}
)
    Diagonal(network.gauges.data[v]) # |> Array
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:gauge_h}
)
    u = size(network.gauges.data[v], 1)
    (u, u)
end

"""
$(TYPEDSIGNATURES)

"""
function probability(en::T, β::Real) where T <: AbstractArray
    en_min = minimum(en)
    exp.(-β .* (en .- en_min))
end

"""
$(TYPEDSIGNATURES)

"""
function connecting_tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    probability(interaction_energy(net, v, w), β)
end

"""
$(TYPEDSIGNATURES)

"""
function sqrt_tensor_up(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    U, Σ, _ = svd(connecting_tensor(net, v, w, β))
    U * Diagonal(sqrt.(Σ))
end

"""
$(TYPEDSIGNATURES)

"""
function sqrt_tensor_down(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::Node, w::Node, β::Real
)
    _, Σ, V = svd(connecting_tensor(net, v, w, β))
    Diagonal(sqrt.(Σ)) * V'
end

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_up}
)
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_up(net, (i, j), (i+1, j), β)
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:sqrt_up}
)
    r, j = Node(v)
    i = floor(Int, r)
    u, d = size(interaction_energy(net, (i, j), (i+1, j)))
    (u, min(d, u))
end

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_down}
)
    r, j = Node(v)
    i = floor(Int, r)
    sqrt_tensor_down(net, (i, j), (i+1, j), β)
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    net::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, ::Val{:sqrt_down}
)
    r, j = Node(v)
    i = floor(Int, r)
    u, d = size(interaction_energy(net, (i, j), (i+1, j)))
    (min(u, d), d)
end

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_up_d}
)
    U, Σ, _ = svd(tensor(network, v, β, Val(:central_d)))
    U * Diagonal(sqrt.(Σ))
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:sqrt_up_d}
)
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (u * ũ, min(u * ũ, d * d̃))
end

"""
$(TYPEDSIGNATURES)

"""
function tensor(
    network::AbstractGibbsNetwork{Node, PEPSNode}, v::PEPSNode, β::Real, ::Val{:sqrt_down_d}
)
    _, Σ, V = svd(tensor(network, v, β, Val(:central_d)))
    Diagonal(sqrt.(Σ)) * V'
end

"""
$(TYPEDSIGNATURES)

"""
function Base.size(
    network::AbstractGibbsNetwork{Node, PEPSNode}, node::PEPSNode, ::Val{:sqrt_down_d}
)
    i, j = floor(Int, node.i), floor(Int, node.j)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (min(u * ũ, d * d̃), d * d̃)
end
