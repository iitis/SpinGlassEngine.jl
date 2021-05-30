using LabelledGraphs

export
    AbstractGibbsNetwork,
    network_graph,
    vertex_map,
    projectors,
    local_energy,
    interaction_energy,
    build_tensor,
    generate_boundary_states,
    local_state_for_node,
    iteration_order


# S: type of the vertex of network
# T: type of the vertex of underlying factor graph
abstract type AbstractGibbsNetwork{S, T} end


struct NotImplementedError{M} <: Exception
    m::M
    NotImplementedError(m::M) where {M} = new{M}(m)
end

not_implmented(name) = throw(NotImplementedError(name))


factor_graph(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.factor_graph

network_graph(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.network_graph

vertex_map(network::AbstractGibbsNetwork{S, T}) where {S, T} = network.vertex_map

projectors(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T} = not_implmented("projectors")

boundary_at_splitting_node(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T} = not_implmented("boundary_at_splitting_node")

node_index(network::AbstractGibbsNetwork{S, T}, node::S) where {S, T} = not_implmented("node_index")

iteration_order(network::AbstractGibbsNetwork{S, T}) where {S, T} = not_implemented("iteration_order")

update_energy(network::AbstractGibbsNetwork{S, T}, σ::Vector{Int}) where {S, T} = not_implmented("update_energy")

conditional_probability(network::AbstractGibbsNetwork{S, T}, v::Vector{Int}) where {S, T} = not_implemented("conditional_probability")


function projector(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    fg = factor_graph(network)
    vmap = vertex_map(network)
    fg_v, fg_w = vmap(v), vmap(w)

    if has_edge(fg, fg_w, fg_v)
        get_prop(fg, fg_w, fg_v, :pr)'
    elseif has_edge(fg, fg_v, fg_w)
        get_prop(fg, fg_v, fg_w, :pl)
    else
        loc_dim = length(local_energy(network, v))
        ones(loc_dim, 1)
    end
end


function spectrum(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    get_prop(factor_graph(network), vertex_map(network)(vertex), :spectrum)
end


function local_energy(network::AbstractGibbsNetwork{S, T}, vertex::S) where {S, T}
    spectrum(network, vertex).energies
end


function interaction_energy(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    fg = factor_graph(network)
    vmap = vertex_map(network)
    fg_v, fg_w = vmap(v), vmap(w)
    if has_edge(fg, fg_w, fg_v)
        en = get_prop(fg, fg_w, fg_v, :en)'
    elseif has_edge(fg, fg_v, fg_w)
        en = get_prop(fg, fg_v, fg_w, :en)
    else
        en = zeros(1, 1)
    end
    en
end


@memoize function build_tensor(network::AbstractGibbsNetwork{S, T}, v::S) where {S, T}
    # TODO: does this require full network, or can we pass only fg?
    loc_exp = exp.(-network.β .* local_energy(network, v))

    projs = projectors(network, v)
    dim = zeros(Int, length(projs))
    @cast A[_, i] := loc_exp[i]

    for (j, pv) ∈ enumerate(projs)
        @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
        dim[j] = size(pv, 2)
    end
    reshape(A, dim..., :)
end


@memoize function build_tensor(network::AbstractGibbsNetwork{S, T}, v::S, w::S) where {S, T}
    en = interaction_energy(network, v, w)
    exp.(-network.β .* (en .- minimum(en)))
end


function generate_boundary_state(network::AbstractGibbsNetwork{S, T}, v::S, w::S, state::Int) where {S, T}
    if v ∉ vertices(network.network_graph) return 1 end
    loc_dim = length(local_energy(network, v))
    pv = projector(network, v, w)
    findfirst(x -> x > 0, pv[state, :])
end


function generate_boundary_states(
    network::AbstractGibbsNetwork,
    σ::AbstractVector{Int},
    node::S
) where {S, T}
    [
        generate_boundary_state(network, v, w, local_state_for_node(network, σ, v))
        for (v, w) ∈ boundary_at_splitting_node(network, node)
    ]
end


function local_state_for_node(
    network::AbstractGibbsNetwork{S, T},
    σ::AbstractVector{Int},
    w::S
) where {S, T}
    k = node_index(network, w)
    0 < k <= length(σ) ? σ[k] : 1
end


function is_compatible(factor_graph, network_graph)
    all(
        has_edge(network_graph, src(edge), dst(edge))
        for edge ∈ edges(factor_graph)
    )
end
