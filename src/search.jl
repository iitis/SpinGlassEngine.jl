export
       SearchParameters,
       merge_branches,
       low_energy_spectrum,
       Solution,
       bound_solution

"""
$(TYPEDSIGNATURES)
"""
struct SearchParameters
    max_states::Int
    cut_off_prob::Real

    function SearchParameters(max_states::Int=1, cut_off_prob::Real=0.0)
        new(max_states, cut_off_prob)
    end
end

"""
$(TYPEDSIGNATURES)
"""
struct Solution
    energies::Vector{<:Real}
    states::Vector{Vector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
end

"""
$(TYPEDSIGNATURES)
"""
@inline empty_solution() = Solution([0.0], [Vector{Int}[]], [0.0], [1], -Inf)

"""
$(TYPEDSIGNATURES)
"""
function Solution(
    sol::Solution, idx::Vector{Int}, ldp::Real=sol.largest_discarded_probability
)
    Solution(
        sol.energies[idx],
        sol.states[idx],
        sol.probabilities[idx],
        sol.degeneracy[idx],
        ldp
    )
end

"""
$(TYPEDSIGNATURES)
"""
@inline function branch_energy(ctr::MpsContractor{T}, eσ::Tuple{<:Real, Vector{Int}}) where T
    eσ[begin] .+ update_energy(ctr, eσ[end])
end

"""
$(TYPEDSIGNATURES)
"""
@inline function branch_energies(ctr::MpsContractor{T}, psol::Solution) where T
    reduce(vcat, branch_energy.(Ref(ctr), zip(psol.energies, psol.states)))
end

"""
$(TYPEDSIGNATURES)
"""
function branch_states(num_states::Int, vec_states::Vector{Vector{Int}})
    states = reduce(hcat, vec_states)
    lstate, nstates = size(states)
    local_basis = collect(1:num_states)
    ns = Array{Int}(undef, lstate+1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate+1, :, :] .= reshape(local_basis, num_states, 1, 1)
    collect(eachcol(reshape(ns, lstate+1, nstates * num_states)))
end

"""
$(TYPEDSIGNATURES)
"""
function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real, Vector{Int}}) where T
    pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))
end

"""
$(TYPEDSIGNATURES)
"""
function discard_probabilities!(psol::Solution, cut_off_prob::Real)
    pcut = maximum(psol.probabilities) + log(cut_off_prob)
    if minimum(psol.probabilities) >= pcut return psol end
    local_ldp = maximum(psol.probabilities[psol.probabilities .< pcut])
    ldp = max(local_ldp, psol.largest_discarded_probability)
    Solution(psol, findall(p -> p >= pcut, psol.probabilities), ldp)
end

"""
$(TYPEDSIGNATURES)
"""
function branch_solution(psol::Solution, ctr::T) where T <: AbstractContractor
    num_states = cluster_size(ctr.peps, ctr.current_node)
    boundaries = boundary_states(ctr, psol.states, ctr.current_node)
    Solution(
        #reduce(vcat, branch_energy.(Ref(ctr), zip(psol.energies, psol.states))),
        branch_energies(ctr, psol),
        branch_states(num_states, psol.states),
        reduce(vcat, branch_probability.(Ref(ctr), zip(psol.probabilities, boundaries))),
        repeat(psol.degeneracy, inner=num_states),
        psol.largest_discarded_probability
    )
end

"""
$(TYPEDSIGNATURES)
"""
function merge_branches(ctr::MpsContractor{T}, merge_type::Symbol=:nofit) where {T}
    function _merge(psol::Solution)
        node = get(ctr.nodes_search_order, length(psol.states[1])+1, ctr.node_outside)
        boundaries = boundary_states(ctr, psol.states, node)
        _, bnd_types = SpinGlassNetworks.unique_dims(boundaries, 1)

        sorting_idx = sortperm(bnd_types)
        sorted_bnd_types = bnd_types[sorting_idx]
        nsol = Solution(psol, Vector{Int}(sorting_idx)) #TODO Vector{Int} should be rm
        energies = typeof(nsol.energies[begin])[]
        states = typeof(nsol.states[begin])[]
        probs = typeof(nsol.probabilities[begin])[]
        degeneracy = typeof(nsol.degeneracy[begin])[]

        start = 1
        bsize = size(boundaries, 1)
        while start <= bsize
            stop = start
            while stop + 1 <= bsize && sorted_bnd_types[start] == sorted_bnd_types[stop+1]
                stop = stop + 1
            end
            best_idx = argmin(@view nsol.energies[start:stop]) + start - 1

            new_degeneracy = 0
            ind_deg = []
            for i in start:stop
                if nsol.energies[i] <= nsol.energies[best_idx] + 1E-12 # this is hack for now
                    new_degeneracy += nsol.degeneracy[i]
                    push!(ind_deg, i)
                end
            end

            if merge_type == :fit
                c = Statistics.median(
                    ctr.betas[end] .* nsol.energies[start:stop] .+ nsol.probabilities[start:stop]
                )
                new_prob = -ctr.betas[end] .* nsol.energies[best_idx] .+ c
                push!(probs, new_prob)
            elseif merge_type == :nofit
                push!(probs, nsol.probabilities[best_idx])
            elseif merge_type == :python
                push!(probs, Statistics.mean(nsol.probabilities[ind_deg]))
            end

            push!(energies, nsol.energies[best_idx])
            push!(states, nsol.states[best_idx])
            push!(degeneracy, new_degeneracy)
            start = stop + 1
        end
        Solution(energies, states, probs, degeneracy, psol.largest_discarded_probability)
    end
    _merge
end

"""
$(TYPEDSIGNATURES)
"""
no_merge(partial_sol::Solution) = partial_sol

"""
$(TYPEDSIGNATURES)
"""
function bound_solution(
    psol::Solution, max_states::Int, δprob::Real, merge_strategy=no_merge
)
    psol = discard_probabilities!(merge_strategy(psol), δprob)
    if length(psol.probabilities) <= max_states return psol end
    idx = partialsortperm(psol.probabilities, 1:max_states + 1, rev=true)
    ldp = max(psol.largest_discarded_probability, psol.probabilities[idx[end]])
    Solution(psol, idx[1:max_states], ldp)
end

"""
$(TYPEDSIGNATURES)
"""
function low_energy_spectrum(
    ctr::T, sparams::SearchParameters, merge_strategy=no_merge; no_cache=false,
) where T <: AbstractContractor
    # Build all boundary mps
    @showprogress "Preprocesing: " for i ∈ ctr.peps.nrows:-1:1 dressed_mps(ctr, i) end

    # Start branch and bound search
    sol = empty_solution()
    @showprogress "Search: " for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        sol = branch_solution(sol, ctr)
        sol = bound_solution(sol, sparams.max_states, sparams.cut_off_prob, merge_strategy)
        # TODO: clear memoize cache partially
        if no_cache Memoization.empty_all_caches!() end
    end

    # Translate variable order (network --> factor graph)
    inner_perm = sortperm([
        ctr.peps.factor_graph.reverse_label_map[idx]
        for idx ∈ ctr.peps.vertex_map.(ctr.nodes_search_order)
    ])

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    sol = Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability
    )

    # Final check if states correspond energies
    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.factor_graph), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol
end
