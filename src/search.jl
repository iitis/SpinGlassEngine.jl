export
       low_energy_spectrum,
       branch_state,
       bound_solution,
       merge_branches,
       Solution,
       SearchParameters,
       exact_marginal_probability,
       exact_conditional_probability

struct SearchParameters
    max_states::Int
    cut_off_prob::Real

    function SearchParameters(max_states::Int=1, cut_off_prob::Real=0.0)
        new(max_states, cut_off_prob)
    end
end
struct Solution
    energies::Vector{<:Real}
    states::Vector{Vector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
end
empty_solution() = Solution([0.0], [Vector{Int}[]], [0.0], [1], -Inf)

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

function branch_energy(ctr::MpsContractor{T}, eσ::Tuple{<:Real, Vector{Int}}) where {T, S}
    eσ[begin] .+ update_energy(ctr, eσ[end])
end

function branch_state(ctr::MpsContractor{T}, σ::Vector{Int}) where {T, S}
    vcat.(Ref(σ), collect(1:cluster_size(ctr.peps, ctr.current_node)))
end

function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real, Vector{Int}}) where T
    # if !( exact_conditional_probability(ctr, pσ[end]) ≈ conditional_probability(ctr, pσ[end]) )
    #     println(pσ[end], exact_conditional_probability(ctr, pσ[end]), conditional_probability(ctr, pσ[end]))
    # end
    pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))
end

@memoize function spectrum(factor_graph::LabelledGraph{S, T}) where {S, T}
    ver = vertices(factor_graph)
    rank = cluster_size.(Ref(factor_graph), ver)
    states = [Dict(ver .=> σ) for σ ∈ Iterators.product([1:r for r ∈ rank]...)]
    energy.(Ref(factor_graph), states), states
end

function exact_marginal_probability(
    ctr::MpsContractor{T},
    σ::Vector{Int}
) where T
    target_state = decode_state(ctr.peps, σ, true)
    energies, states = spectrum(ctr.peps.factor_graph)
    prob = exp.(-ctr.betas[end] .* energies)
    prob ./= sum(prob)
    sum(prob[findall([all(s[k] == v for (k, v) ∈ target_state) for s ∈ states])])
end

function exact_conditional_probability(ctr::MpsContractor{T}, σ::Vector{Int}) where T
    probs = exact_marginal_probability.(Ref(ctr), branch_state(ctr, σ))
    probs ./= sum(probs)
end

function discard_probabilities(psol::Solution, cut_off_prob::Real)
    pcut = maximum(psol.probabilities) + log(cut_off_prob)
    if minimum(psol.probabilities) < pcut
        local_ldp = maximum(psol.probabilities[psol.probabilities .< pcut])
        ldp = max(local_ldp, psol.largest_discarded_probability)
        psol = Solution(psol, findall(p -> p >= pcut, psol.probabilities), ldp)
    end
    psol
end

function branch_solution(psol::Solution, ctr::T) where T <: AbstractContractor
    num_states = cluster_size(ctr.peps, ctr.current_node)
    Solution(
        vcat(branch_energy.(Ref(ctr), zip(psol.energies, psol.states))...),
        vcat(branch_state.(Ref(ctr), psol.states)...),
        vcat(branch_probability.(Ref(ctr), zip(psol.probabilities, psol.states))...),
        repeat(psol.degeneracy, inner=num_states),
        psol.largest_discarded_probability)
end

function merge_branches(ctr::MpsContractor{T}) where {T}
    function _merge(psol::Solution)
        node = get(ctr.nodes_search_order, length(psol.states[1])+1, ctr.node_outside)
        # node = ctr.current_node
        boundaries = hcat(boundary_state.(Ref(ctr), psol.states, Ref(node))...)'
        _, bnd_types = SpinGlassNetworks.unique_dims(boundaries, 1)

        sorting_idx = sortperm(bnd_types)
        sorted_bnd_types = bnd_types[sorting_idx]
        nsol = Solution(psol, Vector{Int}(sorting_idx)) #TODO Vector{Int} should be rm
        @infiltrate
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
            
            #b = -ctr.betas[end] .* nsol.energies[start:stop] .- nsol.probabilities[start:stop] # this should be const
            #c = Statistics.median(ctr.betas[end] .* nsol.energies[start:stop] .+ nsol.probabilities[start:stop])
            #new_prob = -ctr.betas[end] .* nsol.energies[best_idx] .+ c ## base probs on all states with the same boundary
            # using fit to log(prob) = - beta * eng + a0

            new_degeneracy = 0
            for i in start:stop
                if nsol.energies[i] <= nsol.energies[best_idx] + 1E-12 
                    new_degeneracy += nsol.degeneracy[i]
                end
            end

            push!(energies, nsol.energies[best_idx])
            push!(states, nsol.states[best_idx])
            #push!(probs, new_prob)
            push!(probs, nsol.probabilities[best_idx]) 
            push!(degeneracy, new_degeneracy)
            start = stop + 1
        end
        @infiltrate
        Solution(energies, states, probs, degeneracy, psol.largest_discarded_probability)
    end
    _merge
end
no_merge(partial_sol::Solution) = partial_sol

function bound_solution(psol::Solution, max_states::Int, δprob::Real, merge_strategy=no_merge)
    psol = discard_probabilities(merge_strategy(psol), δprob)
    if length(psol.probabilities) > max_states
        idx = partialsortperm(psol.probabilities, 1:max_states + 1, rev=true)
        ldp = max(psol.largest_discarded_probability, psol.probabilities[idx[end]])
        psol = Solution(psol, idx[1:max_states], ldp)
    end
    psol
end

# function bound_solution(psol::Solution, max_states::Int, merge_strategy=no_merge)
#     if length(psol.probabilities) <= max_states
#         probs = vcat(psol.probabilities, -Inf)
#         k = length(probs)
#     else
#         probs = psol.probabilities
#         k = max_states + 1
#     end
#     idx = partialsortperm(probs, 1:k, rev=true)
#     ldp = max(psol.largest_discarded_probability, probs[idx[end]])
#     merge_strategy(Solution(psol, idx[1:k-1], ldp))
# end

#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(
    ctr::T, sparams::SearchParameters, merge_strategy=no_merge
) where T <: AbstractContractor
    # Build all boundary mps
    @showprogress "Preprocesing: " for i ∈ ctr.peps.nrows:-1:1 dressed_mps(ctr, i) end

    # Start branch and bound search
    sol = empty_solution()
    @showprogress "Search: " for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        sol = branch_solution(sol, ctr)
        sol = bound_solution(sol, sparams.max_states, sparams.cut_off_prob, merge_strategy)
        # TODO: clear memoize cache
    end

    # Translate variable order (from network to factor graph)
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

    # Final check if energies and states much each other
    @assert sol.energies ≈ energy.(
        Ref(ctr.peps.factor_graph), decode_state.(Ref(ctr.peps), sol.states)
    )
    sol
end
