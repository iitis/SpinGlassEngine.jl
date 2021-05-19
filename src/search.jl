export AbstractGibbsNetwork
export low_energy_spectrum, branch_state, bound_solution
export Solution
using Memoize

struct Solution
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
    probabilities::Vector{Float64}
    largest_discarded_probability::Float64
end


empty_solution() = Solution([0.], [[]], [1.], -Inf)

function branch_state(network, σ)
    node = node_from_index(network, length(σ) + 1)
    basis = collect(1:length(local_energy(network, node)))
    vcat.(Ref(σ), basis)
end


function branch_solution(partial_sol::Solution, network::AbstractGibbsNetwork)

    if length(partial_sol.states[1]) > 6
        println("Separator 1")
        conditional_probability(network, partial_sol.states[1])
        println("Separator 2")
        conditional_probability(network, partial_sol.states[1])
        println(length(memoize_cache(peps_tensor)))

        exit(42)
    end
    Solution(
        vcat(
            [
                (en .+ update_energy(network, state))
                for (en, state) ∈ zip(partial_sol.energies, partial_sol.states)
            ]
            ...
        ),
        vcat(branch_state.(Ref(network), partial_sol.states)...),
        vcat(
            partial_sol.probabilities .* conditional_probability.(Ref(network), partial_sol.states)
            ...
        ),
        partial_sol.largest_discarded_probability
    )
end


function bound_solution(partial_sol::Solution, max_states::Int)
    if length(partial_sol.probabilities) <= max_states
        probs = vcat(partial_sol.probabilities, -Inf)
        k = length(probs)
    else
        probs = partial_sol.probabilities
        k = max_states + 1
    end

    indices = partialsortperm(probs, 1:k, rev=true)
    new_max_discarded_prob = max(partial_sol.largest_discarded_probability, probs[indices[end]])

    indices = @view indices[1:k-1]

    Solution(
        partial_sol.energies[indices],
        partial_sol.states[indices],
        partial_sol.probabilities[indices],
        new_max_discarded_prob
    )
end


#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(network::AbstractGibbsNetwork, max_states::Int)
    sol = empty_solution()

    for _ ∈ 1:nv(network_graph(network))
        sol = bound_solution(branch_solution(sol, network), max_states)
    end

    # Translate variable order (from network to factor graph)
    inner_perm = sortperm([
        factor_graph(network).reverse_label_map[idx]
        for idx ∈ network.vertex_map.(iteration_order(network))
    ])

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)

    Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.largest_discarded_probability
    )
end
