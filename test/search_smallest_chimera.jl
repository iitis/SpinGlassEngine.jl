@testset "Chimera-like instance has the correct low energy spectrum" begin
    m = 3
    n = 1
    t = 1
    L = n * m * t

    β = 1.0

    bond_dim = 16
    num_states = 2^8

    instance = "$(@__DIR__)/instances/pathological/chim_$(n)_$(m)_$(t).txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        spectrum=full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 4)
    search_params = SearchParameters(num_states, 0.0)

    exact_energies = [-2.6, -1.1, -0.6, -0.4, -0.4, 1.1, 1.9, 2.1]

    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, ) # MPSAnnealing
        for Layout ∈ (EnergyGauges, ) #GaugesEnergy, EngGaugesEng
            for transform ∈ all_lattice_transformations[[1, 2, 3, 4, 5, 6, 7, 8]]

                network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                contractor = MpsContractor{Strategy}(network, [β/8., β/4., β/2., β], params)
                sol = low_energy_spectrum(contractor, search_params)

                @test sol.energies ≈ exact_energies
                ig_states = decode_factor_graph_state.(Ref(fg), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                norm_prob = exp.(sol.probabilities)
                exact_norm_prob = exp.(-β .* (sol.energies .- sol.energies[1]))
                exact_norm_prob = exact_norm_prob./sum(exact_norm_prob)
                @test norm_prob ≈ exact_norm_prob
                clear_memoize_cache()
            end
        end
    end
end
