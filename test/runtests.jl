using Test
using Diona

@testset "Diona" begin
    include("test_triple.jl")
    include("test_constraints.jl")
    include("test_neighbors.jl")
    include("test_lp.jl")
    include("test_noise.jl")
    include("test_inference.jl")
end
