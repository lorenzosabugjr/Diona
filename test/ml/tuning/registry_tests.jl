using Test
include(joinpath(@__DIR__, "..", "common", "registry.jl"))
using .ComparatorRegistry

@testset "Data-only comparison registry" begin
    reg = canonical_registry()
    @test reg.canonical
    @test registry_hash(reg) == registry_hash(canonical_registry())
    @test canonical_serialize(Dict(:b => 2, :a => 1)) ==
          canonical_serialize(Dict(:a => 1, :b => 2))
    for dim in benchmark_dims()
        nn = settings_for(reg, dim, :nn)
        gp = settings_for(reg, dim, :gp)
        @test nn.width == 64 && nn.layers == ceil(Int, dim/2) + 1
        @test nn.epochs == 25_000 && nn.seed == 0
        @test gp.epochs == 500 && gp.inducing_points == 100dim
    end
    # Precision is registry-driven: NNTrainer reads this field instead of
    # naming a float type of its own, so it must be a supported one.
    for dim in benchmark_dims()
        nn = settings_for(reg, dim, :nn)
        @test nn.precision == "Float32"
        @test float_type(nn) === Float32
        @test float_type(settings_for(reg, dim, :gp)) === Float64
    end
    @test float_type("Float64") === Float64
    @test_throws ErrorException float_type("Float16")

    @test SUPPORTED_DEVICES == ("cpu", "cuda")
    @test normalize_device("CUDA") == "cuda"
    @test normalize_device(" Cpu ") == "cpu"
    for bogus in ("gpu", "rocm", "cuda:0", "")
        @test_throws ErrorException normalize_device(bogus)
    end

    # The device is part of the :nn settings hash, so a CPU fragment and a
    # CUDA fragment key to different paths and can never be silently resumed
    # from or compared against one another.
    let nn3 = settings_for(reg, 3, :nn), nn4 = settings_for(reg, 4, :nn)
        @test nn_settings_hash(nn3, "cpu") != nn_settings_hash(nn3, "cuda")
        @test nn_settings_hash(nn3, "cpu") == nn_settings_hash(nn3, "CPU")
        @test nn_settings_hash(nn3, "cuda") != nn_settings_hash(nn4, "cuda")
        @test_throws ErrorException nn_settings_hash(nn3, "gpu")
    end

    @test_throws ErrorException settings_for(reg, 3, :pinn)
    @test_throws ErrorException settings_for(reg, 3, :diona)
    @test_throws ErrorException settings_for(reg, maximum(benchmark_dims()) + 1, :nn)
    withenv("DIONA_REGISTRY_NN_EPOCHS" => "2", "DIONA_REGISTRY_GP_EPOCHS" => "3") do
        changed = registry_with_env_overrides(reg)
        @test !changed.canonical
        @test registry_hash(changed) != registry_hash(reg)
        @test settings_for(changed, 3, :nn).epochs == 2
        @test settings_for(changed, 3, :gp).epochs == 3
        @test settings_for(reg, 3, :nn).epochs == 25_000
    end
end
