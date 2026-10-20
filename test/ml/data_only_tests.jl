using Test, Random
include(joinpath(@__DIR__, "..", "smoke_data.jl"))

# Small, deliberately noncanonical training budgets exercise the real arm wiring.
@testset "Data-only comparison arms" begin
    X = rand(MersenneTwister(17), 3, 24) .* 2 .- 1
    y = [sum(abs2, x) for x in eachcol(X)]
    Q = X[:, 1:3] .+ 0.01
    truth = [sum(abs2, x) for x in eachcol(Q)]
    mktempdir() do directory
        id = 1
        cell = SmokeCell(:ishigami, 3, :sobol, "tiny", 24, 0.05)
        prep = PreparedSmokeCell(cell, id, X, y, y, 0.05, 0.05, 42,
            maximum(y)-minimum(y), [1,2,3], Q, truth, 24, 3,
            PREPARED_SMOKE_CELL_SCHEMA_VERSION, "test")
        materialize_fixture(prep, directory)
        path, _ = fixture_paths(directory, cell, id)
        fx = read_fixture(path)
        # PER-CELL, not global: this 24-point cell carries every arm, while the
        # two Ishigami 50^3 benchmark cells drop :rbf on its own memory
        # pre-flight. Asserting one global tuple would hide that distinction.
        @test applicable_arms(cell) == (:diona, :nn, :gp, :nsm, :rbf, :idw)
        # :rbf runs on every cell, at Float32 everywhere -- one configuration.
        big = SmokeCell(:ishigami, 3, :grid, "50³", 125_000, 0.0)
        @test applicable_arms(big) == (:diona, :nn, :gp, :nsm, :rbf, :idw)
        @test _rbf_precision_for(big.n_train) === :f32
        @test _rbf_precision_for(cell.n_train) === :f32
        if _NN_TRAINER_AVAILABLE
            # `precision` is part of the settings the arm reads, so the stand-in
            # must carry it too -- the arm names no float type of its own.
            settings = (width=4, layers=1, epochs=2, learning_rate=1e-3, seed=0,
                        minibatch_threshold=20, minibatch_batch_size=8,
                        precision="Float32")
            checkpoint = joinpath(directory, "nn.checkpoint")
            result, rows = _run_nn_arm_inprocess(fx, cell, settings, "test-nn";
                                                 checkpoint_path=checkpoint)
            @test result.n_finite_pred == 3
            @test isfinite(result.final_loss)
            @test all(r -> isnan(r.lower) && isnan(r.upper), rows)
            @test isfile(checkpoint)
            again, repeated = _run_nn_arm_inprocess(fx, cell, settings, "test-nn";
                                                    checkpoint_path=checkpoint)
            @test [r.pred for r in repeated] == [r.pred for r in rows]

            # Float64 must still run: this is a device/precision path, not a
            # Float32-only arm.
            f64 = (; settings..., precision="Float64")
            r64, rows64 = _run_nn_arm_inprocess(fx, cell, f64, "test-nn-f64")
            @test r64.n_finite_pred == 3
            @test all(isfinite(r.pred) for r in rows64)

            # CUDA, when the host has one. Everything crossing the arm boundary
            # is still Float64, and a CUDA fit is close to -- never identical
            # to -- the CPU fit at the same seed.
            if NNTrainer._LUXCUDA_LOADED &&
               (try NNTrainer.resolve_device("cuda"); true catch; false end)
                gpu, gpu_rows = _run_nn_arm_inprocess(fx, cell, settings, "test-nn";
                                                      device="cuda")
                @test gpu.n_finite_pred == 3
                @test all(r -> r.pred isa Float64, gpu_rows)
                @test [r.pred for r in gpu_rows] ≈ [r.pred for r in rows] rtol=1e-3
            else
                @info "CUDA unavailable here; the :nn cuda path is not exercised"
            end
        end
        if _GP_TRAINER_AVAILABLE
            reg = canonical_registry()
            reg.gp[3] = GPSettings(3, 2, 4, 1e-2, "Adam",
                1.0, 1.0, 0.1, 1e-6, 0, 2.0, 20_000, 1024, "Float64", "test")
            result, rows = run_gp_arm_cell(fx, reg;
                checkpoint_path=joinpath(directory,"gp.checkpoint"))
            @test result.n_finite_pred == 3
            @test result.n_finite_interval == 3
            @test all(r -> r.lower <= r.pred <= r.upper, rows)
        end
    end
    @test _NN_TRAINER_AVAILABLE || _GP_TRAINER_AVAILABLE
end
