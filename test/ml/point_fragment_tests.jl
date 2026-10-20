using Test

include(joinpath(@__DIR__, "..", "smoke_data.jl"))

@testset "comparative Arrow point fragments" begin
    cell_id = 1
    cell = enumerate_cells()[cell_id]
    query_ids = [1, 2, 3]
    x_query = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
    truth = [1.0, 2.0, 3.0]
    fixture = (
        name=String(cell.name), dim=cell.dim,
        layout=String(cell.layout), tier_label=cell.tier_label,
        n_train_nominal=cell.n_train, noise_level=cell.noise_level,
        cell_id=cell_id, n_query=length(query_ids), query_ids=query_ids,
        x_query=x_query, y_query_truth=truth,
        schema_version="point-fragment-test-v1", fixture_sha256="f"^64,
        noise_seed=42, n_train_actual=cell.n_train,
    )
    output = mktempdir()
    results = ArmCellResult[]
    row_groups = Vector{ArmQueryRow}[]

    # Every arm the cell actually carries, not a frozen three. `applicable_arms`
    # is per-cell, and `write_comparative_artifacts` below defaults its
    # expectation to exactly that set, so a hard-coded subset would be reported
    # as a missing-arm campaign.
    arms = applicable_arms(cell)
    for (k, arm) in enumerate(arms)
        offset = 0.1 * (k - 2)
        # Interval bounds follow the arm's DECLARED interval kind. Handing an
        # arm finite bounds where `comparative_interval_kind` says `:none`
        # would make the fixture misrepresent the arm.
        has_interval = comparative_interval_kind(arm) !== :none
        rows = ArmQueryRow[
            ArmQueryRow(cell_id, i, arm, Vector{Float64}(x_query[:, i]), truth[i],
                        truth[i] + offset,
                        has_interval ? truth[i] - 0.5 : NaN,
                        has_interval ? truth[i] + 0.5 : NaN,
                        :COMPLETED, 0.01)
            for i in query_ids
        ]
        result = arm_cell_result_from_rows(rows;
            cell_id=cell_id, cell=cell, arm=arm, model_type=_comparative_model_type(arm),
            fixture_sha256=fixture.fixture_sha256,
            settings_hash=bytes2hex(sha256("point-fragment-test-" * String(arm))),
            model_seed=0, noise_seed=fixture.noise_seed, n_train=fixture.n_train_actual,
            fit_time_s=(arm == :diona ? 0.0 : 1.0),
            final_loss=(arm == :diona ? NaN : 0.01))
        paths = comparative_fragment_paths(output, cell, cell_id, arm,
                                           fixture.fixture_sha256, result.settings_hash)
        @test dirname(paths.summary) == dirname(paths.points) == dirname(paths.checkpoint)
        @test replace(paths.summary, ".summary.csv" => "") ==
              replace(paths.points, ".points.arrow" => "") ==
              replace(paths.checkpoint, ".checkpoint.jls" => "")
        write_comparative_arm_fragments(output, "point-fragment-test", fixture,
            result, rows; code_provenance="c"^64, worker_project="/tmp/project",
            paths=paths)
        expected = (
            paths=paths, cell_id=cell_id, arm=arm,
            fixture_schema=fixture.schema_version,
            fixture_sha256=fixture.fixture_sha256,
            settings_hash=result.settings_hash, code_provenance="c"^64,
            worker_project="/tmp/project", fx=fixture,
        )
        roundtrip = read_comparative_arm_fragment(expected)
        @test length(roundtrip.rows) == length(rows)
        @test all(isequal(getfield(roundtrip.rows[i], field), getfield(rows[i], field))
                  for i in eachindex(rows) for field in fieldnames(ArmQueryRow))
        push!(results, result)
        push!(row_groups, rows)
    end

    artifacts = write_comparative_artifacts(output, results, row_groups)
    @test isfile(artifacts.summary_path)
    @test !isfile(joinpath(output, "comparative_points.csv"))
    @test length(read_point_fragments(artifacts.points_dir)) == length(arms) * length(query_ids)
end

@testset "Data-only fixture transport and comparison scope" begin
    @test check_gate_cell_enumeration(; verbose=false)
    @test length(enumerate_cells()) == 42
    @test length(_comparative_representative_ids(enumerate_cells())) == 2
    @test length(_comparative_gate_ids(enumerate_cells())) == 3
    @test length(_fast_smoke_cell_ids(enumerate_cells())) == 2
    # 36 cells x 6 arms, with NO exclusion: :rbf fits every cell once the two
    # Ishigami 50^3 cells (n = 125,000) are fitted at Float32 -- 87.9 GiB against
    # the 96 GiB pre-flight, versus 176.1 GiB at Float64. Stated as the arithmetic
    # so a new arm or a new tier moves the expected number visibly.
    @test sum(length(applicable_arms(c)) for c in enumerate_cells()) == 42 * 6
    @test all(:rbf in applicable_arms(c) for c in enumerate_cells())
    # ONE precision on every cell -- no per-cell split to drift.
    @test all(_rbf_precision_for(c.n_train) === :f32 for c in enumerate_cells())
    @test all(c.n_train == 125_000 for c in enumerate_cells()[[35, 36]])
    @test check_gate_reporting_correctness(; verbose=false)
    X = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
    y = [1.0, 2.0, 3.0]
    cell = SmokeCell(:ishigami, 3, :sobol, "tiny", 3, 0.05)
    prep = PreparedSmokeCell(cell, 1, X, y, y .+ 0.01, 0.05, 0.1, 42,
        2.0, [1,2,3], X, y, 3, 3, PREPARED_SMOKE_CELL_SCHEMA_VERSION, "test")
    mktempdir() do directory
        materialize_fixture(prep, directory)
        path, _ = fixture_paths(directory, cell, 1)
        fx = read_fixture(path)
        @test fx.schema_version == "3.0.0"
        @test fx.x_train == X && fx.y_train_noisy == y .+ 0.01
        @test fx.query_ids == [1,2,3] && fx.y_query_truth == y
        @test _comparative_fixture_matches(prep, fx)
        @test prepared_cell_from_fixture(fx).X_train == X
        @test !hasproperty(fx, :collocation_points)
        bytes = read(path)
        bytes[end] ⊻= 0x01
        write(path, bytes)
        @test_throws FixtureError read_fixture(path)
    end
end

@testset "Complete data-only campaign audit" begin
    mktempdir() do directory
        reg = canonical_registry()
        results, row_groups, expectations = ArmCellResult[], Vector{ArmQueryRow}[], Any[]
        payloads = Dict{Tuple{Int,Symbol},Any}()
        interval_kind_ok = Bool[]
        for (id, cell) in enumerate(enumerate_cells())
            fx = (name=String(cell.name), dim=cell.dim,
                layout=String(cell.layout), tier_label=cell.tier_label,
                n_train_nominal=cell.n_train, noise_level=cell.noise_level,
                cell_id=id, n_query=1, query_ids=[1], x_query=zeros(cell.dim,1),
                y_query_truth=[1.0], schema_version=PREPARED_SMOKE_CELL_SCHEMA_VERSION,
                fixture_sha256=bytes2hex(sha256(string(id))), noise_seed=42,
                n_train_actual=cell.n_train)
            for arm in applicable_arms(cell)
                expected = _comparative_expected(cell, id, arm, fx, "synthetic", reg, directory)
                # Bounds follow the arm's DECLARED interval kind: :nn, :rbf and
                # :idw produce no interval, so a finite bound for them here
                # would be a silent misrepresentation in the fixture.
                has_interval = comparative_interval_kind(arm) !== :none
                rows = [ArmQueryRow(id, 1, arm, zeros(cell.dim), 1.0, 1.0,
                    has_interval ? 0.5 : NaN, has_interval ? 1.5 : NaN, :COMPLETED, 0.01)]
                push!(interval_kind_ok,
                      all(r -> (isfinite(r.lower) && isfinite(r.upper)) == has_interval, rows))
                result = arm_cell_result_from_rows(rows; cell_id=id, cell=cell, arm=arm,
                    model_type=_comparative_model_type(arm), fixture_sha256=fx.fixture_sha256,
                    settings_hash=expected.settings_hash, model_seed=0, noise_seed=42,
                    n_train=cell.n_train, fit_time_s=0.0, final_loss=0.0)
                write_comparative_arm_fragments(directory, "test", fx, result, rows;
                    worker_project=expected.worker_project)
                payloads[(id,arm)] = read_comparative_arm_fragment(expected)
                push!(expectations, expected); push!(results, result); push!(row_groups, rows)
            end
        end
        # Every synthesised arm-cell agrees with its own comparative_interval_kind.
        @test length(interval_kind_ok) == 42 * 6
        @test all(interval_kind_ok)
        artifacts = write_comparative_artifacts(directory, results, row_groups)
        outcomes_path = joinpath(directory, "outcomes.csv")
        write(outcomes_path, "synthetic campaign\n")
        campaign = (out_dir=directory, expectations=expectations, payloads=payloads,
            results=results, row_groups=row_groups, artifacts=artifacts, outcomes_path=outcomes_path)
        @test check_gate_comparative_full_audit(campaign; verbose=false).overall
        incomplete = merge(campaign, (expectations=expectations[2:end],))
        @test !check_gate_comparative_full_audit(incomplete; verbose=false).overall
    end
end
