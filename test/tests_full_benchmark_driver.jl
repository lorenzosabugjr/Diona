using Test
using Dates

include(joinpath(@__DIR__, "run_full_benchmark.jl"))
using .FullBenchmarkDriver

module FullBenchmarkHarnessStub
const _SMOKE_OPTS = Dict{String,String}()
const calls = Any[]

function run_comparative_orchestrator(mode; out_dir_override, jobs_override, timeout_override=())
    push!(calls, (:comparative, mode, out_dir_override, copy(jobs_override), timeout_override))
    return (out_dir=out_dir_override, results=[:result], row_groups=[[:row]], success=true)
end

function write_problem_aggregate_artifacts(out_dir, results, row_groups)
    push!(calls, (:problem_report, out_dir, results, row_groups))
    return (csv_path="problem.csv", markdown_path="problem.md", rows=Any[])
end

function check_gate_comparative_full_audit(campaign; problem_artifacts, verbose=true)
    push!(calls, (:comparative_audit, problem_artifacts))
    return (overall=true, audit_path="audit.txt")
end


end

@testset "full benchmark driver" begin
    deterministic_out = FullBenchmarkDriver.default_out_dir(
        now=DateTime(2026, 8, 25, 12, 34, 56),
        random_bytes=UInt8[0x01, 0x23, 0xab, 0xcd],
    )
    @test endswith(deterministic_out, joinpath("results", "2026-08-25-123456-0123abcd"))

    options = FullBenchmarkDriver.parse_options([
        "--out=/tmp/diona-driver-test",
        "--jobs-diona=2",
        "--jobs-nn", "3",
        "--jobs-gp=4",
        "--blas-threads=1",
        "--device=CPU",
    ])
    @test options.out_dir == "/tmp/diona-driver-test"
    @test (options.jobs_diona, options.jobs_nn, options.jobs_gp) == (2, 3, 4)
    @test options.blas_threads == 1
    @test options.device == "cpu"
    @test FullBenchmarkDriver.parse_options(["--help"]) === nothing
    @test_throws ErrorException FullBenchmarkDriver.parse_options(["--jobs-gp=0"])
    @test_throws ErrorException FullBenchmarkDriver.parse_options(["--unknown=value"])

    mktempdir() do directory
        manifest = joinpath(directory, "run_manifest.txt")
        snapshots = Dict(
            environment.name => [(
                name="Example", uuid="00000000-0000-0000-0000-000000000000",
                version="1.2.3", direct=true, source="",
            )] for environment in FullBenchmarkDriver.ENVIRONMENTS
        )
        timings = Dict(stage => Float64(index)
                       for (index, stage) in enumerate(
                           ("bootstrap", "comparative", "report")))
        FullBenchmarkDriver.write_run_manifest(
            manifest, options, timings, snapshots;
            generated_at=DateTime(2026, 8, 25, 12, 34, 56),
            git_commit="0123456789abcdef0123456789abcdef01234567",
        )
        text = read(manifest, String)
        @test occursin("git_commit=0123456789abcdef0123456789abcdef01234567", text)
        @test occursin("julia_version=$(VERSION)", text)
        @test occursin("jobs_nn=3", text)
        @test occursin("wall_clock_report_s=3.0", text)
        @test count(==("package=Example|00000000-0000-0000-0000-000000000000|1.2.3|direct=true"),
                    split(text, '\n')) == length(FullBenchmarkDriver.ENVIRONMENTS)
    end

    mktempdir() do directory
        empty!(FullBenchmarkHarnessStub.calls)
        empty!(FullBenchmarkHarnessStub._SMOKE_OPTS)
        sequence_options = FullBenchmarkDriver.DriverOptions(directory, 5, 6, 7, 2, "cpu", nothing)
        # The validator must be told which device the campaign ran with: the :nn
        # settings hash carries it, so an audit at the wrong device finds no
        # fragments and silently re-runs a finished campaign.
        seen_device = Ref("")
        validate_audit(harness, out_dir; verbose=true, device="cpu") =
            (seen_device[] = device; false)
        result = FullBenchmarkDriver.run_benchmark(
            FullBenchmarkHarnessStub, sequence_options, Dict{String,Any}();
            bootstrap_time_s=1.25,
            comparative_validator=validate_audit,
        )
        @test isfile(result.manifest_path)
        @test seen_device[] == "cpu"
        @test FullBenchmarkHarnessStub._SMOKE_OPTS == Dict(
            "out" => directory, "blas-threads" => "2", "device" => "cpu")

        # ... and a cuda campaign forwards "cuda" to both the harness and the
        # validator, so neither can quietly assume the default.
        empty!(FullBenchmarkHarnessStub.calls)
        empty!(FullBenchmarkHarnessStub._SMOKE_OPTS)
        cuda_options = FullBenchmarkDriver.DriverOptions(directory, 5, 6, 7, 2, "cuda", nothing)
        FullBenchmarkDriver.run_benchmark(
            FullBenchmarkHarnessStub, cuda_options, Dict{String,Any}();
            bootstrap_time_s=1.25, comparative_validator=validate_audit)
        @test seen_device[] == "cuda"
        @test FullBenchmarkHarnessStub._SMOKE_OPTS["device"] == "cuda"
        comparative_call = only(call for call in FullBenchmarkHarnessStub.calls
                                if call isa Tuple && first(call) == :comparative)
        @test comparative_call[2] == :full
        @test comparative_call[4] == Dict(:diona => 5, :nn => 6, :gp => 7)
    end
end
