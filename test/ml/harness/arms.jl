# Per-arm runners: diona, nn, gp, nsm, rbf, idw, and the dispatch over them.
#
# INDEX ONLY -- the code lives under arms/. This is a plain `include`d FRAGMENT,
# not a module: every name must stay top-level in the including scope, because
# test/run_full_benchmark.jl reaches into the harness via getproperty(harness, :name)
# and test/benchmark_data.jl mutates _SMOKE_OPTS directly. The include sits at the
# exact position the code occupied, so evaluation order is unchanged.
#
# NOTE: SCRIPT_PATH stays in smoke_data.jl -- it is abspath(@__FILE__) and is the
# target workers re-invoke, so it must resolve to the entry point, not to this file.


# ---------------------------------------------------------------------------
# Index only. The runners live one per file under arms/ so that each arm's
# provenance can hash just its own runner -- see comparative_code_provenance.
# Order matters: common.jl defines the types the runners construct, and
# dispatch.jl refers to every runner, so it goes last.
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "arms", "common.jl"))
include(joinpath(@__DIR__, "arms", "diona.jl"))
include(joinpath(@__DIR__, "arms", "gp.jl"))
include(joinpath(@__DIR__, "arms", "nn.jl"))
include(joinpath(@__DIR__, "arms", "nsm.jl"))
include(joinpath(@__DIR__, "arms", "surrogates.jl"))
include(joinpath(@__DIR__, "arms", "dispatch.jl"))
