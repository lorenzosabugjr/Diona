#!/usr/bin/env julia
# Turn a finished campaign's fragments into the data behind the manuscript's
# figures (Section 6): one JSON row per arm-cell with
#
#   rel_l2            ε_L2 = ‖U − Ũ‖ / ‖U‖ over the query set
#   coverage          ρ, fraction of queries with u̲ ≤ u ≤ ū (interval arms)
#   median_rel_width  median over queries of w(q) / range(u), range = max − min of u on its box
#   median_rel_width_abs_u
#                     median over queries of w(q) / |u(q)|  (the manuscript's w_r)
#   mean_s_query, median_s_query, fit_time_s, and for :diona the per-stage mean
#   times of Algorithm 2 (neighbor, lh, assembly, bounds, residual, total).
#
# Usage:
#   julia --project=test/ml paper/extract_campaign.jl <runs dir> <out json> [arm=settings_prefix ...]
#
# <runs dir> is <run.out>/comparative_fragments/runs. When one out dir holds
# several campaigns of the same arm (different settings hashes), name the one
# to keep per arm, e.g. diona=962d55947a01 nsm=f3807f34dace; otherwise every
# arm must have exactly one settings hash in the directory or this script stops.
#
# No packages beyond test/ml's own environment: adding one would rotate the
# harness provenance.
using Statistics, Printf
const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "test", "ml", "common", "point_fragments.jl")); using .PointFragments
include(joinpath(REPO, "test", "ml", "common", "driver.jl"));          using .BenchmarkDriver

length(ARGS) >= 2 || error("usage: extract_campaign.jl <runs dir> <out json> [arm=settings_prefix ...]")
const RUNS = abspath(ARGS[1])
const OUT  = abspath(ARGS[2])
const KEEP = Dict{String,String}(split(a, "=")[1] => split(a, "=")[2] for a in ARGS[3:end])
isdir(RUNS) || error("not a directory: $RUNS")
const ORV = Dict(bf.name => output_range(bf) for bf in BENCHMARK_FUNCTIONS)

csvdict(p) = begin
    l = readlines(p); h = split(l[1], ','); v = split(l[2], ','; keepempty=true)
    Dict(Symbol(strip(a)) => String(b) for (a, b) in zip(h, v))
end
num(d, k) = (s = get(d, k, ""); (isempty(s) || s == "NaN") ? NaN : parse(Float64, s))

function stage_means(summary_path)
    tp = replace(summary_path, ".summary.csv" => ".timing.csv")
    isfile(tp) || return nothing
    d = csvdict(tp)
    g(k) = num(d, k)
    return Dict(:neighbor => g(:t_neighbor_mean_s), :lh => g(:t_lh_mean_s),
                :assembly => g(:t_assembly_mean_s), :bounds => g(:t_bounds_mean_s),
                :residual => g(:residual_mean_s), :total => g(:total_mean_s))
end

summaries = sort(filter(x -> endswith(x, ".summary.csv"), readdir(RUNS; join=true)))
isempty(summaries) && error("no *.summary.csv under $RUNS")

# Settings hashes present per (arm, dimension); refuse to merge campaigns
# silently. Per dimension because the :nn and :gp settings depend on D
# (layers, inducing points), so each arm legitimately has one hash per D.
present = Dict{Tuple{String,Int},Set{String}}()
for f in summaries
    d = csvdict(f)
    d[:terminal_state] == "OK" || continue
    push!(get!(present, (d[:arm], parse(Int, d[:dim])), Set{String}()), d[:settings_hash])
end
for ((arm, dim), hs) in present
    haskey(KEEP, arm) && continue
    length(hs) == 1 || error("arm $arm (D=$dim) has $(length(hs)) settings hashes under $RUNS: " *
                             join([h[1:12] for h in hs], ", ") * ". Pass $arm=<prefix> to choose one.")
end

rows = Any[]
for f in summaries
    d = csvdict(f)
    d[:terminal_state] == "OK" || continue
    haskey(KEEP, d[:arm]) && !startswith(d[:settings_hash], KEEP[d[:arm]]) && continue
    stem = first(split(basename(f), "__"))
    noise_db = endswith(stem, "_clean") ? 0 :
        (m = match(r"_snr(\d+)db$", stem); m === nothing ?
            error("cannot read the noise condition from stem '$stem'") : parse(Int, m[1]))
    name = Symbol(d[:name]); orv = ORV[name]
    pf = replace(f, ".summary.csv" => ".points.arrow")
    med_w = NaN; med_w_u = NaN; med_s = NaN
    cov1 = NaN; cov2 = NaN; cov3 = NaN
    if isfile(pf)
        pr = read_point_fragment(pf)
        fin = [r for r in pr if isfinite(Float64(r[:lower])) && isfinite(Float64(r[:upper]))]
        w = [Float64(r[:upper]) - Float64(r[:lower]) for r in fin]
        wu = [(Float64(r[:upper]) - Float64(r[:lower])) / abs(Float64(r[:truth]))
              for r in fin if Float64(r[:truth]) != 0]
        el = [Float64(r[:elapsed_s]) for r in pr if isfinite(Float64(r[:elapsed_s]))]
        isempty(w)  || (med_w = median(w) / orv)
        isempty(wu) || (med_w_u = median(wu))
        isempty(el) || (med_s = median(el))
        if d[:arm] == "gp"
            # The GP interval is pred ± 2σ and symmetric, so σ = (upper − lower)/4
            # recovers the posterior scale and the 1σ/3σ coverages need no refit.
            c = zeros(3); n = 0
            for r in pr
                lo, up = Float64(r[:lower]), Float64(r[:upper])
                pv, tv = Float64(r[:pred]), Float64(r[:truth])
                (isfinite(lo) && isfinite(up) && isfinite(pv) && isfinite(tv)) || continue
                sg = (up - lo) / 4.0
                n += 1
                for (k, z) in enumerate((1.0, 2.0, 3.0))
                    (pv - z * sg <= tv <= pv + z * sg) && (c[k] += 1)
                end
            end
            n == 0 || ((cov1, cov2, cov3) = (c[1] / n, c[2] / n, c[3] / n))
        end
    end
    push!(rows, Dict(
        :cell_id => parse(Int, d[:cell_id]), :arm => d[:arm], :name => String(d[:name]),
        :layout => d[:layout], :tier => d[:tier_label], :dim => parse(Int, d[:dim]),
        :n_train => parse(Int, d[:n_train]), :n_query => parse(Int, d[:n_query]),
        :noise_db => noise_db,
        :rel_l2 => num(d, :rel_l2), :mae => num(d, :mae), :coverage => num(d, :coverage),
        :cov_1sigma => cov1, :cov_2sigma => cov2, :cov_3sigma => cov3,
        :mean_rel_width => num(d, :mean_width) / orv, :median_rel_width => med_w,
        :median_rel_width_abs_u => med_w_u,
        :mean_s_query => num(d, :inference_time_mean_s), :median_s_query => med_s,
        :fit_time_s => num(d, :fit_time_s), :blas_threads => parse(Int, d[:blas_threads]),
        :n_finite_pred => parse(Int, d[:n_finite_pred]),
        :n_finite_interval => parse(Int, d[:n_finite_interval]),
        :output_range => orv,
        :settings_hash => d[:settings_hash], :code_provenance => d[:code_provenance],
        :stages => stage_means(f)))
end
@printf("extracted %d arm-cells\n", length(rows))
for a in sort(unique(r[:arm] for r in rows))
    s = [r for r in rows if r[:arm] == a]
    @printf("  %-6s %3d cells  rel_l2 %.5f..%.5f  fit %.1f..%.1fs  blas=%d\n",
        a, length(s), minimum(r[:rel_l2] for r in s), maximum(r[:rel_l2] for r in s),
        minimum(r[:fit_time_s] for r in s), maximum(r[:fit_time_s] for r in s),
        s[1][:blas_threads])
end
jval(x) = x === nothing ? "null" :
          x isa Dict ? "{" * join(("\"$(k)\":$(jval(v))" for (k, v) in sort(collect(x), by=first)), ",") * "}" :
          x isa AbstractString ? "\"" * x * "\"" :
          x isa Integer ? string(x) :
          (isnan(x) || !isfinite(x)) ? "null" : string(x)
mkpath(dirname(OUT))
open(OUT, "w") do io
    print(io, "{\"rows\":[")
    for (i, r) in enumerate(rows)
        i > 1 && print(io, ",")
        print(io, "{", join(("\"$(k)\":$(jval(v))" for (k, v) in sort(collect(r), by=first)), ","), "}")
    end
    print(io, "]}")
end
println("wrote $OUT")
