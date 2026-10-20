# Arm-generic plumbing: applicability, row/result types, metric reductions.
#
# Plain `include`d FRAGMENT, not a module -- every name stays top-level in the
# including scope. Pulled in by ml/harness/arms.jl, which smoke_data.jl includes.
#
# Split per arm so comparative_code_provenance can hash ONE arm's runner: editing
# this file rotates only every arm (it is shared), not all six.

"""
    applicable_arms(cell::SmokeCell) -> Tuple{Vararg{Symbol}}

The arms run on `cell`. PER-CELL, not global.

On this benchmark it returns all six arms for every cell. It is still per-cell
because `:rbf` builds a DENSE n x n interpolation system and factorises it, and
that is the one thing here that does not scale. Measured 2026-09-21:

    Float64   12.1 bytes/n^2      n = 125,000 -> 176.1 GiB
    Float32    6.04 bytes/n^2     n = 125,000 ->  87.9 GiB

against a 96 GiB pre-flight budget. So the two Ishigami 50^3 cells are fitted at
FLOAT32 rather than excluded, and every other cell at Float64. Float32 is a
storage format, not a tuning knob -- at n = 8,000 the two precisions give
rel-L2 0.00854 and 0.00854, identical to five decimals -- and it is recorded in
the settings hash, so the two can never be confused in a fragment.

The precision is chosen from `n` and the SAME constants `SurrogateArms` itself
uses, never from a cell id, which would silently stop tracking the real
constraint if the tiers changed.

Should a future tier exceed even the Float32 budget, the arm becomes
inapplicable there. That must be an EXCLUSION and not a recorded all-NaN
failure: `common_success_query_ids` intersects the finite-prediction sets of
every arm present on a cell, so one all-NaN arm would zero the common query set
for every other arm on that cell.
"""
function applicable_arms(cell::SmokeCell)
    rbf_fits = _rbf_precision_for(cell.n_train) !== nothing
    return rbf_fits ? (:diona, :nn, :gp, :nsm, :rbf, :idw) :
                      (:diona, :nn, :gp, :nsm, :idw)
end

const _ARM_DISPLAY_NAMES = Dict(:diona => "Diona", :nn => "NN", :gp => "GP",
                                :nsm => "NSM", :rbf => "RBF", :idw => "IDW")
arm_display_name(arm::Symbol) = get(_ARM_DISPLAY_NAMES, arm, uppercasefirst(string(arm)))

struct ArmQueryRow
    cell_id::Int
    query_id::Int
    arm::Symbol
    x::Vector{Float64}
    truth::Float64
    pred::Float64
    lower::Float64
    upper::Float64
    status::Symbol
    elapsed_s::Float64
    # The ABSOLUTE smoothness constant this arm actually used at THIS query --
    # every arm estimates its own from the training data and inflates it by its
    # own fixed factor, so the absolute value is the comparable quantity and the
    # one that drives interval width:
    #   :diona  DionaResult.L_H_used == L̃_H * L_H_factor, where L̃_H is the
    #           per-query LP optimum of manuscript §4.1 (src/inference.jl).
    #   :nsm    NSMModel.gamma == gamma_factor * gamma_raw (constant per cell).
    # NaN for arms that estimate no such constant (:nn, :gp, :rbf, :idw).
    smoothness_used::Float64
end

# 10-argument form for arms that estimate no smoothness constant.
ArmQueryRow(cell_id, query_id, arm, x, truth, pred, lower, upper, status, elapsed_s) =
    ArmQueryRow(cell_id, query_id, arm, x, truth, pred, lower, upper, status,
                elapsed_s, NaN)

struct ArmCellResult
    cell_id::Int
    name::Symbol
    layout::Symbol
    tier_label::String
    dim::Int
    arm::Symbol                    # one of applicable_arms(cell)
    model_type::String             # "Diona", "MLP", "SVGP", "Surrogates.RadialBasis", ...
    fixture_sha256::String         # provenance: which on-disk fixture was consumed
    settings_hash::String          # provenance: sha256 of this arm's canonical settings
    model_seed::Int
    noise_seed::Int
    n_train::Int
    n_query::Int
    fit_time_s::Float64            # global training time; 0.0 for Diona (PRD §7.2)
    inference_time_total_s::Float64
    inference_time_mean_s::Float64
    final_loss::Float64            # NaN for Diona (no scalar training loss)
    n_finite_pred::Int
    n_finite_interval::Int
    rel_l2::Float64
    mae::Float64
    mse::Float64
    coverage::Float64
    mean_width::Float64
    status_counts::Dict{Symbol,Int}
end

function _arm_rel_l2(rows::Vector{ArmQueryRow})
    idx = findall(r -> isfinite(r.pred), rows)
    isempty(idx) && return NaN
    truth = [rows[i].truth for i in idx]
    pred  = [rows[i].pred  for i in idx]
    denom = norm(truth)
    denom == 0.0 && return NaN
    return norm(pred .- truth) / denom
end

function _arm_mae(rows::Vector{ArmQueryRow})
    idx = findall(r -> isfinite(r.pred), rows)
    isempty(idx) && return NaN
    return mean(abs(rows[i].pred - rows[i].truth) for i in idx)
end

function _arm_mse(rows::Vector{ArmQueryRow})
    idx = findall(r -> isfinite(r.pred), rows)
    isempty(idx) && return NaN
    return mean(abs2(rows[i].pred - rows[i].truth) for i in idx)
end

function _arm_interval_stats(rows::Vector{ArmQueryRow})
    idx = findall(r -> isfinite(r.lower) && isfinite(r.upper), rows)
    isempty(idx) && return (NaN, NaN, 0)
    widths  = [rows[i].upper - rows[i].lower for i in idx]
    covered = [(rows[i].truth >= rows[i].lower) && (rows[i].truth <= rows[i].upper) for i in idx]
    return (mean(widths), mean(covered), length(idx))
end

function _arm_status_counts(rows::Vector{ArmQueryRow})
    d = Dict{Symbol,Int}()
    for r in rows
        d[r.status] = get(d, r.status, 0) + 1
    end
    return d
end

function arm_cell_result_from_rows(rows::Vector{ArmQueryRow}; cell_id::Int, cell::SmokeCell,
        arm::Symbol, model_type::String, fixture_sha256::String, settings_hash::String,
        model_seed::Int, noise_seed::Int, n_train::Int, fit_time_s::Float64, final_loss::Float64)
    n_query = length(rows)
    n_finite_pred = count(r -> isfinite(r.pred), rows)
    mean_w, coverage, n_int = _arm_interval_stats(rows)
    elapsed = [r.elapsed_s for r in rows]
    return ArmCellResult(
        cell_id, cell.name, cell.layout, cell.tier_label, cell.dim,
        arm, model_type, fixture_sha256, settings_hash, model_seed, noise_seed,
        n_train, n_query, fit_time_s,
        sum(elapsed), n_query > 0 ? mean(elapsed) : NaN,
        final_loss, n_finite_pred, n_int,
        _arm_rel_l2(rows), _arm_mae(rows), _arm_mse(rows), coverage, mean_w,
        _arm_status_counts(rows),
    )
end

smoke_cell_from_fixture(fx) =
    SmokeCell(Symbol(fx.name), fx.dim, Symbol(fx.layout),
              fx.tier_label, fx.n_train_nominal, fx.noise_level)

function prepared_cell_from_fixture(fx)
    cell = smoke_cell_from_fixture(fx)
    return PreparedSmokeCell(
        cell, fx.cell_id,
        fx.x_train, fx.y_train_clean, fx.y_train_noisy,
        fx.noise_level, fx.noise_bound, fx.noise_seed, fx.output_range,
        Vector{Int}(fx.query_ids), fx.x_query, fx.y_query_truth,
        fx.n_train_actual, fx.n_query,
        fx.schema_version, fx.generated_at,
    )
end
