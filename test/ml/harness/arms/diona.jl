# The :diona arm.
#
# Plain `include`d FRAGMENT, not a module -- every name stays top-level in the
# including scope. Pulled in by ml/harness/arms.jl, which smoke_data.jl includes.
#
# Split per arm so comparative_code_provenance can hash ONE arm's runner: editing
# this file rotates only :diona, not all six.

function run_diona_arm_cell(fx; verbose::Bool=false)
    prep = prepared_cell_from_fixture(fx)
    # Diona's realised absolute smoothness constant per query, L̃_H × L_H_factor
    # (manuscript §4.1 and Table 1). Recorded per query because L̃_H is a
    # per-query LP optimum, not a campaign constant.
    lh_used = fill(NaN, prep.n_query)
    cr = solve_cell_diona(prep; verbose=verbose, lh_used=lh_used)
    rows = ArmQueryRow[
        ArmQueryRow(fx.cell_id, i, :diona, prep.X_query[:, i], cr.y_truth[i],
                    cr.pred[i], cr.lower[i], cr.upper[i], cr.status[i], cr.elapsed[i],
                    lh_used[i])
        for i in 1:cr.n_query
    ]
    settings_hash = bytes2hex(sha256(_diona_settings_string()))
    _write_diona_stage_timings(fx, cr, settings_hash)
    result = arm_cell_result_from_rows(rows; cell_id=fx.cell_id, cell=prep.cell, arm=:diona,
        model_type="Diona", fixture_sha256=fx.fixture_sha256, settings_hash=settings_hash,
        model_seed=_SMOKE_MODEL_SEED, noise_seed=fx.noise_seed, n_train=fx.n_train_actual,
        fit_time_s=0.0, final_loss=NaN)
    return result, rows
end


"""
    _write_diona_stage_timings(fx, cr, settings_hash)

Write the per-stage MEAN inference time for this cell to a sidecar CSV beside
the fragment, as `<stem>.timing.csv`.

WHY A SIDECAR AND NOT A COLUMN ON `ArmCellResult`. That struct lives in
`arms/common.jl`, which is hashed into EVERY arm's `comparative_code_provenance`;
`campaign.jl` treats a fragment whose stored provenance differs from the current
one as stale. Adding a field there would therefore mark all 216 fragments of a
six-arm campaign stale and force a full re-run to add a number that only :diona
has. Everything here is confined to this file, so only :diona rotates.

MEANS, NOT MEDIANS, on purpose: the stage times are additive in the mean
(`mean(total) == sum of stage means`) and are not in the median, so only the
means can be stacked into a bar whose height is the reported per-query cost.
`residual_mean_s` is `mean(elapsed) - sum(stage means)`: the wall time inside
the per-query loop that no stage timer claims. It is written rather than
silently folded into a stage, so a gap is visible instead of inferred.

Non-finite entries are dropped per stage, since a failed solve leaves NaN
behind and one NaN would otherwise poison the whole cell's mean.
"""
function _write_diona_stage_timings(fx, cr, settings_hash::AbstractString)
    out = get(_SMOKE_OPTS, "out", "")
    isempty(out) && return nothing
    dir = joinpath(abspath(out), "comparative_fragments", "runs")
    isdir(dir) || return nothing
    mean_finite(v) = (f = filter(isfinite, v); isempty(f) ? NaN : sum(f) / length(f))
    m_n  = mean_finite(cr.t_neighbor)
    m_l  = mean_finite(cr.t_lh)
    m_a  = mean_finite(cr.t_assembly)
    m_b  = mean_finite(cr.t_bounds)
    m_tot = mean_finite(cr.elapsed)
    resid = m_tot - (m_n + m_l + m_a + m_b)
    stem = comparative_arm_stem(cr.cell, fx.cell_id, :diona, fx.fixture_sha256, settings_hash)
    path = joinpath(dir, stem * ".timing.csv")
    tmp = path * ".tmp" * string(getpid())
    open(tmp, "w") do io
        println(io, "cell_id,arm,n_query,t_neighbor_mean_s,t_lh_mean_s," *
                    "t_assembly_mean_s,t_bounds_mean_s,total_mean_s,residual_mean_s")
        println(io, join((fx.cell_id, "diona", cr.n_query, m_n, m_l, m_a, m_b,
                          m_tot, resid), ","))
    end
    mv(tmp, path; force=true)
    return path
end
