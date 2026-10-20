# Diona-only full sweep: fragments, worker pool, orchestrator.
#
# Extracted verbatim from test/smoke_data.jl. This is a plain `include`d FRAGMENT,
# not a module: every name must stay top-level in the including scope, because
# test/run_full_benchmark.jl reaches into the harness via getproperty(harness, :name)
# and test/benchmark_data.jl mutates _SMOKE_OPTS directly. The include sits at the
# exact position the code occupied, so evaluation order is unchanged.
#
# NOTE: SCRIPT_PATH stays in smoke_data.jl -- it is abspath(@__FILE__) and is the
# target workers re-invoke, so it must resolve to the entry point, not to this file.






function write_cell_fragments(out_dir, run_id, id::Int, cell::SmokeCell, cr::CellResult)
    m = compute_cell_metrics(cr)
    stem = cell_stem(cell, id)
    summary_path = joinpath(out_dir, "fragments", "summary", stem * ".csv")
    points_path  = joinpath(out_dir, "fragments", "points", stem * ".csv")

    atomic_write(summary_path) do io
        println(io, _FULL_SWEEP_SUMMARY_HEADER)
        vals = (run_id, id, cell.name, cell.dim, cell.layout, cell.tier_label,
                cell.n_train, cr.n_train_actual, cell.noise_level, cr.noise_bound_used, cr.n_query,
                m.rel_l2, m.coverage, m.mean_width, m.width_over_2eps,
                m.t_mean, m.t_median, m.t_max,
                _full_sweep_phase_vals(m)...,
                _status_counts_str(m.status_counts), "OK", "")
        println(io, join(_safe_csv_field.(vals), ','))
    end

    atomic_write(points_path) do io
        println(io, _FULL_SWEEP_POINTS_HEADER)
        for i in 1:cr.n_query
            has_uq = isfinite(cr.lower[i]) && isfinite(cr.upper[i])
            covered = has_uq && cr.y_truth[i] >= cr.lower[i] && cr.y_truth[i] <= cr.upper[i]
            width   = has_uq ? cr.upper[i] - cr.lower[i] : NaN
            vals = (run_id, id, cell.name, cell.layout, cell.tier_label,
                    cell.n_train, cell.noise_level, i, cr.y_truth[i], cr.pred[i],
                    cr.lower[i], cr.upper[i], covered, width, cr.status[i], cr.elapsed[i])
            println(io, join(_safe_csv_field.(vals), ','))
        end
    end
    return summary_path, points_path
end

function write_failure_fragment(out_dir, run_id, id::Int, cell::SmokeCell, state::Symbol, errmsg::String)
    stem = cell_stem(cell, id)
    summary_path = joinpath(out_dir, "fragments", "summary", stem * ".csv")
    atomic_write(summary_path) do io
        println(io, _FULL_SWEEP_SUMMARY_HEADER)
        vals = (run_id, id, cell.name, cell.dim, cell.layout, cell.tier_label,
                cell.n_train, NaN, cell.noise_level, NaN, 0,
                NaN, NaN, NaN, NaN, NaN, NaN, NaN,
                fill(NaN, length(_FULL_SWEEP_PHASES) * length(_FULL_SWEEP_STATS))...,
                "", string(state), errmsg)
        println(io, join(_safe_csv_field.(vals), ','))
    end
    return summary_path
end

function run_worker()
    if _smoke_opt("arm", nothing) !== nothing
        return run_comparative_arm_worker()
    end

    out_dir = abspath(_smoke_required("out"))
    run_id  = _smoke_opt("run-id", basename(out_dir))
    id      = parse(Int, _smoke_required("cell"))
    cells   = enumerate_cells()
    (1 <= id <= length(cells)) || error("--cell=$id out of range [1,$(length(cells))]")
    cell    = cells[id]

    LinearAlgebra.BLAS.set_num_threads(1)

    println("worker cell=$id  $(cell.name)/$(cell.layout)/$(cell.tier_label)  " *
            "n_train=$(cell.n_train)  noise=$(cell.noise_level)")
    flush(stdout)

    try
        cr = run_cell(cell; verbose=false)
        write_cell_fragments(out_dir, run_id, id, cell, cr)
        m = compute_cell_metrics(cr)
        @printf("DONE cell=%d rel-L2=%.6g coverage=%.3f n_query=%d t_mean=%.1fms\n",
                id, m.rel_l2, m.coverage, cr.n_query, 1000 * m.t_mean)
    catch err
        write_failure_fragment(out_dir, run_id, id, cell, :ERROR, sprint(showerror, err))
        println(stderr, "FAILED cell=$id: $(sprint(showerror, err))")
        Base.show_backtrace(stderr, catch_backtrace())
        rethrow()
    end
end

function worker_command(id::Int, out_dir::String, run_id::String)
    julia = Base.julia_cmd()
    args = ["--worker", "--cell=$(id)", "--out=$(out_dir)", "--run-id=$(run_id)"]
    return `$julia --threads=1 --project=$(WORKER_PROJECT) $(SCRIPT_PATH) $args`
end

function run_worker_pool(ids::Vector{Int}, out_dir::String, run_id::String,
                         jobs::Int, timeout_s::Real, log_dir::String)
    println("\n=== Full sweep: $(length(ids)) cells, concurrency=$jobs, timeout=$(timeout_s)s/cell ===")
    flush(stdout)
    tokens = Channel{Nothing}(jobs)
    for _ in 1:jobs
        put!(tokens, nothing)
    end
    outcomes = Dict{Int,Symbol}()
    outcomes_lock = ReentrantLock()
    @sync for id in ids
        @async begin
            take!(tokens)
            try
                cmd = worker_command(id, out_dir, run_id)
                log_path = joinpath(log_dir, "cell_$(lpad(id, 3, '0')).log")
                mkpath(dirname(log_path))
                outcome = open(log_path, "w") do io
                    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
                    t0 = time()
                    result = :OK
                    while process_running(proc)
                        if time() - t0 > timeout_s
                            kill(proc)
                            sleep(0.5)
                            process_running(proc) && kill(proc, Base.SIGKILL)
                            result = :TIMEOUT
                            break
                        end
                        sleep(0.5)
                    end
                    result == :TIMEOUT && return :TIMEOUT
                    return proc.exitcode == 0 ? :OK : :FAILED
                end
                lock(outcomes_lock) do
                    outcomes[id] = outcome
                end
                println("finish cell=$id: $outcome")
                flush(stdout)
            finally
                put!(tokens, nothing)
            end
        end
    end
    return outcomes
end

function concatenate_fragments(out_dir::String, kind::String, header::String)
    dir = joinpath(out_dir, "fragments", kind)
    files = isdir(dir) ? sort(filter(f -> endswith(f, ".csv"), readdir(dir; join=true))) : String[]
    path = joinpath(out_dir, kind * ".csv")
    atomic_write(path) do io
        println(io, header)
        for file in files
            lines = readlines(file)
            length(lines) >= 2 || continue
            for line in lines[2:end]
                println(io, line)
            end
        end
    end
    return path, length(files)
end

function read_summary_rows(summary_path::String)
    lines = readlines(summary_path)
    isempty(lines) && return NamedTuple[]
    header = split(lines[1], ',')
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(line) && continue
        fields = split(line, ',')
        d = Dict(Symbol(h) => f for (h, f) in zip(header, fields))
        phase_nt = NamedTuple(col => _parse_float(d[col]) for col in _full_sweep_phase_col_names())
        base_nt = (
            run_id           = d[:run_id],
            cell_id          = parse(Int, d[:cell_id]),
            name             = Symbol(d[:name]),
            dim              = parse(Int, d[:dim]),
            layout           = Symbol(d[:layout]),
            tier_label       = d[:tier_label],
            n_train          = parse(Int, d[:n_train_nominal]),
            n_train_actual   = _parse_float(d[:n_train_actual]),
            noise_level      = parse(Float64, d[:noise_level]),
            noise_bound_used = _parse_float(d[:noise_bound_used]),
            n_query          = parse(Int, d[:n_query]),
            rel_l2           = _parse_float(d[:rel_l2]),
            coverage         = _parse_float(d[:coverage]),
            mean_width       = _parse_float(d[:mean_width]),
            width_over_2eps  = _parse_float(d[:width_over_2eps]),
            t_mean_s         = _parse_float(d[:t_mean_s]),
            t_median_s       = _parse_float(d[:t_median_s]),
            t_max_s          = _parse_float(d[:t_max_s]),
        )
        tail_nt = (
            status_counts    = d[:status_counts],
            state            = Symbol(d[:state]),
            error            = d[:error],
        )
        push!(rows, merge(base_nt, phase_nt, tail_nt))
    end
    return rows
end

function print_console_report(io::IO, rows)
    println(io)
    println(io, "=" ^ 112)
    println(io, "Full Sweep Report")
    println(io, "=" ^ 112)
    @printf(io, "%-4s %-14s %-9s %-10s %9s %6s %8s %8s %9s %9s  %s\n",
            "id", "name", "layout", "tier", "n_train", "noise", "relL2", "coverage",
            "width/2e", "t_mean", "state")
    println(io, "-" ^ 112)
    keys_sorted = sort(unique(string(r.name) for r in rows))
    for n in keys_sorted
        subset = filter(r -> string(r.name) == n, rows)
        for r in sort(subset; by=r -> (string(r.layout), r.n_train, r.noise_level))
            n_tr_shown = isfinite(r.n_train_actual) ? round(Int, r.n_train_actual) : r.n_train
            @printf(io, "%-4d %-14s %-9s %-10s %9d %6.2f %8s %8s %9s %9s  %s\n",
                    r.cell_id, string(r.name), string(r.layout), r.tier_label,
                    n_tr_shown, r.noise_level,
                    isfinite(r.rel_l2) ? @sprintf("%.4f", r.rel_l2) : "—",
                    isfinite(r.coverage) ? @sprintf("%.3f", r.coverage) : "—",
                    isfinite(r.width_over_2eps) ? @sprintf("%.2f", r.width_over_2eps) : "—",
                    isfinite(r.t_mean_s) ? @sprintf("%.1fms", 1000 * r.t_mean_s) : "—",
                    string(r.state))
        end
    end
    println(io, "-" ^ 112)
    n_ok = count(r -> r.state == :OK, rows)
    println(io, "Total: $(length(rows)) rows  ($(n_ok) OK, $(length(rows) - n_ok) TIMEOUT/ERROR)")
end


function run_orchestrator()
    cells_all   = enumerate_cells()
    only_filter = _smoke_opt("only", nothing)
    ids = only_filter === nothing ? collect(1:length(cells_all)) :
          [i for i in 1:length(cells_all) if cell_matches_filter(cells_all[i], only_filter)]
    isempty(ids) && error("no cells matched --only=$(only_filter)")

    out_dir   = abspath(_smoke_opt("out", smoke_data_default_out_dir()))
    run_id    = basename(out_dir)
    jobs      = parse(Int, _smoke_opt("jobs", string(DEFAULT_JOBS_FULL_SWEEP)))
    timeout_s = parse(Float64, _smoke_opt("timeout", string(DEFAULT_TIMEOUT_FULL_SWEEP)))
    mkpath(out_dir)

    println("Full sweep orchestrator: $(length(ids))/$(length(cells_all)) cells selected" *
            (only_filter === nothing ? "" : " (--only=$only_filter)"))
    println("out_dir = $out_dir")
    println("jobs = $jobs   timeout = $(timeout_s)s/cell")
    flush(stdout)

    outcomes = run_worker_pool(ids, out_dir, run_id, jobs, timeout_s, joinpath(out_dir, "logs"))

    println("\nPatching any missing fragments...")
    n_patched = 0
    for id in ids
        cell  = cells_all[id]
        spath = summary_fragment_path(out_dir, cell, id)
        isfile(spath) && continue
        outcome = get(outcomes, id, :UNKNOWN)
        state = outcome == :TIMEOUT ? :TIMEOUT : :ERROR
        msg   = outcome == :TIMEOUT ?
            "worker exceeded $(timeout_s)s timeout, killed by orchestrator" :
            "worker exited (outcome=$outcome) without producing a fragment; see logs/cell_$(lpad(id,3,'0')).log"
        write_failure_fragment(out_dir, run_id, id, cell, state, msg)
        println("  patched cell=$id -> $state")
        n_patched += 1
    end
    n_patched == 0 && println("  none needed.")

    summary_path, ns = concatenate_fragments(out_dir, "summary", _FULL_SWEEP_SUMMARY_HEADER)
    points_path,  np = concatenate_fragments(out_dir, "points",  _FULL_SWEEP_POINTS_HEADER)
    println("\naggregated $ns summary fragments -> $summary_path")
    println("aggregated $np point fragments -> $points_path")

    rows = read_summary_rows(summary_path)
    print_console_report(stdout, rows)

    ok = check_gate_full_sweep_coverage(rows, ids; verbose=true)
    println("\nOutput directory: $out_dir")
    ok || error("Full sweep gate FAILED — see missing cell ids above")
    return out_dir, rows
end
