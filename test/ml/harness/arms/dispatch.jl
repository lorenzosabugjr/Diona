# run_arm_cell: the dispatch over arms, plus shared status vocabulary.
#
# Plain `include`d FRAGMENT, not a module -- every name stays top-level in the
# including scope. Pulled in by ml/harness/arms.jl, which smoke_data.jl includes.
#
# Split per arm so comparative_code_provenance can hash ONE arm's runner: editing
# this file rotates only every arm (it is the dispatch), not all six.

function run_arm_cell(cell::SmokeCell, arm::Symbol, out_dir::AbstractString;
                       cell_id::Union{Int,Nothing}=nothing,
                       reg::ComparatorRegistry.CanonicalRegistry=ComparatorRegistry.canonical_registry(),
                       fx=nothing, verbose::Bool=false, checkpoint_path=nothing,
                       device::String="cpu")
    arm in applicable_arms(cell) ||
        error("run_arm_cell: arm=$arm not applicable to cell $(cell.name) " *
              "(applicable_arms=$(applicable_arms(cell)))")

    if fx === nothing
        prep = prepare_smoke_cell(cell; cell_id=cell_id)
        materialize_fixture(prep, out_dir)
        fpath, _ = fixture_paths(out_dir, cell, prep.cell_id)
        fx = read_fixture(fpath)
    end

    if arm == :diona
        return run_diona_arm_cell(fx; verbose=verbose)
    elseif arm == :gp
        return run_gp_arm_cell(fx, reg; verbose=verbose, checkpoint_path=checkpoint_path)
    elseif arm == :nn
        return run_nn_arm_cell(fx, out_dir, reg; verbose=verbose, checkpoint_path=checkpoint_path, device=device)
    elseif arm == :nsm
        return run_nsm_arm_cell(fx; verbose=verbose)
    elseif arm in (:rbf, :idw)
        return run_surrogate_arm_cell(fx, arm, out_dir; verbose=verbose)
    else
        error("run_arm_cell: arm=$arm is not a recognized arm " *
              "(expected :diona/:gp/:nn/:nsm/:rbf/:idw)")
    end
end

# Only these two arms minimise a scalar training objective; every other arm
# records final_loss = NaN by design (Diona, NSM, RadialBasis and IDW have no
# such quantity), so a gate must not require one.
_arm_reports_training_loss(arm::Symbol) = arm in (:nn, :gp)

const _KNOWN_DIONA_STATUSES = Set(Diona.DIONA_STATUSES)   # src/inference.jl
