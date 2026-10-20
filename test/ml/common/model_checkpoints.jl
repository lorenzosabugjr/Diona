module ModelCheckpoints

using Serialization

export checkpoint_identity, load_model_checkpoint, write_model_checkpoint

const CHECKPOINT_SCHEMA = "diona-trained-model-checkpoint-v1"

"""Return the complete identity key used to decide whether trained state is reusable."""
checkpoint_identity(cell_id::Integer, arm::Symbol, model_seed::Integer,
                    fixture_sha256::AbstractString, settings_hash::AbstractString) = (
    cell_id=Int(cell_id),
    arm=arm,
    model_seed=Int(model_seed),
    fixture_sha256=String(fixture_sha256),
    settings_hash=String(settings_hash),
)

function _identity_matches(actual, expected)
    return all(hasproperty(actual, field) &&
               isequal(getproperty(actual, field), getproperty(expected, field))
               for field in propertynames(expected))
end

"""
    load_model_checkpoint(path, expected_identity)

Deserialize and return a checkpoint payload when its schema and complete identity
match. Missing, stale, corrupt, or structurally invalid files return `nothing`, so
callers can safely fall back to fitting.
"""
function load_model_checkpoint(path::AbstractString, expected_identity)
    isfile(path) || return nothing
    try
        envelope = open(deserialize, path)
        hasproperty(envelope, :schema) && envelope.schema == CHECKPOINT_SCHEMA || return nothing
        hasproperty(envelope, :identity) || return nothing
        hasproperty(envelope, :payload) || return nothing
        _identity_matches(envelope.identity, expected_identity) || return nothing
        return envelope.payload
    catch
        return nothing
    end
end

"""Atomically serialize trained state and its reuse identity."""
function write_model_checkpoint(path::AbstractString, identity, payload)
    mkpath(dirname(path))
    temporary = String(path) * ".tmp-$(getpid())"
    try
        open(temporary, "w") do io
            serialize(io, (schema=CHECKPOINT_SCHEMA, identity=identity, payload=payload))
        end
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return String(path)
end

end # module ModelCheckpoints
