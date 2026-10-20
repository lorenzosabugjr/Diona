# Canonical data-only NN/GP settings and reproducible configuration hashes.

module ComparatorRegistry

using SHA

export NNSettings, GPSettings, CanonicalRegistry,
       canonical_registry, nn_settings, gp_settings,
       settings_for, benchmark_dims, canonical_serialize, registry_hash,
       env_override_keys, registry_with_env_overrides,
       float_type, normalize_device, SUPPORTED_DEVICES, nn_settings_hash,
       NN_8D_PROVENANCE_CAVEAT, GP_INTERVAL_SEMANTICS, NN_DEVICE_SEMANTICS

# 5: NNSettings.precision moved from "Float64" to "Float32" and became
#    load-bearing (NNTrainer reads it instead of hard-coding Lux.f64), and the
#    :nn settings hash gained the execution device. See NN_DEVICE_SEMANTICS.
const SCHEMA_VERSION = 5
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

const NN_8D_PROVENANCE_CAVEAT =
    "NN architecture is derived only from input dimension: width=64 and " *
    "layers=ceil(Int, dim/2)+1. It is not selected or validated per benchmark " *
    "function; the same rule applies uniformly to the formula-derived 8D " *
    "architecture."

const NN_DEVICE_SEMANTICS =
    "The :nn arm runs on the device named by --device (cpu or cuda) at the " *
    "precision named by NNSettings.precision. NEITHER is a tuning knob: the " *
    "architecture, epochs, learning rate and seed are identical on both devices. " *
    "The seed is reproducible WITHIN a device only -- Lux initialisation draws " *
    "from the host RNG and is therefore identical across devices, but Adam's " *
    "accumulation order is not, so a CPU fit and a CUDA fit at the same seed " *
    "agree only to floating-point tolerance, never bit-for-bit. The device is " *
    "therefore folded into the :nn settings hash (nn_settings_hash), so a CPU " *
    "fragment and a CUDA fragment land on different fragment paths and can " *
    "never be silently resumed from, or compared against, one another."

const GP_INTERVAL_SEMANTICS =
    "GP interval = mean +/- z_score * std of the SVGP VARIATIONAL/APPROXIMATE " *
    "posterior over the latent function f (ApproximateGPs.SparseVariationalApproximation), " *
    "NOT the exact GP-regression posterior interval; no formal coverage guarantee. " *
    "z_score=2.0 is the coarse '~95%' convention used throughout this benchmark " *
    "(see test/ml/gp/trainer.jl gp_predict docstring)."

struct NNSettings
    dim::Int
    width::Int
    layers::Int
    optimizer::String
    learning_rate::Float64
    epochs::Int
    seed::Int
    minibatch_threshold::Int
    minibatch_batch_size::Int
    precision::String
    provenance::String
end

struct GPSettings
    dim::Int
    epochs::Int
    inducing_points::Int
    learning_rate::Float64
    optimizer::String
    init_variance::Float64
    init_lengthscale::Float64
    init_noise_std::Float64
    jitter::Float64
    seed::Int
    z_score::Float64
    minibatch_threshold::Int
    minibatch_batch_size::Int
    precision::String
    provenance::String
end


struct CanonicalRegistry
    schema_version::Int
    nn::Dict{Int,NNSettings}
    gp::Dict{Int,GPSettings}
    canonical::Bool
    overrides::Dict{String,String}
end

const _NN_EPOCHS = 25_000
const _GP_EPOCHS = 500
# Every input dimension the cell table uses. :piston added 7. The NN and GP
# settings are pure functions of the dimension (width/layers from
# _arch_for_dim, inducing points from 100*dim), so a new dim needs no new
# tuning -- but it must be listed here or settings_for raises.
const _BENCHMARK_DIMS = (3, 4, 5, 6, 7, 8)

_arch_for_dim(dim::Int) = (width=64, layers=ceil(Int, dim/2)+1)
const _NN_PROVENANCE =
    "width/layers: dimension-only formula width=64, layers=ceil(Int, dim/2)+1; " *
    "not selected or validated per benchmark function. " *
    "epochs=25000: fixed, uniformly applied compute budget; not a literature-derived " *
    "optimum or per-problem tuning choice. lr/minibatch: inherited campaign CLI defaults " *
    "(nn-lr=1e-3, minibatch-threshold=20000, nn-batch-size=2048). " *
    "optimizer=Adam: NNTrainer.train_nn (Optimisers.Adam(lr)). " *
    "precision=Float32: read from this field by NNTrainer.train_nn (Lux.f32(...)), " *
    "chosen so the arm can use the benchmark host's GPU, whose FP64 throughput is " *
    "1/64 of its FP32 rate (RTX 4500 Ada, compute capability 8.9; measured 0.58 vs " *
    "24.9 TFLOPS). It is a compute-budget decision applied uniformly to every cell " *
    "and BOTH devices, not a per-problem or per-device tuning choice. " *
    "The fixture, the queries and every reported value remain Float64."

function nn_settings()
    d = Dict{Int,NNSettings}()
    for dim in _BENCHMARK_DIMS
        arch = _arch_for_dim(dim)
        d[dim] = NNSettings(
            dim, arch.width, arch.layers,
            "Adam", 1.0e-3, _NN_EPOCHS, 0,
            20_000, 2048,
            "Float32",
            _NN_PROVENANCE,
        )
    end
    return d
end
const _GP_PROVENANCE =
    "epochs=500: fixed optimization budget, not dimension-indexed or a per-problem " *
    "tuning knob (and not GPTrainer.train_gp's own 2000-epoch default). GP capacity " *
    "already scales via inducing_points=100*dim (M = 100 * bf.dim). " *
    "learning_rate/minibatch: inherited campaign CLI defaults (gp-lr=1e-2, " *
    "minibatch-threshold=20000, gp-batch-size=1024). " *
    "init_variance=1.0/init_lengthscale=1.0/jitter=1e-6: GPTrainer.train_gp defaults. " *
    "init_noise_std=0.1, z_score=2.0: explicit fit_predict_gp call sites. " *
    "optimizer=Adam: GPTrainer.train_gp (Optimisers.Adam(lr))."

function gp_settings()
    d = Dict{Int,GPSettings}()
    for dim in _BENCHMARK_DIMS
        d[dim] = GPSettings(
            dim, _GP_EPOCHS, 100 * dim,
            1.0e-2, "Adam",
            1.0, 1.0, 0.1, 1.0e-6,
            0, 2.0,
            20_000, 1024,
            "Float64",
            _GP_PROVENANCE,
        )
    end
    return d
end

function canonical_registry()
    return CanonicalRegistry(
        SCHEMA_VERSION,
        nn_settings(),
        gp_settings(),
        true,
        Dict{String,String}(),
    )
end

benchmark_dims() = collect(_BENCHMARK_DIMS)

function nn_settings(reg::CanonicalRegistry, dim::Int)
    haskey(reg.nn, dim) ||
        error("no NN settings for dim=$dim; expected one of $(benchmark_dims())")
    return reg.nn[dim]
end

function gp_settings(reg::CanonicalRegistry, dim::Int)
    haskey(reg.gp, dim) ||
        error("no GP settings for dim=$dim; expected one of $(benchmark_dims())")
    return reg.gp[dim]
end


function settings_for(reg::CanonicalRegistry, dim::Int, arm::Symbol)
    if arm == :nn
        return nn_settings(reg, dim)
    elseif arm == :gp
        return gp_settings(reg, dim)
    end
    error("unsupported comparison arm: $arm")
end

canonical_serialize(x::Bool) = x ? "true" : "false"
canonical_serialize(x::Integer) = string(Int(x))
_float_bits(x::Float64) = string(reinterpret(UInt64, x); base=16, pad=16)
canonical_serialize(x::Float64) = "f64:0x" * _float_bits(x)
canonical_serialize(x::AbstractFloat) = canonical_serialize(Float64(x))
canonical_serialize(x::AbstractString) = "s:" * repr(String(x))
canonical_serialize(x::Symbol) = "sym:" * String(x)

function canonical_serialize(x::AbstractDict)
    entries = [(canonical_serialize(k), canonical_serialize(v)) for (k, v) in x]
    sort!(entries; by=first)
    return "{" * join((k * "=>" * v for (k, v) in entries), ",") * "}"
end

function canonical_serialize(x::AbstractVector)
    return "[" * join(canonical_serialize.(x), ",") * "]"
end

function canonical_serialize(x::T) where {T}
    isstructtype(T) || return "raw:" * repr(x)
    parts = [string(f) * "=" * canonical_serialize(getfield(x, f)) for f in fieldnames(T)]
    return string(nameof(T)) * "(" * join(parts, ",") * ")"
end

"""
    float_type(precision::AbstractString) -> Type{<:AbstractFloat}

Map a settings `precision` field onto the concrete element type the trainer
fits in. Registry-driven: the trainer never names a float type of its own.
"""
function float_type(precision::AbstractString)
    precision == "Float32" && return Float32
    precision == "Float64" && return Float64
    error("unsupported precision: $(repr(precision)); expected \"Float32\" or \"Float64\"")
end
# Duck-typed on purpose: the arm runners accept a stand-in settings object in
# tests (a NamedTuple with small, deliberately noncanonical budgets), and that
# stand-in must carry `precision` exactly as the real struct does rather than
# silently inheriting a default float type from the trainer.
float_type(s) = hasproperty(s, :precision) ? float_type(getproperty(s, :precision)) :
    error("float_type: $(typeof(s)) carries no `precision` field; the :nn arm reads " *
          "its working precision from the settings and never names one of its own")

const SUPPORTED_DEVICES = ("cpu", "cuda")

"""
    normalize_device(device) -> String

Canonicalise a `--device` value. Case-insensitive; errors on anything outside
`SUPPORTED_DEVICES` so a typo cannot silently fall back to the CPU.
"""
function normalize_device(device::AbstractString)
    d = lowercase(strip(String(device)))
    d in SUPPORTED_DEVICES ||
        error("unsupported device: $(repr(String(device))); expected one of " *
              join(SUPPORTED_DEVICES, ", "))
    return d
end

"""
    nn_settings_hash(s::NNSettings, device) -> String

The :nn settings hash. It covers the canonical settings AND the execution
device, because a CPU fit and a CUDA fit at the same seed are not bit-identical
(see `NN_DEVICE_SEMANTICS`). Every :nn call site must use this one function --
the campaign compares the hash the arm returns against the hash it expected.
"""
function nn_settings_hash(s::NNSettings, device::AbstractString)
    text = canonical_serialize(s) * "|device=" * normalize_device(device)
    return bytes2hex(sha256(text))
end

function registry_hash(reg::CanonicalRegistry)
    return bytes2hex(sha256(canonical_serialize(reg)))
end

const ENV_OVERRIDE_KEYS = (
    "DIONA_REGISTRY_NN_EPOCHS", "DIONA_REGISTRY_NN_LR",
    "DIONA_REGISTRY_GP_EPOCHS", "DIONA_REGISTRY_GP_LR",
)

env_override_keys() = collect(ENV_OVERRIDE_KEYS)

function registry_with_env_overrides(base::CanonicalRegistry=canonical_registry())
    overrides = Dict{String,String}()
    for k in ENV_OVERRIDE_KEYS
        haskey(ENV, k) && (overrides[k] = ENV[k])
    end
    isempty(overrides) && return base

    nn_epochs = haskey(overrides, "DIONA_REGISTRY_NN_EPOCHS") ?
        parse(Int, overrides["DIONA_REGISTRY_NN_EPOCHS"]) : nothing
    nn_lr = haskey(overrides, "DIONA_REGISTRY_NN_LR") ?
        parse(Float64, overrides["DIONA_REGISTRY_NN_LR"]) : nothing
    gp_epochs = haskey(overrides, "DIONA_REGISTRY_GP_EPOCHS") ?
        parse(Int, overrides["DIONA_REGISTRY_GP_EPOCHS"]) : nothing
    gp_lr = haskey(overrides, "DIONA_REGISTRY_GP_LR") ?
        parse(Float64, overrides["DIONA_REGISTRY_GP_LR"]) : nothing

    nn = Dict{Int,NNSettings}()
    for (dim, s) in base.nn
        nn[dim] = NNSettings(
            s.dim, s.width, s.layers, s.optimizer,
            something(nn_lr, s.learning_rate), something(nn_epochs, s.epochs), s.seed,
            s.minibatch_threshold, s.minibatch_batch_size, s.precision,
            s.provenance * " [ENV override applied -- NON-CANONICAL]",
        )
    end

    gp = Dict{Int,GPSettings}()
    for (dim, s) in base.gp
        gp[dim] = GPSettings(
            s.dim, something(gp_epochs, s.epochs), s.inducing_points,
            something(gp_lr, s.learning_rate), s.optimizer,
            s.init_variance, s.init_lengthscale, s.init_noise_std, s.jitter,
            s.seed, s.z_score, s.minibatch_threshold, s.minibatch_batch_size,
            s.precision, s.provenance * " [ENV override applied -- NON-CANONICAL]",
        )
    end

    return CanonicalRegistry(base.schema_version, nn, gp, false, overrides)
end

end # module ComparatorRegistry
