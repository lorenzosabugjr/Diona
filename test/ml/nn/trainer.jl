# Data-only Lux MLP training with deterministic initialization and minibatching.
#
# DEVICE AND PRECISION
#
# `train_nn` takes both as arguments; it names no float type and no device of
# its own. The campaign supplies them from ComparatorRegistry (precision) and
# --device (device), so a change of either rotates the :nn settings hash.
#
# Everything crossing this module's boundary is Float64: `X_data`/`u_data` come
# in Float64, `nn_predict` takes a Float64 vector and returns a Float64 scalar,
# and `loss_history` is Float64. The working precision applies strictly INSIDE:
# the training arrays and the parameters are converted on entry and the
# parameters are converted back to host arrays on exit, so a GPU-trained model
# serialises, checkpoints and predicts exactly like a CPU-trained one.
#
# Determinism: initialisation draws from the host RNG at `seed` and happens on
# the host, so the initial parameters are identical on both devices at a given
# precision. The Adam loop is not: a GPU reduction sums in a different order, so
# a CPU fit and a CUDA fit at the same seed agree only to floating-point
# tolerance. The seed is reproducible WITHIN a device.

module NNTrainer

using Printf
using Lux                    # re-exports cpu_device/gpu_device/CPUDevice from MLDataDevices
using Optimisers
using Zygote
using Random
using SHA

export build_mlp, nn_predict, train_nn,
       nn_output, nn_data_loss, nn_params_hash, resolve_device

# LuxCUDA is what registers MLDataDevices' CUDA device, so `gpu_device()` cannot
# see the GPU until it is loaded. It is loaded HERE, at module definition, and
# not lazily inside resolve_device: a runtime `@eval using` would leave us
# calling the extension's freshly defined `functional` methods from a stale
# world age. Guarded, because this same file is included under
# --project=test/ml, which has neither Lux nor CUDA; there the CPU path is
# unaffected and --device=cuda errors with an actionable message.
const _LUXCUDA_LOADED = try
    @eval using LuxCUDA
    true
catch err
    @debug "LuxCUDA unavailable in this project; --device=cuda will error" exception=err
    false
end

"""
    resolve_device(device) -> MLDataDevices device

Map "cpu"/"cuda" onto a Lux device object. `--device=cuda` on a host with no
functional CUDA device is a hard error, never a silent fall back to the CPU:
the campaign records the requested device in the :nn settings hash, so a
fragment that claims CUDA must actually have run on it.
"""
function resolve_device(device::AbstractString)
    d = lowercase(strip(String(device)))
    d == "cpu" && return cpu_device()
    if d == "cuda"
        _LUXCUDA_LOADED ||
            error("device=cuda requested but LuxCUDA is not available in this Julia " *
                  "project ($(Base.active_project())); the :nn arm runs under " *
                  "test/ml/nn_env, which resolves CUDA and LuxCUDA")
        dev = gpu_device(; force=true)
        dev isa CPUDevice &&
            error("device=cuda requested but gpu_device() resolved to the CPU; " *
                  "load CUDA/LuxCUDA and check CUDA.functional()")
        return dev
    end
    error("unsupported device: $(repr(String(device))); expected \"cpu\" or \"cuda\"")
end

"""
    _assert_on_device(dev, label, x)

Hard check that `x` really reached `dev`, run after every move in `train_nn`.

A silent host fallback is the failure mode worth guarding: LuxCUDA absent so the
adaptor no-ops, an array that never left the host, a structure the adaptor did
not recurse into. None of those raise anything by themselves -- they surface
only as a disappointing wall-clock, which is indistinguishable from "the model
is too small to benefit". Assert it instead, so a fallback is an ERROR and a
timing can be trusted.
"""
function _assert_on_device(dev, label::AbstractString, x)
    actual = Lux.get_device_type(x)
    if dev isa CPUDevice
        actual <: CPUDevice ||
            error("train_nn: $label should be on the host but reports $(actual)")
    else
        actual <: CPUDevice &&
            error("train_nn: $label did not reach $(dev) -- it is still a host " *
                  "array ($(actual)). The fit would silently run on the CPU at " *
                  "the requested precision and merely look slow.")
    end
    return nothing
end

_to_precision(::Type{Float32}, x) = Lux.f32(x)
_to_precision(::Type{Float64}, x) = Lux.f64(x)
_to_precision(::Type{T}, x) where {T} =
    error("unsupported working precision: $T; expected Float32 or Float64")

# First numeric leaf of a Lux parameter tree. The types are concrete, so this
# constant-folds -- it is called once per query and must not allocate.
_params_eltype(x::AbstractArray{T}) where {T<:Number} = T
_params_eltype(x::Union{NamedTuple,Tuple}) = isempty(x) ? Float64 : _params_eltype(first(x))
_params_eltype(::Any) = Float64

function build_mlp(
    n_dim::Int;
    hidden_width::Int=64,
    n_hidden_layers::Int=3,
)
    hidden = ntuple(n_hidden_layers) do layer
        n_in = layer == 1 ? n_dim : hidden_width
        Lux.Dense(n_in => hidden_width, tanh)
    end
    return Lux.Chain(hidden..., Lux.Dense(hidden_width => 1))
end

function _nn_output(model, ps, st, x::AbstractArray)
    y, _ = model(x, ps, st)
    return only(y)
end

"""
    nn_predict(model, ps, st, x::AbstractVector{Float64}) -> Float64

Public prediction shape, Float64 in and Float64 out whatever the working
precision was. The query is converted to the parameters' own element type so
the forward pass runs entirely in that precision and the result is widened
once, at the end.

Prediction always runs on the host: `train_nn` returns host parameters.
"""
function nn_predict(model, ps, st, x::AbstractVector{Float64})::Float64
    xw = convert(Vector{_params_eltype(ps)}, x)
    return Float64(_nn_output(model, ps, st, xw))
end

nn_output(model, ps, st, x::AbstractArray) = _nn_output(model, ps, st, x)

function _nn_data_loss(model, ps, st, X_data, u_data)
    u_pred, _ = model(X_data, ps, st)
    return sum(abs2, vec(u_pred) .- u_data) / length(u_data)
end

nn_data_loss(model, ps, st, X_data, u_data) = Float64(_nn_data_loss(model, ps, st, X_data, u_data))

function nn_params_hash(ps)
    flat, _ = Optimisers.destructure(ps)
    return bytes2hex(sha256(reinterpret(UInt8, collect(Float64.(flat)))))
end

function train_nn(
    X_data::Matrix{Float64},
    u_data::Vector{Float64};
    hidden_width::Int=64,
    n_hidden_layers::Int=3,
    epochs::Int=3000,
    lr::Float64=1e-3,
    seed::Int=0,
    precision::Type{<:AbstractFloat}=Float64,
    device::AbstractString="cpu",
    batch_size::Int=0,
    print_every::Int=500,
    init_callback::Union{Nothing,Function}=nothing,
    lr_schedule::Union{Nothing,Function}=nothing,
    check_finite_gradients::Bool=false,
)
    model = build_mlp(
        size(X_data, 1);
        hidden_width=hidden_width,
        n_hidden_layers=n_hidden_layers,
    )

    T = precision
    dev = resolve_device(device)
    host = cpu_device()
    on_device = !(dev isa CPUDevice)

    # Initialisation is host-side and precision-only, so `init_callback` sees
    # the same parameters on either device and the init hash stays comparable.
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = _to_precision(T, Lux.setup(rng, model))
    init_callback !== nothing && init_callback(ps)

    ps = dev(ps)
    st = dev(st)
    X_work = dev(T.(X_data))
    u_work = dev(T.(u_data))
    _assert_on_device(dev, "parameters", ps)
    _assert_on_device(dev, "training inputs", X_work)
    _assert_on_device(dev, "training targets", u_work)
    eltype(X_work) === T && eltype(u_work) === T ||
        error("train_nn: training arrays are $(eltype(X_work))/$(eltype(u_work)), " *
              "expected $T -- the working precision did not take effect")
    if print_every > 0
        @printf("train_nn: device=%s precision=%s params=%s data=%s\n",
                device, T, typeof(ps.layer_1.weight), typeof(X_work))
    end

    opt_state = Optimisers.setup(Optimisers.Adam(T(lr)), ps)
    loss_history = Vector{Float64}(undef, epochs)
    n_data = size(X_data, 2)
    use_minibatch = batch_size > 0 && batch_size < n_data
    batch_rng = Random.MersenneTwister(seed + 10_000)
    batch_order = collect(1:n_data)
    use_minibatch && Random.shuffle!(batch_rng, batch_order)
    batch_cursor = 1

    for epoch in 1:epochs
        lr_schedule !== nothing && Optimisers.adjust!(opt_state, lr_schedule(epoch))
        if use_minibatch
            if batch_cursor + batch_size - 1 > n_data
                Random.shuffle!(batch_rng, batch_order)
                batch_cursor = 1
            end
            idx = view(batch_order, batch_cursor:(batch_cursor + batch_size - 1))
            batch_cursor += batch_size
            # A device array indexed by a scattered index vector must be
            # gathered into a dense array; a host array keeps the view it
            # always used, so the CPU path is unchanged.
            if on_device
                idx_dev = dev(collect(idx))
                Xb = X_work[:, idx_dev]
                ub = u_work[idx_dev]
            else
                Xb = view(X_work, :, idx)
                ub = view(u_work, idx)
            end
        else
            Xb = X_work
            ub = u_work
        end
        data_loss, pullback = Zygote.pullback(
            current_ps -> _nn_data_loss(model, current_ps, st, Xb, ub),
            ps,
        )
        grad_ps = only(pullback(one(T)))
        if check_finite_gradients
            flat_gradient, _ = Optimisers.destructure(grad_ps)
            all(isfinite, flat_gradient) ||
                error("non-finite gradient before optimizer update $epoch")
        end
        total = Float64(data_loss)
        loss_history[epoch] = total

        if print_every > 0 && epoch % print_every == 0
            @printf(
                "epoch %d: data_loss=%.8e\n", epoch, total,
            )
        end

        opt_state, ps = Optimisers.update!(opt_state, ps, grad_ps)
    end

    # Back to host arrays: the caller checkpoints, serialises and predicts with
    # these, and none of that should know which device fitted them. Moving them
    # also forces any outstanding device work to complete, so the caller's
    # `@elapsed` around this call measures the whole fit.
    #
    # Asserted before the move as well as after: the parameters must still be on
    # the fitting device at the END of the loop, not just at the start. An
    # optimiser step that quietly reconstructed them as host arrays would
    # otherwise leave the whole loop running on the CPU undetected.
    _assert_on_device(dev, "parameters after training", ps)
    ps_host, st_host = host(ps), host(st)
    _assert_on_device(host, "returned parameters", ps_host)
    return model, ps_host, st_host, loss_history
end

end # module NNTrainer
