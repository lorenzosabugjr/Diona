# Sparse variational Gaussian process (SVGP) comparator arm for the Part-1
# data-only benchmark (Diona vs NN vs GP, 5 synthetic functions, 3-8 dims).
#
# Library choice: AbstractGPs.jl (GP/kernel primitives, `posterior`, `mean_and_var`)
# + ApproximateGPs.jl's `SparseVariationalApproximation` (the Hensman, Matthews &
# Ghahramani 2015 SVGP: a variational distribution q(u) = N(m, S) over M inducing
# "pseudo-points" u = f(z), optimised jointly with the kernel hyperparameters and
# the inducing locations z against the ELBO). This is the standard scalable-GP
# building block in the Julia GP ecosystem and is what the PRD names explicitly.
#
# Kernel: ARD Matern-5/2 (`KernelFunctions.Matern52Kernel()` with a per-dimension
# lengthscale via `with_lengthscale`), scaled by a learned output-variance.
#
# Training: mirrors nn/trainer.jl's structure (build model -> `Optimisers.setup`
# -> loop of `Zygote.pullback` + `Optimisers.update!`), but the objective being
# minimised is the negative ELBO -- the idiomatic ApproximateGPs.jl training
# target -- rather than a bespoke data loss. All trainable quantities
# (kernel hyperparameters, inducing locations, variational mean/covariance) are
# optimised jointly in a single NamedTuple of parameters, exactly as
# ApproximateGPs' own regression example does (see `examples/a-regression` in the
# ApproximateGPs.jl package), just swapping Flux.jl for Optimisers.jl+Zygote.jl.
#
# ==============================================================================
# PRD HARD REQUIREMENT -- variational, not exact, posterior:
#
# The credible interval produced by `gp_predict` is the **variational /
# approximate** SVGP posterior interval (mean +/- z*std of q(f*), the sparse
# variational posterior), NOT the exact-GP-regression posterior interval. It
# coincides with the exact posterior only in the special case where the
# inducing points span the training inputs and q(u) attains its KL-optimal
# form (see ApproximateGPs.jl's own "GPR/VFE equivalences" test). In the
# general M << N regime this benchmark uses, the SVGP interval is an
# *approximation* to the true GP posterior uncertainty, with no formal
# coverage guarantee. (Diona's interval is not a guarantee either: it is an
# admissible-set projection under ESTIMATED noise and smoothness constants, so
# neither arm's interval is certified -- they are different approximations.)
# Every place this
# interval is reported (plots, tables, benchmark writeups) MUST disclose this
# "variational/approximate credible interval" qualifier -- do not let it get
# silently relabeled as "the GP posterior" or "the exact credible interval".
# ==============================================================================
#
# Promoted unmodified in behavior from `test/sandbox/ml/train_gp.jl`,
# see `plans/plan_20260821_125209.md` / `plans/prd_20260821_125209.md`).
# Public API: `train_gp`, `gp_predict`.

module GPTrainer

using AbstractGPs
using ApproximateGPs
using KernelFunctions
using Distributions
using PDMats: PDMat
using QuasiMonteCarlo
using LinearAlgebra
using Optimisers
using Zygote
using Random
using Statistics
using Printf

export train_gp, gp_posterior, gp_predict

# ------------------------------------------------------------------------------
# Kernel / variational-covariance helpers
# ------------------------------------------------------------------------------

"""
    _make_matern52_kernel(log_variance, log_lengthscale) -> Kernel

ARD Matern-5/2 kernel: `variance * Matern52Kernel()` with a per-input-dimension
lengthscale (`KernelFunctions.with_lengthscale`). Hyperparameters live in
log-space (`log_variance`, `log_lengthscale`) so unconstrained gradient steps
keep `variance` and every `lengthscale[d]` positive via `exp`.

`log_variance` and `log_lengthscale` are `Vector{Float64}` (length 1 and D
respectively), not bare `Float64`/`Float64`-vector-of-1 scalars, deliberately --
see the note in `train_gp` on why bare scalar fields silently fail to train
under `Optimisers.update!`.
"""
function _make_matern52_kernel(
    log_variance::AbstractVector{<:Real}, log_lengthscale::AbstractVector{<:Real}
)
    variance = exp(log_variance[1])
    lengthscale = exp.(log_lengthscale)
    return variance * with_lengthscale(Matern52Kernel(), lengthscale)
end

"""
    _variational_cov(A::AbstractMatrix{<:Real}) -> PDMats.PDMat

Build the variational covariance `S = L * L'` (`L = LowerTriangular(A)`),
wrapped as a `PDMat` for `Distributions.MvNormal`. Only the lower triangle of
`A` is used -- `LowerTriangular` masks the strict upper triangle to zero --
which is the usual unconstrained parametrisation of an SPD matrix via its own
Cholesky factor (`A` is free, `S` is guaranteed positive semi-definite by
construction for any `A`).

Implementation note: ApproximateGPs.jl's own examples/tests build `S` as
`PDMat(Cholesky(LowerTriangular(A)))` (constructing a `Cholesky` object
directly from the triangular factor, avoiding a redundant `cholesky()` call).
That specific `Cholesky(::LowerTriangular)` constructor has **no working
Zygote adjoint** on the Zygote/Julia versions pinned in this project's
Manifest -- confirmed directly:

    julia> Zygote.gradient(A -> sum(Cholesky(LowerTriangular(A)).L), A)
    ERROR: MethodError: no method matching LowerTriangular(::@NamedTuple{data::LowerTriangular{...}})

i.e. the pullback for `Cholesky(::LowerTriangular)` returns a tangent that the
next pullback (`LowerTriangular`'s own) can't consume. Building `S` explicitly
as `L*L'` and letting `PDMat(Symmetric(S))` call the *ordinary* `cholesky`
(which composes fine under Zygote) sidesteps the issue; numerically it is the
same matrix.
"""
function _variational_cov(A::AbstractMatrix{<:Real})
    L = LowerTriangular(A)
    return PDMat(Symmetric(L * L' + 1.0e-10I))
end

"""
    _svgp_parts(ps, jitter) -> (f, fz, svgp)

Assemble the prior GP `f`, the inducing-point `FiniteGP` `fz`, and the
`SparseVariationalApproximation` from the current parameter NamedTuple `ps`.
Shared by the training loss and by posterior construction after training, so
the two can never accidentally diverge in how they build the model.
"""
function _svgp_parts(ps, jitter::Float64)
    kernel = _make_matern52_kernel(ps.log_variance, ps.log_lengthscale)
    f = GP(kernel)
    fz = f(ColVecs(ps.z), jitter)
    q = MvNormal(ps.m, _variational_cov(ps.A))
    return f, fz, SparseVariationalApproximation(fz, q)
end

"""Reconstruct the variational posterior from saved trained parameters."""
function gp_posterior(ps; jitter::Float64=1.0e-6)
    _, _, svgp = _svgp_parts(ps, jitter)
    return posterior(svgp)
end

function _neg_elbo(ps, X_cols::ColVecs, y::Vector{Float64}, jitter::Float64)
    f, _, svgp = _svgp_parts(ps, jitter)
    sigma2 = exp(ps.log_noise[1])
    fx = f(X_cols, sigma2)
    return -elbo(svgp, fx, y)
end

# ------------------------------------------------------------------------------
# Inducing-point initialisation
# ------------------------------------------------------------------------------

"""
    _init_inducing_points(X_data::Matrix{Float64}, M::Int, rng::AbstractRNG) -> Matrix{Float64}

Initial (D x M) inducing-point locations. `z` is a free variational parameter
that moves under gradient descent during training, so this only needs to be a
*reasonable* starting layout -- but a degenerate start (e.g. all M points
piled on top of a handful of training points, which can happen once
`M = 100*D` starts to exceed the training-set size N at higher D) makes
`Kuu` ill-conditioned and slows optimisation, so we branch on `M` vs `N`:

  * `M <= N` (the common case at low D / large N): a uniform
    without-replacement subsample of the N training inputs. This is the
    standard SVGP/GPflow default initialisation.
  * `M >  N` (common once D grows and `M = 100*D` outstrips the training-set
    size): keep all N training points -- they're free, already-informative
    starting locations -- and fill the remaining `M - N` inducing points with
    a Latin Hypercube space-filling design (`QuasiMonteCarlo.jl`) over the
    bounding box of the training inputs, so the extra inducing points spread
    out to cover the input domain (including regions the training sample
    happens to under-sample) rather than clumping.
"""
function _init_inducing_points(X_data::Matrix{Float64}, M::Int, rng::AbstractRNG)
    D, N = size(X_data)
    if M <= N
        idx = randperm(rng, N)[1:M]
        return X_data[:, idx]
    end

    lo = vec(minimum(X_data; dims=2))
    hi = vec(maximum(X_data; dims=2))
    # Guard degenerate (zero-width, e.g. constant-column) dimensions so
    # QuasiMonteCarlo doesn't choke on an empty sampling interval.
    flat = hi .== lo
    if any(flat)
        lo = copy(lo)
        hi = copy(hi)
        lo[flat] .-= 0.5
        hi[flat] .+= 0.5
    end
    extra = QuasiMonteCarlo.sample(M - N, lo, hi, QuasiMonteCarlo.LatinHypercubeSample())
    return hcat(X_data, extra)
end

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

"""
    train_gp(
        X_data::Matrix{Float64},
        u_data::Vector{Float64};
        M::Int=100 * size(X_data, 1),
        epochs::Int=2000,
        lr::Float64=1e-2,
        seed::Int=0,
        init_variance::Float64=1.0,
        init_lengthscale::Float64=1.0,
        init_noise_std::Float64=0.1,
        jitter::Float64=1e-6,
        print_every::Int=200,
    ) -> (posterior, ps, loss_history)

Train an SVGP (ARD Matern-5/2 kernel) on `(X_data, u_data)` by minimising the
negative ELBO jointly over the kernel hyperparameters, the observation-noise
variance, the inducing-point locations `z` (D x M), and the variational
distribution `q(u) = N(m, S)`.

`X_data` is `D x N` (n_dim x N_data), matching `train_nn`'s layout in
`nn/trainer.jl`. The training-loop *shape* mirrors `train_nn` (build
model -> `Optimisers.setup` -> loop of `Zygote.pullback` + `Optimisers.update!`),
but the objective is the SVGP ELBO -- ApproximateGPs.jl's own idiomatic
training target -- rather than a bespoke data loss.

Implementation note on why every trainable scalar is stored as a
`Vector{Float64}` of length 1 (`log_variance`, `log_noise`) rather than a bare
`Float64`: `Optimisers.jl` only treats `AbstractArray` leaves of the parameter
tree as trainable (`Optimisers.isnumeric` requires `x isa AbstractArray`).
A bare `Float64` field sitting next to array fields in the same NamedTuple
gets a real, nonzero gradient from Zygote, but `Optimisers.update!` silently
leaves it bit-for-bit unchanged every step -- confirmed directly:

    julia> Optimisers.update!(Optimisers.setup(Adam(0.1), (a=1.0, b=[1.0])),
                               (a=1.0, b=[1.0]), (a=5.0, b=[5.0]))
    (a = 1.0, b = [0.9...])   # `a` never moved despite a real gradient

Wrapping every scalar hyperparameter as a length-1 vector sidesteps this
footgun so the kernel variance and observation noise actually train.

Returns:
  * `posterior` -- the trained `AbstractGPs` approximate-posterior GP
    (`ApproxPosteriorGP{<:SparseVariationalApproximation}`); pass this to
    `gp_predict`.
  * `ps` -- the final NamedTuple of trained parameters (kernel + inducing
    points + variational distribution), in case a caller wants to inspect or
    reuse them.
  * `loss_history` -- per-epoch negative-ELBO trace (mirrors
    `train_nn`'s `loss_history`).

See the module-level docstring / comment block at the top of this file for
the PRD-required disclosure: the interval `gp_predict` derives from
`posterior` is a **variational/approximate** credible interval, not the exact
GP-regression posterior interval.
"""
function train_gp(
    X_data::Matrix{Float64},
    u_data::Vector{Float64};
    M::Int=100 * size(X_data, 1),
    epochs::Int=2000,
    lr::Float64=1.0e-2,
    seed::Int=0,
    init_variance::Float64=1.0,
    init_lengthscale::Float64=1.0,
    init_noise_std::Float64=0.1,
    jitter::Float64=1.0e-6,
    batch_size::Int=0,
    print_every::Int=200,
)
    D, N = size(X_data)
    length(u_data) == N ||
        error("u_data length $(length(u_data)) does not match X_data columns $N")
    M >= 1 || error("M must be >= 1")

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    Z_init = _init_inducing_points(X_data, M, rng)
    M_eff = size(Z_init, 2)  # == M always (subsample or subsample+fill both return M columns)

    ps = (
        log_variance=[log(init_variance)],
        log_lengthscale=fill(log(init_lengthscale), D),
        log_noise=[log(init_noise_std^2)],
        z=Z_init,
        m=zeros(M_eff),
        A=Matrix{Float64}(sqrt(init_variance) * I, M_eff, M_eff),
    )

    X_cols = ColVecs(X_data)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
    loss_history = Vector{Float64}(undef, epochs)

    # Deterministic cyclic minibatching via ApproximateGPs' own `num_data`
    # ELBO-scaling keyword (test/ml/common/registry.jl's
    # `minibatch_threshold`/`minibatch_batch_size` convention -- promoted in
    # behavior from the original campaign script's `train_gp_minibatch`,
    # reimplemented here so it lives with the trainer it belongs to rather
    # than a second copy in a campaign script). `batch_size=0` (default)
    # preserves the exact prior full-batch behavior/output.
    use_minibatch = batch_size > 0 && batch_size < N
    batch_rng = Random.MersenneTwister(seed + 20_000)
    batch_order = collect(1:N)
    use_minibatch && Random.shuffle!(batch_rng, batch_order)
    batch_cursor = 1

    full_loss_fn = current_ps -> _neg_elbo(current_ps, X_cols, u_data, jitter)

    for epoch in 1:epochs
        if use_minibatch
            if batch_cursor + batch_size - 1 > N
                Random.shuffle!(batch_rng, batch_order)
                batch_cursor = 1
            end
            idx = view(batch_order, batch_cursor:(batch_cursor + batch_size - 1))
            batch_cursor += batch_size
            Xb = ColVecs(X_data[:, idx])
            ub = u_data[idx]
            loss_fn = function (current_ps)
                f, _, svgp = _svgp_parts(current_ps, jitter)
                fx = f(Xb, exp(current_ps.log_noise[1]))
                return -elbo(svgp, fx, ub; num_data=N)
            end
        else
            loss_fn = full_loss_fn
        end
        loss_val, pullback = Zygote.pullback(loss_fn, ps)
        grad_ps = only(pullback(1.0))
        loss_history[epoch] = Float64(loss_val)

        if print_every > 0 && epoch % print_every == 0
            @printf("epoch %d: neg_elbo=%.8e\n", epoch, loss_history[epoch])
        end

        opt_state, ps = Optimisers.update!(opt_state, ps, grad_ps)
    end

    trained_posterior = gp_posterior(ps; jitter=jitter)

    return trained_posterior, ps, loss_history
end

# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------

"""
    gp_predict(posterior, x::AbstractVector{Float64}; z_score::Float64=2.0) ->
        (mean::Float64, lower::Float64, upper::Float64)

Point prediction + credible interval at a single query point `x` (length D),
from a `posterior` returned by `train_gp`. Signature/shape mirrors
`nn_predict(model, ps, st, x)` in `nn/trainer.jl`, and `Diona.infer`'s
`(nominal, lower, upper)`-style return, so a benchmark harness can call all
three arms uniformly.

`lower`/`upper` are `mean -+ z_score * std`, where `std` is the posterior
standard deviation of the **latent function** `f` at `x` (i.e. `var(f*)`,
NOT inflated by the learned observation noise `exp(ps.log_noise[1])`). This
is the correct analogue of Diona's admissible-set interval, which also bounds the
estimated *function value*, not a hypothetical future noisy observation.
`z_score=1.96` gives an (approximately -- see caveat below) 95% interval;
this file's validation harness uses `z_score=2.0` as the coarser "~95%"
convention used elsewhere in this benchmark.

# PRD caveat -- variational, not exact, posterior (repeated from module header)
`posterior` is an `ApproximateGPs.SparseVariationalApproximation` posterior:
`std` here is the **variational/approximate** SVGP posterior standard
deviation, not the exact-GP-regression posterior standard deviation. Any
report, plot, or table that surfaces `lower`/`upper` from this function MUST
carry that qualifier -- do not present it as "the GP posterior interval"
without the word "variational"/"approximate" attached.
"""
function gp_predict(posterior, x::AbstractVector{Float64}; z_score::Float64=2.0)
    mu, var = mean_and_var(posterior, [x])
    m = Float64(mu[1])
    sd = sqrt(max(Float64(var[1]), 0.0))
    return m, m - z_score * sd, m + z_score * sd
end

end # module GPTrainer
