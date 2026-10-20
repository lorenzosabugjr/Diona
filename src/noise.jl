# Noise-bound estimation (manuscript, Section 4.1).
#
# ε̄ is rarely known a priori. The manuscript delegates its estimation to the
# set-membership approach of Novara & Fagiano (2016): look at the spread of
# measurements inside small balls, where the function itself barely varies, and
# read half the mean spread as the noise half-width ε̃.

"""
    estimate_noise_bound(samples; x_cols, z_col, rho_start_fraction=0.01,
                         rho_increment_fraction=0.005, min_ball_points=2,
                         min_ball_coverage=0.0) -> ε̃

Set-membership estimate of the noise bound ε̄ (Novara & Fagiano, 2016).

The ball radius ρ grows linearly from `rho_start_fraction` to larger fractions
of the domain diameter in steps of `rho_increment_fraction`. At each radius,
every sample whose ball holds at least `max(2, min_ball_points)` samples
contributes its local spread `max z − min z`. The search stops at the first
radius where the contributing balls have mean occupancy ≥ `min_ball_points`
and cover at least `min_ball_coverage` of the samples; ε̃ is half the mean
spread. Returns `Inf` if no radius qualifies and `0.0` for a degenerate domain.

The estimate is biased low on sparse data: for `k` points i.i.d. uniform on
`[−h, h]`, `E[max − min] = 2h (k−1)/(k+1)`, so a ball with two points gives
`ε̃ ≈ h/3`. An under-estimated ε̃ can make the admissible set empty
(`:INFEASIBLE`) rather than merely tight; raising `min_ball_points` trades
some signal variation for a less biased estimate.
"""
function estimate_noise_bound(samples::AbstractMatrix{<:Real};
                              x_cols::AbstractVector{<:Integer},
                              z_col::Integer,
                              rho_start_fraction::Float64=0.01,
                              rho_increment_fraction::Float64=0.005,
                              min_ball_points::Int=2,
                              min_ball_coverage::Float64=0.0)
    n_samples, n_cols = size(samples)
    n_samples > 1 || throw(ArgumentError("need at least 2 samples to estimate noise"))
    rho_start_fraction > 0 || throw(ArgumentError("rho_start_fraction must be positive"))
    rho_increment_fraction > 0 || throw(ArgumentError("rho_increment_fraction must be positive"))
    isempty(x_cols) && throw(ArgumentError("x_cols cannot be empty"))
    all(1 .<= x_cols .<= n_cols) || throw(ArgumentError("x_cols out of bounds"))
    1 <= z_col <= n_cols || throw(ArgumentError("z_col out of bounds"))
    z_col in x_cols && throw(ArgumentError("z_col must differ from x_cols"))

    w = Matrix{Float64}(samples[:, x_cols])
    z = Vector{Float64}(samples[:, z_col])

    diameter = norm(vec(maximum(w, dims=1)) .- vec(minimum(w, dims=1)))
    diameter == 0.0 && return 0.0

    rho_base = diameter * rho_start_fraction
    rho_step = diameter * rho_increment_fraction
    max_iter = ceil(Int, max(0.0, 1.0 - rho_start_fraction) / rho_increment_fraction) + 2

    tree = KDTree(permutedims(w); leafsize=10)
    local_variations = Float64[]
    ball_sizes = Int[]

    for k in 0:max_iter
        rho = rho_base + k * rho_step
        empty!(local_variations); empty!(ball_sizes)
        for t in 1:n_samples
            idxs = inrange(tree, @view(w[t, :]), rho, true)
            min_z = z[t]; max_z = z[t]; count = 0
            @inbounds for idx in idxs
                count += 1
                v = z[idx]
                v < min_z && (min_z = v)
                v > max_z && (max_z = v)
            end
            if count >= max(2, min_ball_points)
                push!(local_variations, max_z - min_z)
                push!(ball_sizes, count)
            end
        end
        if !isempty(local_variations) &&
           mean(ball_sizes) >= min_ball_points &&
           length(local_variations) >= min_ball_coverage * n_samples
            return mean(local_variations) / 2
        end
    end
    return Inf
end
