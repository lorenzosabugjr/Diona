# Linear programs via HiGHS (manuscript, Table 1: "LPs are computed with HiGHS.jl").
#
# Every Diona solve is an LP of the form
#
#     min / max  cᵀ x    s.t.  A x ≤ b,   l ≤ x ≤ u,
#
# passed to HiGHS through its C API (no modelling layer), one instance per call.
# Diona reads only the primal solution. Statuses map to:
#   :OPTIMAL, :INFEASIBLE, :UNBOUNDED, :SOLVE_FAILED (time limit or numerical error).

const LP_TIME_LIMIT_S = 60.0
const LH_LP_TIME_LIMIT_S = 10.0

struct LPSolution
    status::Symbol
    x::Vector{Float64}
    objective::Float64
end

function _highs_status(code::Integer)
    code == HiGHS.kHighsModelStatusOptimal && return :OPTIMAL
    code == HiGHS.kHighsModelStatusInfeasible && return :INFEASIBLE
    code == HiGHS.kHighsModelStatusUnbounded && return :UNBOUNDED
    code == HiGHS.kHighsModelStatusUnboundedOrInfeasible && return :UNBOUNDED_OR_INFEASIBLE
    return :SOLVE_FAILED
end

"""
    _highs_pass(A, b, cost, col_lower, col_upper; time_limit) -> highs pointer

Create a HiGHS instance holding `min costᵀx s.t. A x ≤ b, col_lower ≤ x ≤ col_upper`.
The caller owns the pointer and must `Highs_destroy` it.
"""
function _highs_pass(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64},
                     cost::Vector{Float64}, col_lower::Vector{Float64},
                     col_upper::Vector{Float64}; time_limit::Float64)
    m, n = size(A)
    length(b) == m && length(cost) == n && length(col_lower) == n && length(col_upper) == n ||
        throw(DimensionMismatch("LP data sizes are inconsistent"))
    h = HiGHS.Highs_create()
    h == C_NULL && error("Highs_create returned NULL")
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", 0)
    HiGHS.Highs_setIntOptionValue(h, "threads", 1)
    HiGHS.Highs_setDoubleOptionValue(h, "time_limit", time_limit)
    # Dual simplex without presolve: on these small dense-ish LPs presolve costs
    # more than it saves (measured 5.3 → 4.0 ms per bound pair), and the optimum
    # is identical to 1e-11.
    HiGHS.Highs_setStringOptionValue(h, "presolve", "off")
    inf = HiGHS.Highs_getInfinity(h)
    row_lower = fill(-inf, m)
    row_upper = copy(b)
    lo = map(v -> isfinite(v) ? v : (v < 0 ? -inf : inf), col_lower)
    up = map(v -> isfinite(v) ? v : (v < 0 ? -inf : inf), col_upper)
    a_start = HiGHS.HighsInt.(@view(A.colptr[1:n]) .- 1)
    a_index = HiGHS.HighsInt.(A.rowval .- 1)
    status = HiGHS.Highs_passLp(h, HiGHS.HighsInt(n), HiGHS.HighsInt(m), HiGHS.HighsInt(nnz(A)),
                                HiGHS.kHighsMatrixFormatColwise, HiGHS.kHighsObjSenseMinimize,
                                0.0, cost, lo, up, row_lower, row_upper,
                                a_start, a_index, A.nzval)
    if status == HiGHS.kHighsStatusError
        HiGHS.Highs_destroy(h)
        error("Highs_passLp failed")
    end
    return h
end

function _highs_solve(h, n::Int)
    HiGHS.Highs_run(h)
    st = _highs_status(HiGHS.Highs_getModelStatus(h))
    st == :UNBOUNDED_OR_INFEASIBLE && (st = :SOLVE_FAILED)   # only presolve reports this; it is off
    st == :OPTIMAL || return LPSolution(st, Float64[], NaN)
    x = Vector{Float64}(undef, n)
    HiGHS.Highs_getSolution(h, x, C_NULL, C_NULL, C_NULL)
    return LPSolution(:OPTIMAL, x, HiGHS.Highs_getObjectiveValue(h))
end

"""
    solve_lp(A, b, cost; col_lower=-Inf, col_upper=Inf, time_limit) -> LPSolution

Minimise `costᵀ x` subject to `A x ≤ b` and the column bounds.
"""
function solve_lp(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}, cost::Vector{Float64};
                  col_lower::Vector{Float64}=fill(-Inf, length(cost)),
                  col_upper::Vector{Float64}=fill(Inf, length(cost)),
                  time_limit::Float64=LP_TIME_LIMIT_S)
    h = _highs_pass(A, b, cost, col_lower, col_upper; time_limit=time_limit)
    try
        return _highs_solve(h, length(cost))
    finally
        HiGHS.Highs_destroy(h)
    end
end

"""
    extremal_values(A, b, j; time_limit) -> (lower, upper, status)

The minimum and maximum of coordinate `x_j` over `{x : A x ≤ b}` (equations
(11a)–(11b) with `x_j = u'`). One HiGHS instance serves both LPs: the maximum
re-uses the basis of the minimum. On any failure both values are `nothing`
and `status` names the first failure.
"""
function extremal_values(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}, j::Integer;
                         time_limit::Float64=LP_TIME_LIMIT_S)
    n = size(A, 2)
    1 <= j <= n || throw(BoundsError(1:n, j))
    cost = zeros(n); cost[j] = 1.0
    h = _highs_pass(A, b, cost, fill(-Inf, n), fill(Inf, n); time_limit=time_limit)
    try
        lo = _highs_solve(h, n)
        lo.status == :OPTIMAL || return nothing, nothing, lo.status
        HiGHS.Highs_changeObjectiveSense(h, HiGHS.kHighsObjSenseMaximize)
        hi = _highs_solve(h, n)
        hi.status == :OPTIMAL || return nothing, nothing, hi.status
        return lo.x[j], hi.x[j], :OPTIMAL
    finally
        HiGHS.Highs_destroy(h)
    end
end
