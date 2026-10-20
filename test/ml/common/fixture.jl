# Versioned, checksummed transport for shared regression datasets.

module SmokeFixture

using SHA

export FixtureError,
       FIXTURE_MAGIC, FIXTURE_FORMAT_VERSION, SUPPORTED_SCHEMA_VERSIONS,
       ARRAY_FIELD_ORDER,
       write_fixture, read_fixture, validate_fixture,
       fixture_manifest_path

struct FixtureError <: Exception
    msg::String
end
Base.showerror(io::IO, e::FixtureError) = print(io, "FixtureError: ", e.msg)

const FIXTURE_MAGIC = "DIONAFX1"                 # 8 ASCII bytes
const FIXTURE_FORMAT_VERSION = UInt32(1)          # binary LAYOUT version
const CHECKSUM_NBYTES = 32                        # SHA-256 digest size
const SUPPORTED_SCHEMA_VERSIONS = ("3.0.0",)
const ARRAY_FIELD_ORDER = (:x_train, :y_train_clean, :y_train_noisy,
                            :x_query, :y_query_truth, :query_ids)

function _toml_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        c == '\\' ? print(io, "\\\\") :
        c == '"'  ? print(io, "\\\"") :
        c == '\n' ? print(io, "\\n")  :
        c == '\r' ? print(io, "\\r")  :
        print(io, c)
    end
    return String(take!(io))
end

function _toml_unescape(s::AbstractString)
    io = IOBuffer()
    i, n = firstindex(s), lastindex(s)
    while i <= n
        c = s[i]
        if c == '\\' && i < n
            j = nextind(s, i)
            nc = s[j]
            if nc == 'n'; print(io, '\n')
            elseif nc == 'r'; print(io, '\r')
            elseif nc == '\\'; print(io, '\\')
            elseif nc == '"'; print(io, '"')
            else; print(io, c); print(io, nc)
            end
            i = nextind(s, j)
        else
            print(io, c)
            i = nextind(s, i)
        end
    end
    return String(take!(io))
end

_toml_kv(k::AbstractString, v::AbstractString)              = "$k = \"" * _toml_escape(v) * "\""
_toml_kv(k::AbstractString, v::Bool)                        = "$k = $(v)"
_toml_kv(k::AbstractString, v::Integer)                     = "$k = $(Int(v))"
_toml_kv(k::AbstractString, v::AbstractFloat)               = "$k = $(string(Float64(v)))"
_toml_kv(k::AbstractString, v::AbstractVector{<:Integer})   = "$k = [" * join(string.(Int.(v)), ", ") * "]"

function _build_meta_text(scalar_pairs::AbstractVector, array_descs::AbstractVector)
    io = IOBuffer()
    for (k, v) in scalar_pairs
        println(io, _toml_kv(String(k), v))
    end
    for ad in array_descs
        println(io)
        println(io, "[[arrays]]")
        println(io, _toml_kv("name", ad.name))
        println(io, _toml_kv("shape", collect(Int, ad.shape)))
        println(io, _toml_kv("dtype", ad.dtype))
        println(io, _toml_kv("nbytes", ad.nbytes))
        println(io, _toml_kv("sha256", ad.sha256))
    end
    return String(take!(io))
end

function _parse_toml_value(v::AbstractString)
    if startswith(v, '"') && endswith(v, '"') && length(v) >= 2
        return _toml_unescape(chop(v; head=1, tail=1))
    elseif startswith(v, '[')
        endswith(v, ']') || throw(FixtureError("malformed fixture metadata: unterminated array literal: $v"))
        inner = strip(chop(v; head=1, tail=1))
        isempty(inner) && return Int[]
        return [parse(Int, strip(p)) for p in split(inner, ',')]
    else
        return String(v)   # raw numeric literal; caller parses via parse(Int/Float64, ...)
    end
end

function _parse_meta_text(text::AbstractString)
    scalars = Dict{String,Any}()
    arrays = Vector{Dict{String,Any}}()
    current = nothing
    for raw_line in split(text, '\n')
        line = strip(raw_line)
        (isempty(line) || startswith(line, '#')) && continue
        if line == "[[arrays]]"
            current = Dict{String,Any}()
            push!(arrays, current)
            continue
        end
        rng = findfirst(" = ", line)
        rng === nothing && throw(FixtureError("malformed fixture metadata line (no ' = '): $(repr(line))"))
        key = String(strip(line[1:first(rng)-1]))
        valtext = strip(line[last(rng)+1:end])
        val = _parse_toml_value(valtext)
        (current === nothing ? scalars : current)[key] = val
    end
    return scalars, arrays
end

_raw_bytes(a::Array{Float64}) = Vector{UInt8}(reinterpret(UInt8, vec(a)))
_raw_bytes(a::Array{Int64})   = Vector{UInt8}(reinterpret(UInt8, vec(a)))

_dtype_name(::Type{Float64}) = "Float64"
_dtype_name(::Type{Int64})   = "Int64"

function write_fixture(path::AbstractString;
        schema_version::AbstractString,
        cell_id::Integer, name::AbstractString, dim::Integer,
        layout::AbstractString, tier_label::AbstractString,
        n_train_nominal::Integer, n_train_actual::Integer, n_query::Integer,
        noise_level::Real, noise_bound::Real, noise_seed::Integer,
        model_seed::Integer, output_range::Real,
        x_train::AbstractMatrix{<:Real}, y_train_clean::AbstractVector{<:Real},
        y_train_noisy::AbstractVector{<:Real}, x_query::AbstractMatrix{<:Real},
        y_query_truth::AbstractVector{<:Real}, query_ids::AbstractVector{<:Integer},
        generated_at::AbstractString = "",
        manifest_path::Union{Nothing,AbstractString} = nothing,
    )
    xt  = Matrix{Float64}(x_train)
    ytc = Vector{Float64}(y_train_clean)
    ytn = Vector{Float64}(y_train_noisy)
    xq  = Matrix{Float64}(x_query)
    yqt = Vector{Float64}(y_query_truth)
    qid = Vector{Int64}(query_ids)

    dim_ = size(xt, 1)
    (size(xt, 2) == length(ytc) == length(ytn)) ||
        throw(FixtureError("x_train/y_train_clean/y_train_noisy column-count mismatch: " *
                            "$(size(xt,2))/$(length(ytc))/$(length(ytn))"))
    (size(xq, 2) == length(yqt) == length(qid)) ||
        throw(FixtureError("x_query/y_query_truth/query_ids column-count mismatch: " *
                            "$(size(xq,2))/$(length(yqt))/$(length(qid))"))
    size(xq, 1) == dim_ ||
        throw(FixtureError("x_train dim=$dim_ but x_query dim=$(size(xq,1))"))

    arrays_data = (
        (:x_train, xt), (:y_train_clean, ytc), (:y_train_noisy, ytn),
        (:x_query, xq), (:y_query_truth, yqt), (:query_ids, qid),
    )
    @assert Tuple(first.(arrays_data)) == ARRAY_FIELD_ORDER "arrays_data order drifted from ARRAY_FIELD_ORDER"

    array_bytes = Vector{UInt8}[]
    array_descs = NamedTuple[]
    for (fname, a) in arrays_data
        b = _raw_bytes(a)
        push!(array_bytes, b)
        push!(array_descs, (
            name = String(fname), shape = collect(Int, size(a)),
            dtype = _dtype_name(eltype(a)), nbytes = length(b),
            sha256 = bytes2hex(sha256(b)),
        ))
    end

    scalar_pairs = Pair{String,Any}[
        "fixture_format_version" => Int(FIXTURE_FORMAT_VERSION),
        "schema_version"         => String(schema_version),
        "generated_at"           => String(generated_at),
        "cell_id"                => Int(cell_id),
        "name"                   => String(name),
        "dim"                    => Int(dim),
        "layout"                 => String(layout),
        "tier_label"             => String(tier_label),
        "n_train_nominal"        => Int(n_train_nominal),
        "n_train_actual"         => Int(n_train_actual),
        "n_query"                => Int(n_query),
        "noise_level"            => Float64(noise_level),
        "noise_bound"            => Float64(noise_bound),
        "noise_seed"             => Int(noise_seed),
        "model_seed"             => Int(model_seed),
        "output_range"           => Float64(output_range),
    ]
    meta_text  = _build_meta_text(scalar_pairs, array_descs)
    meta_bytes = Vector{UInt8}(codeunits(meta_text))

    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())-$(string(rand(UInt32); base=16))"
    ctx = SHA256_CTX()
    open(tmp, "w") do io
        magic_b = Vector{UInt8}(codeunits(FIXTURE_MAGIC))
        fv_b    = Vector{UInt8}(reinterpret(UInt8, [FIXTURE_FORMAT_VERSION]))
        ml_b    = Vector{UInt8}(reinterpret(UInt8, [UInt32(length(meta_bytes))]))
        for b in (magic_b, fv_b, ml_b, meta_bytes, array_bytes...)
            write(io, b)
            update!(ctx, b)
        end
    end
    digest = digest!(ctx)   # 32 raw bytes; finalizes ctx (single use)
    open(tmp, "a") do io
        write(io, digest)
    end
    mv(tmp, path; force=true)

    fixture_sha256_hex = bytes2hex(digest)
    manifest_scalar_pairs = vcat(scalar_pairs, ["fixture_sha256" => fixture_sha256_hex])
    manifest_text = _build_meta_text(manifest_scalar_pairs, array_descs)
    mpath = manifest_path === nothing ? fixture_manifest_path(path) : manifest_path
    mkpath(dirname(mpath))
    mtmp = mpath * ".tmp-$(getpid())-$(string(rand(UInt32); base=16))"
    open(mtmp, "w") do io
        write(io, manifest_text)
    end
    mv(mtmp, mpath; force=true)

    n_bytes = filesize(path)
    return (path=path, manifest_path=mpath, fixture_sha256=fixture_sha256_hex, n_bytes=n_bytes)
end

fixture_manifest_path(fixture_path::AbstractString) = fixture_path * ".manifest.toml"

function read_fixture(path::AbstractString; check_manifest::Bool=true,
                       manifest_strict::Bool=false,
                       manifest_path::Union{Nothing,AbstractString}=nothing)
    isfile(path) || throw(FixtureError("fixture file not found: $path"))
    bytes = read(path)
    length(bytes) >= 16 + CHECKSUM_NBYTES ||
        throw(FixtureError("truncated fixture file ($(length(bytes)) bytes; need >= $(16+CHECKSUM_NBYTES)): $path"))

    magic = String(bytes[1:8])
    magic == FIXTURE_MAGIC ||
        throw(FixtureError("bad magic in $path: expected $(repr(FIXTURE_MAGIC)), got $(repr(magic)) " *
                            "— not a Diona smoke fixture, or the header is corrupted"))

    fv = only(reinterpret(UInt32, bytes[9:12]))
    fv == FIXTURE_FORMAT_VERSION ||
        throw(FixtureError("unsupported fixture binary format_version=$fv in $path " *
                            "(this SmokeFixture supports format_version=$(FIXTURE_FORMAT_VERSION)) " *
                            "— regenerate the fixture with a matching fixture.jl"))

    ml = Int(only(reinterpret(UInt32, bytes[13:16])))
    header_end = 16 + ml
    length(bytes) >= header_end + CHECKSUM_NBYTES ||
        throw(FixtureError("truncated fixture file: declared metadata length=$ml bytes but file is only " *
                            "$(length(bytes)) bytes total: $path"))

    meta_text = String(bytes[17:header_end])
    scalars, array_metas = _parse_meta_text(meta_text)

    payload = bytes[1:end-CHECKSUM_NBYTES]
    trailer = bytes[end-CHECKSUM_NBYTES+1:end]
    computed = sha256(payload)
    computed == trailer ||
        throw(FixtureError("checksum mismatch in $path: fixture is corrupted or was modified after writing " *
                            "(expected $(bytes2hex(trailer)), computed $(bytes2hex(computed)))"))
    fixture_sha256_hex = bytes2hex(trailer)

    schema_version = scalars["schema_version"]::String
    schema_version in SUPPORTED_SCHEMA_VERSIONS ||
        throw(FixtureError("unsupported PreparedSmokeCell schema_version=$(repr(schema_version)) in $path " *
                            "(this loader supports $(SUPPORTED_SCHEMA_VERSIONS)) " *
                            "— fixture was produced by an incompatible smoke_data.jl revision; regenerate it"))

    length(array_metas) == length(ARRAY_FIELD_ORDER) ||
        throw(FixtureError("fixture declares $(length(array_metas)) array(s), expected $(length(ARRAY_FIELD_ORDER)): $path"))
    for (am, expected_name) in zip(array_metas, ARRAY_FIELD_ORDER)
        am["name"] == String(expected_name) ||
            throw(FixtureError("fixture array order/name mismatch: expected '$expected_name', found '$(am["name"])' " *
                                "at that position in $path"))
    end

    offset = header_end
    arrays = Dict{Symbol,Array}()
    for am in array_metas
        name   = am["name"]::String
        shape  = Tuple(am["shape"]::Vector{Int})
        dtype  = am["dtype"]::String
        nbytes = parse(Int, am["nbytes"]::String)
        offset + nbytes <= length(bytes) - CHECKSUM_NBYTES ||
            throw(FixtureError("truncated fixture file: array '$name' extends past end of file: $path"))
        chunk = bytes[offset+1:offset+nbytes]
        got_sha = bytes2hex(sha256(chunk))
        expect_sha = am["sha256"]::String
        got_sha == expect_sha ||
            throw(FixtureError("component checksum mismatch for array '$name' in $path: expected $expect_sha, " *
                                "got $got_sha — data was modified, reordered, or corrupted after writing"))
        arr = dtype == "Float64" ? reinterpret(Float64, chunk) :
              dtype == "Int64"   ? reinterpret(Int64, chunk)   :
              throw(FixtureError("unknown array dtype '$dtype' for '$name' in $path"))
        arrays[Symbol(name)] = reshape(collect(arr), shape)
        offset += nbytes
    end
    offset == length(bytes) - CHECKSUM_NBYTES ||
        throw(FixtureError("fixture file has $(length(bytes) - CHECKSUM_NBYTES - offset) unexplained trailing " *
                            "byte(s) before the checksum footer: $path"))

    manifest_ok = nothing
    if check_manifest
        mpath = manifest_path === nothing ? fixture_manifest_path(path) : manifest_path
        if isfile(mpath)
            mismatch_reason = nothing
            mscalars, marrays = _parse_meta_text(read(mpath, String))
            if get(mscalars, "fixture_sha256", nothing) != fixture_sha256_hex
                mismatch_reason = "manifest fixture_sha256 does not match fixture file: $mpath vs $path"
            elseif length(marrays) != length(array_metas)
                mismatch_reason = "manifest array count ($(length(marrays))) does not match fixture ($(length(array_metas))): $mpath"
            else
                for (am, mam) in zip(array_metas, marrays)
                    if am["sha256"] != mam["sha256"]
                        mismatch_reason = "manifest hash for array '$(am["name"])' does not match fixture file: $mpath vs $path"
                        break
                    end
                end
            end
            manifest_ok = mismatch_reason === nothing
            if !manifest_ok
                manifest_strict && throw(FixtureError(mismatch_reason))
            end
        end
    end

    meta_nt = (
        schema_version   = schema_version,
        generated_at     = scalars["generated_at"]::String,
        cell_id          = parse(Int, scalars["cell_id"]::String),
        name             = scalars["name"]::String,
        dim              = parse(Int, scalars["dim"]::String),
        layout           = scalars["layout"]::String,
        tier_label       = scalars["tier_label"]::String,
        n_train_nominal  = parse(Int, scalars["n_train_nominal"]::String),
        n_train_actual   = parse(Int, scalars["n_train_actual"]::String),
        n_query          = parse(Int, scalars["n_query"]::String),
        noise_level      = parse(Float64, scalars["noise_level"]::String),
        noise_bound      = parse(Float64, scalars["noise_bound"]::String),
        noise_seed       = parse(Int, scalars["noise_seed"]::String),
        model_seed       = parse(Int, scalars["model_seed"]::String),
        output_range     = parse(Float64, scalars["output_range"]::String),
        fixture_sha256   = fixture_sha256_hex,
        manifest_ok      = manifest_ok,
    )
    return merge(meta_nt, (
        x_train             = arrays[:x_train],
        y_train_clean       = arrays[:y_train_clean],
        y_train_noisy       = arrays[:y_train_noisy],
        x_query             = arrays[:x_query],
        y_query_truth       = arrays[:y_query_truth],
        query_ids           = arrays[:query_ids],
    ))
end

function validate_fixture(path::AbstractString; kwargs...)
    try
        read_fixture(path; kwargs...)
        return true, nothing
    catch e
        return false, sprint(showerror, e)
    end
end

end # module SmokeFixture
