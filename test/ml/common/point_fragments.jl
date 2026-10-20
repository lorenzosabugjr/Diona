module PointFragments

using Arrow

export read_point_fragment, read_point_fragments, write_point_fragment

_materialize(value::AbstractString) = String(value)
_materialize(value) = value

"""Atomically write a Tables.jl-compatible collection to an Arrow fragment."""
function write_point_fragment(path::AbstractString, rows)
    destination = String(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp-$(getpid())"
    try
        Arrow.write(temporary, rows)
        mv(temporary, destination; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

"""Read an Arrow fragment as dictionaries keyed by its column names."""
function read_point_fragment(path::AbstractString)
    table = Arrow.Table(path)
    names = propertynames(table)
    isempty(names) && return Dict{Symbol,Any}[]
    columns = [getproperty(table, name) for name in names]
    nrows = length(first(columns))
    all(length(column) == nrows for column in columns) ||
        error("Arrow point fragment has columns of unequal length: $path")
    return [Dict{Symbol,Any}(name => _materialize(columns[j][i])
                            for (j, name) in enumerate(names))
            for i in 1:nrows]
end

"""Read all `*.points.arrow` fragments in a directory in stable path order."""
function read_point_fragments(directory::AbstractString)
    isdir(directory) || error("point fragment directory does not exist: $directory")
    paths = sort(filter(path -> endswith(path, ".points.arrow"),
                        readdir(directory; join=true)))
    isempty(paths) && error("no .points.arrow fragments found in $directory")
    return reduce(vcat, (read_point_fragment(path) for path in paths);
                  init=Dict{Symbol,Any}[])
end

end
