struct OMXLookup
    file::HDF5.File #should match parent OMXFile
end

@enumx OMXIndexDimension Rows Columns Both

struct OMXIndex{T}
    index::Vector{T}
    dim::OMXIndexDimension.T
end

struct OMXFile
    file::HDF5.File
    lookup::OMXLookup
end

"""
    OMXFile(filename)

Open the OMX file `filename`
"""
function OMXFile(filename)
    h5file = h5open(filename)
    OMXFile(h5file, OMXLookup(h5file))
end

Base.close(f::OMXFile) = close(f.file)

Base.keys(f::OMXFile) = keys(f.file["data"])

Base.size(f::OMXFile) = tuple(read(attributes(f.file)["SHAPE"])...)

function Base.getindex(f::OMXFile, key::String)
    data = read(f.file, "data/$key")

    # row-major to column-major, https://juliaio.github.io/HDF5.jl/stable/#Language-interoperability-with-row-and-column-major-order-arrays
    data = permutedims(data, reverse(1:ndims(data)))

    attrs = attributes(f.file["data/$key"])
    kv = Dict{String, Any}(map(keys(attrs)) do k
            k => read(attributes(f.file["data/$key"])[k])
        end...)

    return OpenMtx(data, kv)
end

Base.keys(l::OMXLookup) = keys(l.file["lookup"])

function Base.getindex(l::OMXLookup, key::AbstractString)
    index = read(l.file, "/lookup/$key")
    attrs = attributes(l.file["/lookup/$key"])
    dimtype = if haskey(attrs, "dim")
        dim = read(attrs["dim"])
        if dim == 0
            OMXIndexDimension.Rows
        elseif dim == 1
            OMXIndexDimension.Columns
        else
            error("Invalid dim value $dim")
        end
    else
        OMXIndexDimension.Both
    end

    return OMXIndex(index, dimtype)
end
