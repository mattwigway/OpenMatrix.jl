struct OpenMtx{T}
    matrix::Matrix{T}
    kv::Dict{String, Any}
end

function OpenMtx(mtx::Matrix{T}, kv) where T
    if haskey(kv, "NA")
        na_val = kv["NA"]

        final_mtx = Matrix{Union{eltype(mtx), Missing}}(undef)
        copy!(final_mtx, mtx)

        final_mtx[final_mtx .== na_val] .= missing
    else
        final_mtx = mtx
    end

    return OpenMtx{T}(mtx, kv)
end

Base.eltype(::OpenMtx{T} where T) = T

# Think about how to do getindex here. We want to support querying by:
#  index
#  lookup (hmm... need to reference original file)
#  arbitrary attribute name

# unindexed lookup
Base.getindex(m::OpenMtx, idx...) = Base.getindex(m.matrix, idx...)