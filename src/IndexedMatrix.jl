# R and C in type signature so that the isnothing in getindex gets elided at compile time
struct LookupMatrix{T, R <: Union{Nothing, OMXIndex}, C <: Union{Nothing, OMXIndex}}
    matrix::OpenMtx{T}
    rowlookup::R
    collookup::C
end

struct LookupUnitRange{T <: Integer} <: AbstractUnitRange{T}
    lookup::OMXIndex{T}
end

Base.first(l::LookupUnitRange) = first(l.lookup.index)
Base.last(l::LookupUnitRange) = last(l.lookup.index)
Base.iterate(l::LookupUnitRange) = (first(l), 2)
Base.iterate(l::LookupUnitRange, loc) = loc ≤ length(l.lookup.index) ? (l.lookup.index[loc], loc + 1) : nothing

lookup(m::OpenMtx, lookup) = lookup(m, lookup, lookup)

"""
    index(m, rowlookup, collookup)

Return an indexed version of matrix m, with different lookups for rows and columns.

If you want to use a lookup only for rows and not for columns, or vice versa, pass `nothing` as
the respective lookup.
"""
function lookup(m::OpenMtx{T}, rowlookup::R, collookup::C) where {T, R, C}
    !isnothing(rowlookup) && rowlookup.dim ∈ [OMXIndexDimension.Rows, OMXIndexDimension.Both] || error("Row lookup not applicable to rows.")
    !isnothing(collookup) && collookup.dim ∈ [OMXIndexDimension.Columns, OMXIndexDimension.Both] || error("Row lookup not applicable to rows.")

    return LookupMatrix{T, R, C}(m, rowlookup, collookup)
end

function size(lm::LookupMatrix)
    base_size = size(lm.m.matrix)
    xsize = isnothing(lm.rowlookup) ? base_size[1] : length(lm.rowlookup.index)
    ysize = isnothing(lm.collookup) ? base_size[1] : length(lm.collookup.index)

    return xsize, ysize
end

# no lookup: pass through raw index
parseidx(idx, ::Nothing) = idx
parseidx(idx::Colon, ::OMXIndex) = idx
function parseidx(idx, l::OMXIndex)
    # TODO index these somehow? This could be very slow.
    pos = findfirst(l.index .== idx)
    isnothing(pos) && throw(KeyError(idx))
    pos
end

function Base.getindex(lm::LookupMatrix, row, col)
    # TODO why is this allocating?
    return lm.matrix.matrix[parseidx(row, lm.rowlookup), parseidx(col, lm.collookup)]
end

function Base.axes(lm::LookupMatrix)
    base_axes = axes(lm.matrix.matrix)
    rows = isnothing(lm.rowlookup) ? base_axes[1] : LookupUnitRange(lm.rowlookup)
    cols = isnothing(lm.collookup) ? base_axes[2] : LookupUnitRange(lm.collookup)

    return rows, cols
end