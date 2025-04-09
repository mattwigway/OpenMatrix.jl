module OpenMatrix

import HDF5
import HDF5: h5open, attributes
import EnumX: @enumx

# Write your package code here.
include("OpenMtx.jl")
include("OMXFile.jl")
include("IndexedMatrix.jl")

export OMXFile, lookup

end
