module CartesianCutCell

import SparseArrays
import LinearAlgebra

greet() = print("Hello World!")

include("operators.jl")
include("matrix.jl")
include("solve.jl")
include("geometry.jl")
include("boundary.jl")
end # module CartesianCutCell
