using StaticArrays
using SparseArrays
using CartesianArrays
using CartesianGeometry
using Plots

import Base: OneTo
const T = Float64

struct HyperCuboid{N,T}
    sides::NTuple{N,T}
    center::NTuple{N,T}
end

function (object::HyperCuboid{1})(x, _...)
    (; sides, center) = object
    min_distance = minimum(abs.(x .- center[1]) .- sides ./ 2)
    return min_distance
end

function (object::HyperCuboid{2})(x, y, _...)
    (; sides, center) = object
    min_distance_x = minimum(abs.(x .- center[1]) .- sides[1] / 2)
    min_distance_y = minimum(abs.(y .- center[2]) .- sides[2] / 2)
    return min(min_distance_x, min_distance_y)
end

function (object::HyperCuboid{3})(x, y, z)
    (; sides, center) = object
    min_distance_x = minimum(abs.(x .- center[1]) .- sides[1] / 2)
    min_distance_y = minimum(abs.(y .- center[2]) .- sides[2] / 2)
    min_distance_z = minimum(abs.(z .- center[3]) .- sides[3] / 2)
    return min(min(min_distance_x, min_distance_y), min_distance_z)
end


# Domain
outer = (1:9, 1:9) #Define the outer domain (universe)
inner = (1:9, 1:9) #Define the inner domain (node)

halos = ((1:8, 1:8),
         (2:8, 2:8)) # Define the two staggered grid

xy_collocated = collocated.(Ref(identity), outer, inner) # Generate a collocated mesh

# Geometry
levelset = HyperSphere(0.25, 0.5 .* one.(eltype.(xy_collocated)))
levelset2 = HyperCuboid((2, 3), (0, 0))

# Moments (1st order)
# Volume 
mom = CVector{SVector{length(halos[1])+1,T}}(undef, halos[1])
integrate!(mom, Tuple{0}, levelset, xy_collocated, halos[1])

v = first.(mom) # V Vector Volume 1st Moment
bary = deleteat.(mom, 1) # Centroids
print(v)

# Surface
domains = ntuple(_ -> halos[1], length(halos[1]))
a = CVector{T}(undef, (halos[1]..., OneTo(length(halos[1])))) # A vector 1st Moment
integrate!(a, Tuple{1}, levelset, xy_collocated, domains)
print(a)


# Moments (2nd order)
domains = ntuple(_ -> halos[2], length(halos[2]))
w = CVector{T}(undef, (halos[2]..., OneTo(length(halos[2])))) # W vector 2nd Moment
integrate!(w, Tuple{0}, levelset, xy_collocated, bary, domains)

# Surface
b = CVector{T}(undef, (halos[1]..., OneTo(length(halos[1])))) # B Vector 2nd Moment
integrate!(b, Tuple{1}, levelset, xy_collocated, bary, halos[1])


