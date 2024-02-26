using StaticArrays
using SparseArrays
using CartesianArrays
using CartesianGeometry

import Base: OneTo
const T = Float64
include("matrix.jl")
include("operators.jl")
include("solve.jl")

function generate_mesh_and_geometry(levelset_type, radius, center, outer_bounds, inner_bounds)
    # Domain
    outer = outer_bounds #Define the outer domain (universe)
    inner = inner_bounds #Define the inner domain (node)

    xy_collocated = collocated.(Ref(identity), outer, inner) # Generate a collocated mesh

    # Geometry
    levelset = levelset_type(radius, center .* one.(eltype.(xy_collocated))) #Cela crée une HyperSphere avec un rayon de 0.25 et un centre en (0, 0).

    return xy_collocated, levelset
end

function calculate_first_order_moments(halos, levelset, xy_collocated)

    # Moments (1st order)
    # Volume 
    mom = CVector{SVector{length(halos[1])+1,T}}(undef, halos[1])
    integrate!(mom, Tuple{0}, levelset, xy_collocated, halos[1])

    v = first.(mom) # V Vector Volume 1st Moment
    bary = deleteat.(mom, 1) # Centroids
    v_diag = spdiagm(0 => v)

    # Surface
    domains = ntuple(_ -> halos[1], length(halos[1]))
    a = CVector{T}(undef, (halos[1]..., OneTo(length(halos[1])))) # A vector 1st Moment
    integrate!(a, Tuple{1}, levelset, xy_collocated, domains)
    
    mid = div(length(a), 2)
    ax = a[1:mid]
    ay = a[mid+1:end]

    ax_diag = spdiagm(0 => ax)
    ay_diag = spdiagm(0 => ay)

    return v_diag, ax_diag, ay_diag, bary
end

function calculate_second_order_moments(halos, levelset, xy_collocated, bary)
    # Moments (2nd order)
    domains = ntuple(_ -> halos[2], length(halos[2]))
    w = CVector{T}(undef, (halos[2]..., OneTo(length(halos[2])))) # W vector 2nd Moment
    integrate!(w, Tuple{0}, levelset, xy_collocated, bary, domains)
    w_diag = spdiagm(0 => w)

    # Surface
    b = CVector{T}(undef, (halos[1]..., OneTo(length(halos[1])))) # B Vector 2nd Moment
    integrate!(b, Tuple{1}, levelset, xy_collocated, bary, halos[1])
    b_diag = spdiagm(0 => b)

    mid = div(length(b), 2)
    bx = b[1:mid]
    by = b[mid+1:end]

    bx_diag = spdiagm(0 => bx)
    by_diag = spdiagm(0 => by)

    return w_diag, bx_diag, by_diag
end


outer = (-1:11, -1:19)
inner = (1:9, 1:17)
radius = 0.25
center = (0.0, 0.0)
levelset_type = HyperSphere{2, Float64}
halos = ((-1:10, -1:18),
(0:10, 0:18))

xy_collocated, levelset = generate_mesh_and_geometry(levelset_type, radius, center, outer, inner)

nx = length(xy_collocated[1])
ny = length(xy_collocated[2])
# -1 : =0 (interface)
# 0 : >0 (inside)
# 1 : <0 (outside)

v_diag, ax_diag, ay_diag, bary = calculate_first_order_moments(halos, levelset, xy_collocated) # 
w_diag, bx_diag, by_diag = calculate_second_order_moments(halos, levelset, xy_collocated, bary)

Delta_x_minus = backward_difference_matrix_sparse_2D_x(nx, ny)
Delta_y_minus = backward_difference_matrix_sparse_2D_y(nx, ny)

println(size(Delta_x_minus))
println(size(Delta_y_minus))
println(size(bx_diag))
println(size(by_diag))
#G = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag, by_diag)

#readline()