using StaticArrays
using SparseArrays
using CartesianArrays
using CartesianGeometry

import Base: OneTo
const T = Float64
include("matrix.jl")
include("operators.jl")
include("solve.jl")

function generate_mesh(outer, inner)
    xy_collocated = collocated.(Ref(identity), outer, inner) # Generate a collocated mesh
    xy_staggered = staggered.(Ref(identity), outer, inner) # Generate a staggered mesh

    return xy_collocated, xy_staggered
end

function calculate_first_order_moments(levelset, xyz)
    # first-kind moments
    V, bary = integrate(Tuple{0}, levelset, xyz, T, nan)
    As = integrate(Tuple{1}, levelset, xyz, T, nan)

    indices_non_nuls = findall(!iszero, V)

    #V[V .== 0] .= 1 # Replace by 1 to Avoid division by zero (for Moore Penrose Pseudo Inverse)
    #V = spdiagm(0 => vol) # Diagonal matrix of Volumes : V

    Ax = As[1] # Surface in x : Ax
    Ay = As[2] # Surface in y : Ay

    #ax[ax .== 0] .= 1 # Replace by 1 to Avoid division by zero (for Moore Penrose Pseudo Inverse)
    #ay[ay .== 0] .= 1 # Replace by 1 to Avoid division by zero (for Moore Penrose Pseudo Inverse)

    ax_diag = spdiagm(0 => Ax) # Diagonal matrix of Ax
    ay_diag = spdiagm(0 => Ay) # Diagonal matrix of Ay

    return V, bary, ax_diag, ay_diag
end

function calculate_second_order_moments(levelset, xyz, bary)
    # Moments (2nd order)
    Ws = integrate(Tuple{0}, levelset, xyz, T, nan, bary)
    Wx = Ws[1] # Surface in x : Wx
    Wy = Ws[2] # Surface in y : Wy

    wx_diag = spdiagm(0 => Wx) # Diagonal matrix of Wx
    wy_diag = spdiagm(0 => Wy) # Diagonal matrix of Wy

    w_diag = blockdiag(wx_diag, wy_diag)

    # Surface
    Bs = integrate(Tuple{1}, levelset, xyz, T, nan, bary)
    Bx = Bs[1] # Surface in x : Bx
    By = Bs[2] # Surface in y : By

    bx_diag = spdiagm(0 => Bx) # Diagonal matrix of Bx
    by_diag = spdiagm(0 => By) # Diagonal matrix of By

    return w_diag, bx_diag, by_diag
end

universe = (-1:11, -1:19)
node = (1:9, 1:17)

# define mesh
xyz, xyz_staggered = generate_mesh(universe, node)

nx = length(xyz[1])
ny = length(xyz[2])

# define level set
const R = 0.25
const a, b = 0.5, 0.5

levelset = HyperSphere(R, (a, b))

v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(levelset, xyz)
w_diag, bx_diag, by_diag = calculate_second_order_moments(levelset, xyz, bary)

Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

println("Size Delta_x_minus : ",size(Delta_x_minus))
println("Size Delta_y_minus : ",size(Delta_y_minus))
println("Size Delta_x_plus : ",size(Delta_x_plus))
println("Size Delta_y_plus : ",size(Delta_y_plus))
println("Size ax_diag : ",size(ax_diag))
println("Size ay_diag : ",size(ay_diag))
println("Size v_diag : ",size(v_diag))
println("Size bx_diag : ",size(bx_diag))
println("Size by_diag : ",size(by_diag))
println("Size w_diag : ",size(w_diag))

@show w_diag

G = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag, by_diag)
GT = build_matrix_G_T(nx, ny, Delta_x_plus, Delta_y_plus, bx_diag, by_diag)
H = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag, ay_diag, bx_diag, by_diag)
"""
# -1 : =0 (interface)
# 0 : >0 (inside)
# 1 : <0 (outside)
"""