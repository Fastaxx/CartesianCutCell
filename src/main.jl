using StaticArrays
using SparseArrays
using CartesianArrays
using CartesianGeometry
using LinearAlgebra
using Plots
using IterativeSolvers

Plots.default(show = true)

include("matrix.jl")
include("operators.jl")
include("solve.jl")
include("mesh.jl")
include("boundary.jl")

universe = (-1:1:11, -1:1:19)
node = (1:9, 1:17)

# define mesh
xyz, xyz_staggered = generate_mesh(universe, node)

nx = length(xyz[1])
ny = length(xyz[2])

# define level set
const R = 0.25
const a, b = 0.5, 0.5

levelset = HyperSphere(R, (a, b))

# calculate first and second order moments
v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(levelset, xyz)
w_diag, bx_diag, by_diag = calculate_second_order_moments(levelset, xyz, bary)

# Operators
Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

# Construct matrices
G = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag, by_diag)
GT = build_matrix_G_T(nx, ny, Delta_x_plus, Delta_y_plus, bx_diag, by_diag)
H = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag, ay_diag, bx_diag, by_diag)
HT = build_matrix_H_T(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag, ay_diag, bx_diag, by_diag)
Wdagger = sparse_inverse(w_diag)
IGamma = build_igamma(HT)
Ia = build_diagonal_matrix_sparse(ones(nx*ny))
Ib = build_diagonal_matrix_sparse(ones(nx*ny))

# Dirichlet boundary conditions
# Forcing function
function f_omega(x, y)
    return 4 # 2*π^4 * sin(2*π^2 *x*y) * (x^2 + y^2)
end

# Analytic solution
function p(x, y)
    return 1-x^2-y^2 # cos(π^2*x*y)*sin(π^2*x*y)
end

f_omega_values = [f_omega(x, y) for (x, y) in bary]
p_omega = [levelset(x, y) < 0 ? p(x, y) : 0 for (x, y) in bary]
g_gamma = zeros(nx*ny)

x_dirichlet = solve_Ax_b_poisson(nx, ny, G, GT, Wdagger, H, v_diag, f_omega_values, g_gamma)

diff = x_dirichlet - p_omega
@show size(x_dirichlet)
@show size(p_omega)
"""
# Neumann boundary conditions
# Il faut imposer une condition en plus

x_neumann = solve_Ax_b_neumann(G, GT, Wdagger, H, HT, v_diag, f_omega_values, IGamma, g_gamma)


# Robin boundary conditions
x_robin = solve_Ax_b_robin(G, GT, Wdagger, H, HT, Ib, Ia, v_diag, f_omega_values, IGamma, g_gamma)

@show x_robin
"""
readline()
