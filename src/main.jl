using StaticArrays
using SparseArrays
using CartesianArrays
using CartesianGeometry
using LinearAlgebra
using Plots
using IterativeSolvers
using Arpack
Plots.default(show = true)

include("matrix.jl")
include("operators.jl")
include("solve.jl")
include("mesh.jl")
include("boundary.jl")
include("utils.jl")

mesh_step_size_list = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
mesh_step_size = mesh_step_size_list[3]
universe = (-31:mesh_step_size:31, -31:mesh_step_size:31)
node = (-10:mesh_step_size:10, -10:mesh_step_size:10) # Avec cette taille on a un rayon de 1.0

# define mesh
xyz, xyz_staggered = generate_mesh(universe, node)

nx = length(xyz[1])
ny = length(xyz[2])

# define level set
const R = 1.0
const a, b = 0.5, 0.5

struct HyperSphere{N,T}
    radius::T
    center::NTuple{N,T}
end

function (object::HyperSphere{1})(x, _...)
    (; radius, center) = object
    (x - center[1]) ^ 2 - radius ^ 2
end
# Change level set sqrt a verifier
function (object::HyperSphere{2})(x, y, _...)
    (; radius, center) = object
    (x - center[1]) ^ 2 + (y - center[2]) ^ 2 - radius^2
end

function (object::HyperSphere{3})(x, y, z)
    (; radius, center) = object
    (x - center[1]) ^ 2 + (y - center[2]) ^ 2 + (z - center[3]) ^ 2 - radius ^ 2
end

struct HyperCuboid{N,T}
    lengths::NTuple{N,T}
    center::NTuple{N,T}
end

function (object::HyperCuboid{1})(x, _...)
    (; lengths, center) = object
    abs(x - center[1]) - lengths[1] / 2
end

function (object::HyperCuboid{2})(x, y, _...)
    (; lengths, center) = object
    max(abs(x - center[1]) - lengths[1] / 2, abs(y - center[2]) - lengths[2] / 2)
end

function (object::HyperCuboid{3})(x, y, z)
    (; lengths, center) = object
    max(abs(x - center[1]) - lengths[1] / 2, abs(y - center[2]) - lengths[2] / 2, abs(z - center[3]) - lengths[3] / 2)
end

struct RectangleTrou{N,T}
    radius::T
    center::NTuple{N,T}
end

function (object::RectangleTrou{1})(x, _...)
    (; radius, center) = object
    -((x - center[1]) ^ 2 - radius ^ 2)
end

function (object::RectangleTrou{2})(x, y, _...)
    (; radius, center) = object
    -((x - center[1]) ^ 2 + (y - center[2]) ^ 2 - radius ^ 2)
end

function (object::RectangleTrou{3})(x, y, z)
    (; radius, center) = object
    -((x - center[1]) ^ 2 + (y - center[2]) ^ 2 + (z - center[3]) ^ 2 - radius ^ 2)
end
levelset = HyperSphere(R, (a, b))
#levelset = HyperCuboid{2, Float64}((1.0, 1.0), (0.5, 0.5))
levelset = RectangleTrou(R, (a, b))

# calculate first and second order moments
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(levelset, xyz)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(levelset, xyz, bary)

# Operators
Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

# Construct matrices
G = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag, by_diag)
GT = build_matrix_G_T(nx, ny, Delta_x_plus, Delta_y_plus, bx_diag, by_diag)
minus_GTHT = build_matrix_minus_GTHT(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag, ay_diag)
H = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag, ay_diag, bx_diag, by_diag)
HT = build_matrix_H_T(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag, ay_diag, bx_diag, by_diag)
Wdagger = sparse_inverse(w_diag)
IGamma = build_igamma(HT)
Ia = build_diagonal_matrix_sparse(ones(nx*ny))
Ib = build_diagonal_matrix_sparse(ones(nx*ny))

# Dirichlet boundary conditions
# Forcing function
function f_omega(x, y)
    return 4.0 #-2.0*x*(y-1.0)*(y-2.0*x+x*y+2.)*exp(x-y)
end

# Analytic solution
function p(x, y)
    return (1-x^2-y^2) #exp(x-y)*x*(1.0-x)*y*(1.0-y)
end

f_omega_values = [f_omega(x, y) for (x, y) in bary]
p_omega = [p(x, y) for (x, y) in bary]

g_gamma = zeros(nx*ny)
p_omega = [value for (i, value) in enumerate(p_omega) if !(i in border_cells_wx)]

# Calculate the coordinates relative to the center of the sphere
bary_centered = [(x - a, y - b) for (x, y) in bary]

# Calculate the values of the analytic solution
p_omega = [p(x, y) for (x, y) in bary_centered]

x_dirichlet = solve_Ax_b_poisson(nx, ny, G, GT, Wdagger, H, v_diag, f_omega_values, g_gamma)
#x_dirichlet = [value for (i, value) in enumerate(x_dirichlet) if !(i in border_cells_wx)]

x_dirichlet_matrix = reshape(x_dirichlet, (nx, ny))
heatmap(x_dirichlet_matrix, title="Poisson Equation - Dirichlet BC", aspect_ratio=1)
readline()


"""
# Neumann boundary conditions
# Il faut imposer une condition en plus

x_neumann = solve_Ax_b_neumann(G, GT, Wdagger, H, HT, v_diag, f_omega_values, IGamma, g_gamma)


# Robin boundary conditions
x_robin = solve_Ax_b_robin(G, GT, Wdagger, H, HT, Ib, Ia, v_diag, f_omega_values, IGamma, g_gamma)

@show x_robin
"""
