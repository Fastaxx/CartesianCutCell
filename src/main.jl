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
include("plot.jl")

mesh_step_size_list = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
mesh_step_size = mesh_step_size_list[2]
universe = (-20:mesh_step_size:20, -20:mesh_step_size:20) #(-20:20, -20:20)
node = (-5:mesh_step_size:5, -5:mesh_step_size:5) # Avec cette taille on a un rayon de 1.0 (-5:5, -5:5)

# define mesh
xyz, xyz_staggered = generate_mesh(universe, node)
nx = length(xyz[1])
ny = length(xyz[2])

@show nx*ny

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
#levelset = HyperCuboid{2, Float64}((0.5, 0.5), (0.5, 0.5))
#levelset = RectangleTrou(R, (a, b))

# Cut cells Init:
cut_cells = get_cut_cells(levelset, xyz)
cut_cells_boundary = create_boundary(cut_cells, length(xyz[1]), length(xyz[2]), 0.0)

# Border Init
border_cells = get_border_cells(xyz)

# Définir les conditions de bord
boundary_conditions = Dict(
    "left" => DirichletCondition(0.0),  # Remplacer par la condition de bord gauche
    "right" => DirichletCondition(0.0),  # Remplacer par la condition de bord droite
    "top" => DirichletCondition(0.0),  # Remplacer par la condition de bord supérieure
    "bottom" => DirichletCondition(0.0)  # Remplacer par la condition de bord inférieure
)

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

mid = length(bary) ÷ 2
barx = bary[mid] 

# Dirichlet boundary conditions
# Forcing function
function f_omega(x, y)
    return 4.0 #2*pi^4*sin(2*pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2) 
end

# Analytic solution
function p(x, y)
    return 1-(x-a)^2-(y-b)^2 #cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))  
end

f_omega_values = [f_omega(x, y) for (x, y) in bary]
g_gamma = cut_cells_boundary

p_omega = [levelset(x, y) > 0 ? 0 : p(x, y) for (x, y) in bary]
p_omega_without_cutcells = [value for (i, value) in enumerate(p_omega) if !(i in cut_cells)]

x_dirichlet = solve_Ax_b_poisson(nx, ny, G, GT, Wdagger, H, v_diag, f_omega_values, g_gamma, border_cells,boundary_conditions)
x_dirichlet_without_cutcells = [value for (i, value) in enumerate(x_dirichlet) if !(i in cut_cells)]

solid_indices, fluid_indices, cut_cell_indices = get_volume_indices(V, maximum(V))

sol_approx_fluid = [x_dirichlet[i] for i in fluid_indices]
sol_approx_solid = [x_dirichlet[i] for i in solid_indices]
sol_approx_cut_cell = [x_dirichlet[i] for i in cut_cell_indices]

sol_true_fluid = [p_omega[i] for i in fluid_indices]
sol_true_solid = [p_omega[i] for i in solid_indices]
sol_true_cut_cell = [p_omega[i] for i in cut_cell_indices]

x_dirichlet_matrix = reshape(x_dirichlet, (nx, ny))
heatmap(x_dirichlet_matrix, title="Poisson Equation - Dirichlet BC", aspect_ratio=1)
readline()

x_analytic_matrix = reshape(p_omega, (nx, ny))
heatmap(x_analytic_matrix, title="Analytic Solution", aspect_ratio=1)
readline()


diff = x_dirichlet - p_omega

"""
diff_matrix = reshape(diff, (nx, ny))
heatmap(diff_matrix, title="Difference between Dirichlet and Analytic Solution", aspect_ratio=1)
readline()
"""

l2_norm_error = volume_integrated_p_norm(x_dirichlet, p_omega, V, 2)
@show l2_norm_error
@show norm(x_dirichlet-p_omega, 2)

V_fluid = [V[i] for i in fluid_indices]
l2_norm_error_fluid = volume_integrated_p_norm(sol_approx_fluid, sol_true_fluid, V_fluid, 2)
@show l2_norm_error_fluid
@show norm(sol_approx_fluid-sol_true_fluid, 2)

V_cut_cell = [V[i] for i in cut_cell_indices]
l2_norm_error_cut_cell = volume_integrated_p_norm(sol_approx_cut_cell, sol_true_cut_cell, V_cut_cell, 2)
@show l2_norm_error_cut_cell
@show norm(sol_approx_cut_cell-sol_true_cut_cell, 2)



"""
# Neumann boundary conditions
# Il faut imposer une condition en plus

x_neumann_w, x_neumann_g = solve_Ax_b_neumann(G, GT, Wdagger, H, HT, v_diag, f_omega_values, IGamma, g_gamma)
@show size(x_neumann_g)

x_neumann_matrix = reshape(x_neumann_w, (nx, ny))
heatmap(x_neumann_matrix, title="Poisson Equation - Neumann BC", aspect_ratio=1)
readline()

# Robin boundary conditions
x_robin_w, x_robin_g = solve_Ax_b_robin(G, GT, Wdagger, H, HT, Ib, Ia, v_diag, f_omega_values, IGamma, g_gamma)
@show x_robin_w

x_robin_matrix = reshape(x_robin_w, (nx, ny))
heatmap(x_robin_matrix, title="Poisson Equation - Robin BC", aspect_ratio=1)
readline()
"""
"""
# Calcul du gradient numérique
grad = compute_grad_operator(x_dirichlet, zeros(nx*ny), Wdagger, G, H)

grad_x_true = [levelset(x, y) > 0 ? 0 : -2.0*(x-a) for (x, y) in bary]
grad_y_true = [levelset(x, y) > 0 ? 0 : -2.0*(y-b) for (x, y) in bary]

mid = length(grad) ÷ 2
grad_x = grad[1:mid]
grad_y = grad[mid+1:end]


grad_x_matrix = reshape(grad_x, (nx, ny))'
heatmap(grad_x_matrix, title="Gradient X Operator", aspect_ratio=1)
readline()

grad_x_true_matrix = reshape(grad_x_true, (nx, ny))'
heatmap(grad_x_true_matrix, title="Gradient X True", aspect_ratio=1)
readline()

grad_y_matrix = reshape(grad_y, (nx, ny))'
heatmap(grad_y_matrix, title="Gradient Y Operator", aspect_ratio=1)
readline()

grad_y_true_matrix = reshape(grad_y_true, (nx, ny))'
heatmap(grad_y_true_matrix, title="Gradient Y True", aspect_ratio=1)
readline()

l2_norm_error_grad_x = volume_integrated_p_norm(grad_x, grad_x_true, V, 2)
@show l2_norm_error_grad_x

l2_norm_error_grad_y = volume_integrated_p_norm(grad_y, grad_y_true, V, 2)
@show l2_norm_error_grad_y

grad_x_fluid = [grad_x[i] for i in fluid_indices]
grad_x_true_fluid = [grad_x_true[i] for i in fluid_indices]
l2_norm_error_grad_x_fluid = volume_integrated_p_norm(grad_x_fluid, grad_x_true_fluid, V_fluid, 2)
@show l2_norm_error_grad_x_fluid

grad_y_fluid = [grad_y[i] for i in fluid_indices]
grad_y_true_fluid = [grad_y_true[i] for i in fluid_indices]
l2_norm_error_grad_y_fluid = volume_integrated_p_norm(grad_y_fluid, grad_y_true_fluid, V_fluid, 2)
@show l2_norm_error_grad_y_fluid

grad_x_cut_cell = [grad_x[i] for i in cut_cell_indices]
grad_x_true_cut_cell = [grad_x_true[i] for i in cut_cell_indices]
l2_norm_error_grad_x_cut_cell = volume_integrated_p_norm(grad_x_cut_cell, grad_x_true_cut_cell, V_cut_cell, 2)
@show l2_norm_error_grad_x_cut_cell

grad_y_cut_cell = [grad_y[i] for i in cut_cell_indices]
grad_y_true_cut_cell = [grad_y_true[i] for i in cut_cell_indices]
l2_norm_error_grad_y_cut_cell = volume_integrated_p_norm(grad_y_cut_cell, grad_y_true_cut_cell, V_cut_cell, 2)
@show l2_norm_error_grad_y_cut_cell

"""