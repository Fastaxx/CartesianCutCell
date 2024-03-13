using StaticArrays
using SparseArrays
using CartesianArrays
using CartesianGeometry
using LinearAlgebra
using Plots
using IterativeSolvers
using Arpack
Plots.default(show = true)

include("../src/matrix.jl")
include("../src/operators.jl")
include("../src/solve.jl")
include("../src/mesh.jl")
include("../src/boundary.jl")
include("../src/utils.jl")

### Test gradient
mesh_step_size_list = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
mesh_step_size = mesh_step_size_list[3]
universe = (-20:mesh_step_size:20, -20:mesh_step_size:20) #(-20:20, -20:20)
node = (-5:mesh_step_size:5, -5:mesh_step_size:5) # Avec cette taille on a un rayon de 1.0 (-5:5, -5:5)

# define mesh
xyz, xyz_staggered = generate_mesh(universe, node)

nx = length(xyz[1])
ny = length(xyz[2])

# define level set
const R = 1.0
const a, b = 0.5, 0.5
levelset = HyperSphere(R, (a, b))

# calculate first and second order moments
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(levelset, xyz)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(levelset, xyz, bary)

# Operators
Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

function Φ(x, y)
    return 2.0*x
end

function ∂Φ∂x(x, y)
    return 2.0
end

function ∂Φ∂y(x, y)
    return 0.0
end

grad_true_x = [levelset(x, y) > 0 ? 0 : ∂Φ∂x(x, y) for (x, y) in bary]
grad_true_x_without_border = [value for (i, value) in enumerate(grad_true_x) if !(i in border_cells_wx)]

grad_true_y = [∂Φ∂y(x, y) for (x, y) in bary]
grad_true_y_without_border = [value for (i, value) in enumerate(grad_true_y) if !(i in border_cells_wy)]

p_omega = [Φ(x, y) for (x, y) in bary]
p_gamma = [Φ(x, y) for (x, y) in bary]

Wdagger = sparse_inverse(w_diag)
G = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag, by_diag)
H = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag, ay_diag, bx_diag, by_diag)

grad_approx = compute_grad_operator(p_omega, p_gamma, Wdagger, G, H)
midpoint = length(grad_approx) ÷ 2
grad_approx_x = grad_approx[1:midpoint]
grad_approx_y = grad_approx[midpoint+1:end]

grad_approx_x_without_border = [value for (i, value) in enumerate(grad_approx_x) if !(i in border_cells_wx)]
grad_approx_x_border = [value for (i, value) in enumerate(grad_approx_x) if (i in border_cells_wx)]
grad_approx_y_without_border = [value for (i, value) in enumerate(grad_approx_y) if !(i in border_cells_wy)]
grad_approx_y_border = [value for (i, value) in enumerate(grad_approx_y) if (i in border_cells_wy)]

diff = grad_approx_x - grad_true_x

grad_x_matrix = reshape(grad_approx_x, nx, ny)
heatmap(grad_x_matrix, title = "Heatmap of grad_approx_x_with_border")
readline()

grad_x_true_matrix = reshape(grad_true_x, nx, ny)
heatmap(grad_x_true_matrix, title = "Heatmap of grad_true_x")
readline()

diff_matrix = reshape(diff, nx, ny)
heatmap(diff_matrix, title = "Heatmap of diff")
readline()

l2_norm_error_x = volume_integrated_p_norm(grad_approx_x, grad_true_x, V, 2.0)

@show l2_norm_error_x
