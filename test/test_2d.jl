using CartesianLevelSet
using CartesianGeometry
using Plots
Plots.default(show = true)
using StaticArrays
using SparseArrays
using CartesianArrays
using LinearAlgebra
using IterativeSolvers
using Arpack
include("../src/matrix.jl")
include("../src/operators.jl")
include("../src/solve.jl")
include("../src/boundary.jl")
include("../src/utils.jl")
include("../src/mesh.jl")
include("../src/plot.jl")

mesh_step_size_list = [2., 1., 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
mesh_step_size = mesh_step_size_list[2]
grid = CartesianGrid((80, 80) , (mesh_step_size, mesh_step_size))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage 
x, y = mesh
nx = length(x)
ny = length(y)
dx, dy = 1/nx, 1/ny
@show nx*ny
domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))
circle1 = SignedDistanceFunction((x, y, _=0) -> sqrt((x-0.5)^2+(y-0.5)^2) - 0.25 , domain)
circle2 = SignedDistanceFunction((x, y, _=0) -> sqrt((x-0.5)^2+(y-0.5)^2) - 10.0 , domain)
barycentres = vec([(xi + dx/2, yi + dy/2) for xi in x[1:end], yi in y[1:end]])

# circle 1
values = evaluate_levelset(circle1.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

# circle 2
values_2 = evaluate_levelset(circle2.sdf_function, mesh)
cut_cells_2 = CartesianLevelSet.get_cut_cells(values_2)
intersection_points_2 = get_intersection_points(values_2, cut_cells_2)
midpoints_2 = get_segment_midpoints(values_2, cut_cells_2, intersection_points_2)

# calculate first and second order moments circle1
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(circle1.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(circle1.sdf_function, mesh, bary)

# calculate first and second order moments circle2
V_2, v_diag_2, bary_2, ax_diag_2, ay_diag_2 = calculate_first_order_moments(circle2.sdf_function, mesh)
w_diag_2, bx_diag_2, by_diag_2, border_cells_wx_2, border_cells_wy_2 = calculate_second_order_moments(circle2.sdf_function, mesh, bary_2)

# Operators
Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

centre_cercle = [0.5, 0.5]
angles_deg = calculate_angles(bary, cut_cells, nx, ny, centre_cercle)

function Φ(x, y)
    return 2.0*x
end

function ∂Φ∂x(x, y)
    return 2.0
end

function ∂Φ∂y(x, y)
    return 0
end

# Gradient approx
# Circle 1
p_omega = [Φ(x, y) for (x, y) in bary]
p_gamma = [Φ(x, y) for (x, y) in bary] # a verfier

# Gradient true
grad_true_x = [circle1.sdf_function(x, y) >= 0 ? 0 : ∂Φ∂x(x, y) for (x, y) in bary]
grad_true_y = [circle1.sdf_function(x, y) >= 0 ? 0 : ∂Φ∂y(x, y) for (x, y) in bary]
grad_true_y_without_border = [value for (i, value) in enumerate(grad_true_y) if !(i in border_cells_wy)]
grad_true_x_without_border = [value for (i, value) in enumerate(grad_true_x) if !(i in border_cells_wx)]

Wdagger = sparse_inverse(w_diag)
G = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag, by_diag)
H = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag, ay_diag, bx_diag, by_diag)

grad_approx = compute_grad_operator(p_omega, p_gamma, Wdagger, G, H)
midpoint = length(grad_approx) ÷ 2
grad_approx_x = grad_approx[1:midpoint]
grad_approx_y = grad_approx[midpoint+1:end]

grad_x_matrix = reshape(grad_approx_x, ny, nx)'
heatmap(grad_x_matrix, title = "Gradient x")
contour!(values', levels=[0], color=:blue)
readline()

grad_y_matrix = reshape(grad_approx_y, ny, nx)'
heatmap(grad_y_matrix, title = "Gradient y")
readline()

# Circle 2 
p_omega_2 = [Φ(x, y) for (x, y) in bary_2]
p_gamma_2 = [Φ(x, y) for (x, y) in bary_2] # a verfier
# Gradient true
grad_true_x_2 = [circle2.sdf_function(x, y) >= 0 ? 0 : ∂Φ∂x(x, y) for (x, y) in bary_2]
grad_true_y_2 = [circle2.sdf_function(x, y) >= 0 ? 0 : ∂Φ∂y(x, y) for (x, y) in bary_2]

Wdagger_2 = sparse_inverse(w_diag_2)
G_2 = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag_2, by_diag_2)
H_2 = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag_2, ay_diag_2, bx_diag_2, by_diag_2)

grad_approx_2 = compute_grad_operator(p_omega_2, p_gamma_2, Wdagger_2, G_2, H_2)
midpoint_2 = length(grad_approx_2) ÷ 2
grad_approx_x_2 = grad_approx_2[1:midpoint_2]
grad_approx_y_2 = grad_approx_2[midpoint_2+1:end]

grad_x_matrix_2 = reshape(grad_approx_x_2, ny, nx)'
heatmap(grad_x_matrix_2, title = "Gradient x")
readline()

grad_y_matrix_2 = reshape(grad_approx_y_2, ny, nx)'
heatmap(grad_y_matrix_2, title = "Gradient y")
readline()

# Diff
diff = abs.(grad_true_x - grad_approx_x)
diff_matrix = reshape(diff, ny, nx)'
diff_matrix = [value == 2 ? 0 : value for value in diff_matrix]
heatmap(diff_matrix, title = "Heatmap of difference between grad_true_x and grad_approx_x")
readline()

diff = abs.(grad_true_y - grad_approx_y)
diff_matrix = reshape(diff, ny, nx)'
diff_matrix = [value == 2 ? 0 : value for value in diff_matrix]
heatmap(diff_matrix, title = "Heatmap of difference between grad_true_y and grad_approx_y")
readline()

# Diff 2
diff_2 = abs.(grad_true_x_2 - grad_approx_x_2)
diff_2_matrix = reshape(diff_2, ny, nx)'

heatmap(diff_2_matrix, title = "Heatmap of difference between grad_true_x_2 and grad_approx_x_2")
readline()

diff_2 = abs.(grad_true_y_2 - grad_approx_y_2)
diff_2_matrix = reshape(diff_2, ny, nx)'

heatmap(diff_2_matrix, title = "Heatmap of difference between grad_true_y_2 and grad_approx_y_2")
readline()

# Diff 1/2
grad_approx_x_2 = [circle1.sdf_function(x, y) > 0 ? 0 : grad_approx_x_2[i] for (i, (x, y)) in enumerate(bary)]
grad_approx_y_2 = [circle1.sdf_function(x, y) > 0 ? 0 : grad_approx_y_2[i] for (i, (x, y)) in enumerate(bary)]

diff_1_2 = abs.(grad_approx_x_2-grad_approx_x)
diff_1_2_matrix = reshape(diff_1_2, ny, nx)'
heatmap(diff_1_2_matrix, title = "Heatmap of difference between grad_approx_x_2 and grad_approx_x")
readline()

diff_1_2 = abs.(grad_approx_y_2-grad_approx_y)
diff_1_2_matrix = reshape(diff_1_2, ny, nx)'
heatmap(diff_1_2_matrix, title = "Heatmap of difference between grad_approx_y_2 and grad_approx_y")
readline()

# Volume Error
solid_indices, fluid_indices, cut_cell_indices = get_volume_indices(V)

grad_approx_x_fluid = [grad_approx_x[i] for i in fluid_indices]
grad_true_x_fluid = [grad_true_x[i] for i in fluid_indices]

grad_approx_y_fluid = [grad_approx_y[i] for i in fluid_indices]
grad_true_y_fluid = [grad_true_y[i] for i in fluid_indices]

V_fluid = [V[i] for i in fluid_indices]
@show volume_integrated_p_norm(grad_true_x_fluid, grad_approx_x_fluid, V_fluid, 2)#/volume_integrated_p_norm(grad_true_x_fluid, zeros(length(grad_true_x_fluid)), V_fluid, 2)

grad_approx_x_cut = [grad_approx_x[i] for i in cut_cell_indices]
grad_true_x_cut = [grad_true_x[i] for i in cut_cell_indices]

grad_approx_y_cut = [grad_approx_y[i] for i in cut_cell_indices]
grad_true_y_cut = [grad_true_y[i] for i in cut_cell_indices]

V_cut = [V[i] for i in cut_cell_indices]
@show volume_integrated_p_norm(grad_true_x_cut, grad_approx_x_cut, V_cut, 2)#/volume_integrated_p_norm(grad_true_x_cut, zeros(length(grad_true_x_cut)), V_cut, 2)

@show volume_integrated_p_norm(grad_true_x, grad_approx_x, V, 2)#/volume_integrated_p_norm(grad_true_x, zeros(length(grad_true_x)), V, 2)