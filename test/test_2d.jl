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

mesh_step_size_list = [2., 1., 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
mesh_step_size = mesh_step_size_list[2]
grid = CartesianGrid((320, 320) , (mesh_step_size, mesh_step_size))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage 
x, y = mesh
nx = length(x)
ny = length(y)
@show nx*ny
domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))
circle1 = SignedDistanceFunction((x, y, _=0) -> sqrt((x-0.5)^2+(y-0.5)^2) - 0.25 , domain)
#circle1 = CartesianLevelSet.complement(circle1)

values = evaluate_levelset(circle1.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

# calculate first and second order moments
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(circle1.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(circle1.sdf_function, mesh, bary)

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

grad_true_x = [circle1.sdf_function(x, y) >= 0 ? 0 : ∂Φ∂x(x, y) for (x, y) in bary]
grad_true_y = [circle1.sdf_function(x, y) > 0 ? 0 : ∂Φ∂y(x, y) for (x, y) in bary]
grad_true_y_without_border = [value for (i, value) in enumerate(grad_true_y) if !(i in border_cells_wy)]
grad_true_x_without_border = [value for (i, value) in enumerate(grad_true_x) if !(i in border_cells_wx)]

p_omega = [Φ(x, y) for (x, y) in bary]
p_gamma = [Φ(x, y) for (x, y) in bary]

Wdagger = sparse_inverse(w_diag)
G = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag, by_diag)
H = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag, ay_diag, bx_diag, by_diag)

grad_approx = compute_grad_operator(p_omega, p_gamma, Wdagger, G, H)
midpoint = length(grad_approx) ÷ 2
grad_approx_x = grad_approx[1:midpoint]
grad_approx_y = grad_approx[midpoint+1:end]

"""
grad_x_matrix = reshape(grad_approx_x, ny, nx)'
heatmap(grad_x_matrix, title = "Gradient x")
readline()

grad_y_matrix = reshape(grad_approx_y, ny, nx)'
heatmap(grad_y_matrix, title = "Gradient y")
readline()
"""

diff = grad_true_x - grad_approx_x
diff_matrix = reshape(diff, ny, nx)'

"""
heatmap(diff_matrix, title = "Heatmap of difference between grad_true_x and grad_approx_x")
readline()
"""

@show volume_integrated_p_norm(grad_true_x, grad_approx_x, V, Inf)/volume_integrated_p_norm(grad_true_x, zeros(length(grad_true_x)), V, Inf)

diff = grad_true_y - grad_approx_y
"""
diff_matrix = reshape(diff, ny, nx)'
heatmap(diff_matrix, title = "Heatmap of difference between grad_true_y and grad_approx_y")
readline()
"""

solid_indices, fluid_indices, cut_cell_indices = get_volume_indices(V, maximum(V))

grad_approx_x_fluid = [grad_approx_x[i] for i in fluid_indices]
grad_true_x_fluid = [grad_true_x[i] for i in fluid_indices]

grad_approx_y_fluid = [grad_approx_y[i] for i in fluid_indices]
grad_true_y_fluid = [grad_true_y[i] for i in fluid_indices]

V_fluid = [V[i] for i in fluid_indices]
@show volume_integrated_p_norm(grad_true_x_fluid, grad_approx_x_fluid, V_fluid, Inf)/volume_integrated_p_norm(grad_true_x_fluid, zeros(length(grad_true_x_fluid)), V_fluid, Inf)

grad_approx_x_cut = [grad_approx_x[i] for i in cut_cell_indices]
grad_true_x_cut = [grad_true_x[i] for i in cut_cell_indices]

grad_approx_y_cut = [grad_approx_y[i] for i in cut_cell_indices]
grad_true_y_cut = [grad_true_y[i] for i in cut_cell_indices]

V_cut = [V[i] for i in cut_cell_indices]
@show volume_integrated_p_norm(grad_true_x_cut, grad_approx_x_cut, V_cut, Inf)/volume_integrated_p_norm(grad_true_x_cut, zeros(length(grad_true_x_cut)), V_cut, Inf)


num_cut_cells, percentage_cut_cells, percentage_uncut_cells, total_volume, mean_volume, min_volume, max_volume, std_dev_volume, median_volume = cut_cell_statistics(V, cut_cell_indices)

cut_cell_volumes = [V[i] for i in cut_cell_indices]

# Créez un histogramme des volumes des cut-cells
p = histogram(cut_cell_volumes, bins=50, xlabel="Volume", ylabel="Nombre de cut-cells", title="Distribution des volumes des cut-cells")

# Ajoutez des lignes verticales pour indiquer la moyenne, la médiane, le minimum et le maximum
vline!(p, [mean_volume], color=:blue, label="Moyenne")
vline!(p, [median_volume], color=:green, label="Médiane")
vline!(p, [min_volume], color=:red, label="Min")
vline!(p, [max_volume], color=:yellow, label="Max")

# Affichez le graphique
display(p)
readline()

@show num_cut_cells
@show percentage_cut_cells
@show percentage_uncut_cells
@show total_volume
@show mean_volume
@show min_volume
@show max_volume
@show std_dev_volume
@show median_volume
