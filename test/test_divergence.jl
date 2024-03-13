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

### Test divergence
mesh_step_size_list = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
mesh_step_size = mesh_step_size_list[4]
universe = (-20:mesh_step_size:20, -20:mesh_step_size:20) #(-20:20, -20:20)
node = (-5:mesh_step_size:5, -5:mesh_step_size:5) # Avec cette taille on a un rayon de 1.0 (-5:5, -5:5)

# define mesh
xyz, xyz_staggered = generate_mesh(universe, node)

nx = length(xyz_staggered[1])
ny = length(xyz_staggered[2])

@show nx*ny
# define level set
const R = 1.0
const a, b = 0.5, 0.5
levelset = HyperSphere(R, (a, b))

# calculate first and second order moments
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(levelset, xyz_staggered)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(levelset, xyz_staggered, bary)

# Operators
Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

# Cut cells Init:
cut_cells = get_cut_cells(levelset, xyz_staggered)

function u_x(x, y)
    return 2.0*(x-a)^2
end

function u_y(x, y)
    return 2.0*(y-b)^2
end

ux = [u_x(x, y) for (x, y) in bary]
uy = [u_y(x, y) for (x, y) in bary]

function u(ux, uy)
    return vcat(ux, uy)
end

# Ground truth
div_true = [levelset(x, y) >= 0 ? 0 : 4.0*x+4.0*y for (x, y) in bary]

# Approximation
q_omega = u(ux, uy)
q_gamma = u(ux, uy)
GT = build_matrix_G_T(nx, ny, Delta_x_plus, Delta_y_plus, bx_diag, by_diag)
minus_GTHT = build_matrix_minus_GTHT(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag, ay_diag)
HT = build_matrix_H_T(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag, ay_diag, bx_diag, by_diag)
Wdagger = sparse_inverse(w_diag)
Vdagger = sparse_inverse(v_diag)

div_approx = compute_divergence(q_omega, q_gamma, Vdagger, GT, minus_GTHT, HT, false)

max_volume = maximum(V)
solid_indices, fluid_indices, cut_cells_indices = get_volume_indices(V, max_volume)

# A Vérifier si on récupére bien le bon volume
# Solid Part
V_solid = [V[index] for index in solid_indices]
solid_div_approx = [div_approx[index] for index in solid_indices]
solid_div_true = [0 for i in 1:length(solid_div_approx)]
diff_solid = solid_div_approx - solid_div_true
l2_error_solid = volume_integrated_p_norm(solid_div_approx, solid_div_true, V_solid, 2.0)
@show l2_error_solid

# Fluid Part
V_fluid = [V[index] for index in fluid_indices]
fluid_div_approx = [div_approx[index] for index in fluid_indices]
fluid_div_true = [4 for i in 1:length(fluid_indices)]
diff_fluid = fluid_div_approx - fluid_div_true
l2_error_fluid = volume_integrated_p_norm(fluid_div_approx, fluid_div_true, V_fluid, 2.0)
@show l2_error_fluid

# Cut cells Part
V_cut_cells = [V[index] for index in cut_cells_indices]
cut_cells_div_approx = [div_approx[index] for index in cut_cells_indices]
cut_cells_div_true = [4 for i in 1:length(cut_cells_indices)]
diff_cut_cells = cut_cells_div_approx - cut_cells_div_true
l2_error_cut_cells = volume_integrated_p_norm(cut_cells_div_approx, cut_cells_div_true, V_cut_cells, 2.0)
@show l2_error_cut_cells

# Plot divergence Approximation
div_approx_matrix = reshape(div_approx, (nx, ny))
heatmap(div_approx_matrix, aspect_ratio = 1, color = :blues, xlabel = "x", ylabel = "y", title = "Divergence Approximation")
readline()

# Plot divergence Ground Truth
div_true_matrix = reshape(div_true, (nx, ny))
heatmap(div_true_matrix, aspect_ratio = 1, color = :blues, xlabel = "x", ylabel = "y", title = "Divergence Truth")
readline()

# Plot divergence Approximation - Ground Truth
div_diff = div_approx - div_true
div_diff_matrix = reshape(div_diff, (nx, ny))
heatmap(div_diff_matrix, aspect_ratio = 1, color = :blues, xlabel = "x", ylabel = "y", title = "Divergence Approximation - Truth")
readline()

l2_norm_error = volume_integrated_p_norm(div_approx, div_true, V, 2.0)
@show l2_norm_error

#cut_cells_div_approx = [div_approx[i] for i in cut_cells]
#other_cells_div_approx = [div_approx[i] for i in 1:length(div_approx) if i ∉ cut_cells]
