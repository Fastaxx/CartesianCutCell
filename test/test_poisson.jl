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

## Test Poisson
grid = CartesianGrid((360, 360) , (4.0, 4.0))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage
x, y = mesh
nx,ny = length(x), length(y)
dx, dy = 1/nx, 1/ny 
@show nx*ny

a,b = 2, 2
domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))
circle = SignedDistanceFunction((x, y, _=0) -> sqrt((x-a)^2+(y-b)^2) - 1 , domain)

values = evaluate_levelset(circle.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

cut_cells_boundary = create_boundary(cut_cells, nx, ny, 0.0)

border_cells = get_border_cells(mesh)

# Définir les conditions de bord
boundary_conditions = Dict(
    "left" => DirichletCondition(0.0),  # Remplacer par la condition de bord gauche
    "right" => DirichletCondition(0.0),  # Remplacer par la condition de bord droite
    "top" => DirichletCondition(0.0),  # Remplacer par la condition de bord supérieure
    "bottom" => DirichletCondition(0.0)  # Remplacer par la condition de bord inférieure
)

# calculate first and second order moments
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(circle.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(circle.sdf_function, mesh, bary)

# Operators Global
Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

# Operators Circle
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

# Forcing function
function f_omega(x, y)
    return 4 #4*pi^4*cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2)
end

# Analytic solution
function p(x, y)
    return 1-(x-a)^2-(y-b)^2  #cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))  
end

f_omega_values = [f_omega(x, y) for (x, y) in bary]
g_gamma = cut_cells_boundary

# Solve
p_omega,A = solve_Ax_b_poisson(nx, ny, G, GT, Wdagger, H, v_diag, f_omega_values, g_gamma, border_cells, boundary_conditions)
p_omega_without_cutcells = [value for (i, value) in enumerate(p_omega) if !(i in cut_cells)]

# Analytic solution
p_true = [(x, y) in cut_cells ? 0 : (circle.sdf_function(x, y) > 0 ? 0 : p(x, y)) for (x, y) in bary]

# Error
diff = abs.(p_omega - p_true)

# Plot
p1 = heatmap(x, y, reshape(p_omega, (nx, ny))', c = :viridis, aspect_ratio = 1, title = "Numerical solution")
p2 = heatmap(x, y, reshape(p_true, (nx, ny))', c = :viridis, aspect_ratio = 1, title = "Analytic solution")
p3 = heatmap(x, y, reshape(diff, (nx, ny))', c = :viridis, aspect_ratio = 1, title = "Error")
plot(p1, p2, p3, layout = (1, 3), size = (1200, 400))
readline()

# Plot Boundary field 
angles_deg = calculate_angles(bary, cut_cells, nx, ny, [a,b])
linear_indices = [LinearIndices((nx, ny))[i] for i in cut_cells]
diff_cut = diff[linear_indices]

scatter(angles_deg, diff_cut, xlabel="Angle (degrees)", ylabel="Error", title="Error vs Angle")
readline()

# Gradient of the solution
grad_true_x = [circle.sdf_function(x, y) >= 0 ? 0 : -2*(x-a) for (x, y) in bary]
grad_true_y = [circle.sdf_function(x, y) >= 0 ? 0 : -2*(y-b) for (x, y) in bary]
#grad_true_x = [circle.sdf_function(x, y) >= 0 ? 0 : pi^2*(y-b)*cos(2*pi^2*(x-a)*(y-b)) for (x, y) in bary]
#grad_true_y = [circle.sdf_function(x, y) >= 0 ? 0 : pi^2*(x-a)*cos(2*pi^2*(x-a)*(y-b)) for (x, y) in bary]

grad_approx = compute_grad_operator(p_omega, p_omega, Wdagger, G, H)

midpoint = length(grad_approx) ÷ 2
grad_approx_x = grad_approx[1:midpoint]
grad_approx_y = grad_approx[midpoint+1:end]

grad_x_matrix = reshape(grad_approx_x, ny, nx)'
grad_y_matrix = reshape(grad_approx_y, ny, nx)'

heatmap(grad_x_matrix, title = "Gradient x", c = :viridis, aspect_ratio = 1, xlabel = "x", ylabel = "y")
readline()

heatmap(reshape(grad_true_x, ny, nx)', title = "Gradient x true", c = :viridis, aspect_ratio = 1, xlabel = "x", ylabel = "y")
readline()

heatmap(grad_y_matrix, title = "Gradient y", c = :viridis, aspect_ratio = 1, xlabel = "x", ylabel = "y")
readline()

heatmap(reshape(grad_true_y, ny, nx)', title = "Gradient y true", c = :viridis, aspect_ratio = 1, xlabel = "x", ylabel = "y")
readline()

# Error
diff_x = abs.(grad_approx_x - grad_true_x)
diff_y = abs.(grad_approx_y - grad_true_y)

heatmap(reshape(diff_x, ny, nx)', title = "Error x", c = :viridis, aspect_ratio = 1, xlabel = "x", ylabel = "y")
readline()

heatmap(reshape(diff_y, ny, nx)', title = "Error y", c = :viridis, aspect_ratio = 1, xlabel = "x", ylabel = "y")
readline()


# Norm Error
solid_indices, fluid_indices, cut_cell_indices = get_volume_indices(V)

grad_approx_x_fluid = [grad_approx_x[i] for i in fluid_indices]
grad_true_x_fluid = [grad_true_x[i] for i in fluid_indices]
grad_approx_y_fluid = [grad_approx_y[i] for i in fluid_indices]
grad_true_y_fluid = [grad_true_y[i] for i in fluid_indices]

V_fluid = [V[index] for index in fluid_indices]
l2_error_x = volume_integrated_p_norm(grad_approx_x_fluid, grad_true_x_fluid, V_fluid, 2.0)
l2_error_y = volume_integrated_p_norm(grad_approx_y_fluid, grad_true_y_fluid, V_fluid, 2.0)

@show l2_error_x
@show l2_error_y

grad_approx_x_cut_cells = [grad_approx_x[i] for i in cut_cell_indices]
grad_true_x_cut_cells = [grad_true_x[i] for i in cut_cell_indices]
grad_approx_y_cut_cells = [grad_approx_y[i] for i in cut_cell_indices]
grad_true_y_cut_cells = [grad_true_y[i] for i in cut_cell_indices]

V_cut_cells = [V[index] for index in cut_cell_indices]
l2_error_x_cut_cells = volume_integrated_p_norm(grad_approx_x_cut_cells, grad_true_x_cut_cells, V_cut_cells, 2.0)
l2_error_y_cut_cells = volume_integrated_p_norm(grad_approx_y_cut_cells, grad_true_y_cut_cells, V_cut_cells, 2.0)

@show l2_error_x_cut_cells
@show l2_error_y_cut_cells

l2_error_all_x = volume_integrated_p_norm(grad_approx_x, grad_true_x, V, 2.0)
l2_error_all_y = volume_integrated_p_norm(grad_approx_y, grad_true_y, V, 2.0)

@show l2_error_all_x
@show l2_error_all_y

