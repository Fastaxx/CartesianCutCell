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
grid = CartesianGrid((80, 80) , (mesh_step_size, mesh_step_size))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage 
x, y = mesh
nx = length(x)
ny = length(y)
@show nx*ny
domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))
carre = SignedDistanceFunction((x, y, _=0) -> max(abs(x-0.5), abs(y-0.5)) - 0.25 , domain)
circle = SignedDistanceFunction((x, y, _=0) -> sqrt((x-0.5)^2+(y-0.5)^2) - 0.25 , domain)
#circle1 = CartesianLevelSet.complement(circle1)

values = evaluate_levelset(circle.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

# Cut cells Init:
cut_cells = get_cut_cells(circle.sdf_function, mesh)
cut_cells_boundary = create_boundary(cut_cells, length(x), length(y), 1.0)

# Border Init
border_cells = get_border_cells(mesh)
border_cells_boundary = create_boundary(border_cells, length(x), length(y), 1.0)

# calculate first and second order moments
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(circle.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(circle.sdf_function, mesh, bary)

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
    return 1-(x-0.5)^2-(y-0.5)^2 #cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))  
end


f_omega_values = [f_omega(x, y) for (x, y) in bary]
g_gamma = cut_cells_boundary + border_cells_boundary

p_omega = [circle.sdf_function(x, y) > 0 ? 0 : p(x, y) for (x, y) in bary]
p_omega_without_cutcells = [value for (i, value) in enumerate(p_omega) if !(i in cut_cells)]

x_dirichlet = solve_Ax_b_poisson(nx, ny, G, GT, Wdagger, H, v_diag, f_omega_values, g_gamma, border_cells,border_cells_boundary)

x_dirichlet_matrix = reshape(x_dirichlet, (nx, ny))
heatmap(x_dirichlet_matrix, title = "Heatmap of Dirichlet solution")
readline()