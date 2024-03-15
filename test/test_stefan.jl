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

# Domain
mesh_step_size_list = [3., 2., 1., 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
mesh_step_size = mesh_step_size_list[1]
grid = CartesianGrid((40, 40) , (mesh_step_size, mesh_step_size))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage 
x, y = mesh
nx, ny = length(x), length(y)
@show nx*ny

domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))
circle = SignedDistanceFunction((x, y, _=0) -> sqrt((x-1.5)^2+(y-1.5)^2) - 1.0 , domain)
circle_complement = CartesianLevelSet.complement(circle)

# Cut cells Circle Init:
values = evaluate_levelset(circle.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

cut_cells_boundary = create_boundary(cut_cells, nx, ny, 0.0)

# Cut cells Circle Complement Init
values_complement = evaluate_levelset(circle_complement.sdf_function, mesh)
cut_cells_complement = CartesianLevelSet.get_cut_cells(values_complement)
intersection_points_complement = get_intersection_points(values_complement, cut_cells_complement)
midpoints_complement = get_segment_midpoints(values_complement, cut_cells_complement, intersection_points_complement)

cut_cells_boundary_complement = create_boundary(cut_cells_complement, nx, ny, 0.0)

# Border Init : 
border_cells = get_border_cells(mesh)
border_cells_boundary = create_boundary(border_cells, nx, ny, 0.0)

# calculate first and second order moments for the circle
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(circle.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(circle.sdf_function, mesh, bary)

# calculate first and second order moments for the circle complement
V_complement, v_diag_complement, bary_complement, ax_diag_complement, ay_diag_complement = calculate_first_order_moments(circle_complement.sdf_function, mesh)
w_diag_complement, bx_diag_complement, by_diag_complement, border_cells_wx_complement, border_cells_wy_complement = calculate_second_order_moments(circle_complement.sdf_function, mesh, bary_complement)

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

# Operators Circle Complement
G_complement = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag_complement, by_diag_complement)
GT_complement = build_matrix_G_T(nx, ny, Delta_x_plus, Delta_y_plus, bx_diag_complement, by_diag_complement)
minus_GTHT_complement = build_matrix_minus_GTHT(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag_complement, ay_diag_complement)
H_complement = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag_complement, ay_diag_complement, bx_diag_complement, by_diag_complement)
HT_complement = build_matrix_H_T(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag_complement, ay_diag_complement, bx_diag_complement, by_diag_complement)
Wdagger_complement = sparse_inverse(w_diag_complement)


# Dirichlet boundary conditions
# Forcing function
function f_omega(x, y)
    return 0.0 #2*pi^4*sin(2*pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2) 
end

# Analytic solution
function p(x, y)
    return 1-(x-1.5)^2-(y-1.5)^2 #cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))  
end

f_omega_values = [f_omega(x, y) for (x, y) in bary]
g_gamma = cut_cells_boundary + border_cells_boundary
p_omega = [circle.sdf_function(x, y) > 0 ? 0 : p(x, y) for (x, y) in bary]

g_gamma_complement = cut_cells_boundary_complement + border_cells_boundary
p_omega_complement = [circle_complement.sdf_function(x, y) > 0 ? 0 : p(x, y) for (x, y) in bary_complement]


"""
# Robin boundary conditions
IGamma = build_igamma(HT)
Ia = build_diagonal_matrix_sparse(ones(nx*ny))
Ib = build_diagonal_matrix_sparse(ones(nx*ny))

# Forcing function
function f_omega(x, y)
    return 2*pi^4*sin(2*pi^2*(x-1.5)*(y-1.5))*((x-1.5)^2+(y-1.5)^2) 
end

f_omega_values = [f_omega(x, y) for (x, y) in bary]
g_gamma = cut_cells_boundary + border_cells_boundary

x_robin_w, x_robin_g = solve_Ax_b_robin(G, GT, Wdagger, H, HT, Ib, Ia, v_diag, f_omega_values, IGamma, g_gamma)

x_robin_matrix = reshape(x_robin_w, (nx, ny))
heatmap(x_robin_matrix, title="Poisson Equation - Robin BC", aspect_ratio=1)
readline()

scatter(x_robin_g, title="1D plot of x_robin_g", label="x_robin_g")
readline()
"""

function build_block_matrix_system(G1, GT1, Wdagger1, H1, HT1, G2, GT2, Wdagger2, H2, HT2)
    # Construire les blocs de la matrice
    block1 = GT1 * Wdagger1 * G1
    block2 = GT1 * Wdagger1 * H1
    block3 = HT1 * Wdagger1 * G1
    block4 = HT1 * Wdagger1 * H1 + HT2 * Wdagger2 * H2
    block5 = HT2 * Wdagger2 * G2
    block6 = GT2 * Wdagger2 * H2
    block7 = GT2 * Wdagger2 * G2

    # Créer deux blocs de zéros de la même taille que block1 et block2
    zero_block1 = zeros(size(block1))
    zero_block2 = zeros(size(block7))
    
    line1 = hcat(block7, zero_block1, block6)
    line2 = hcat(zero_block2, block1, block2)
    line3 = hcat(block5, block3, block4)

    # Construire la matrice en blocs
    A = vcat(line1, line2, line3)

    return A
end

function build_rhs_vector(V1, f_omega1, V2, f_omega2)
    # Construire les blocs du vecteur de droite
    block1 = V1 * f_omega1
    block2 = zeros(size(block1))
    block3 = V2 * f_omega2

    # Construire le vecteur de droite
    b = vcat(block3, block1, block2)

    return b
end

function solve_block_matrix_system(A, b, border_cells, border_cells_boundary)
    
    for (i, cell) in enumerate(border_cells)
        linear_index = LinearIndices((nx, ny))[cell]
        A[linear_index, :] .= 0
        A[linear_index, linear_index] = 1
        b[linear_index] = border_cells_boundary[linear_index]
    end

    # Résoudre le système linéaire
    x = gmres(A, b)

    return x
end

# Forcing function
function f_omega_1(x, y)
    return 4.0 #2*pi^4*sin(2*pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2) 
end

function f_omega_2(x, y)
    return 8.0 #2*pi^4*sin(2*pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2) 
end

f_omega_values_1 = [f_omega_1(x, y) for (x, y) in bary]
f_omega_values_2 = [f_omega_2(x, y) for (x, y) in bary_complement]

# Blocs de la matrice
A = build_block_matrix_system(G, GT, Wdagger, H, HT, G_complement, GT_complement, Wdagger_complement, H_complement, HT_complement)
rhs = build_rhs_vector(v_diag, f_omega_values_1, v_diag_complement, f_omega_values_2)
sol = solve_block_matrix_system(A, rhs, border_cells, border_cells_boundary)

# Extraire T_1, T_1_g, T_2 du vecteur solution
mid = length(sol) ÷ 3
T_2 = sol[1:mid]
T_1_g = sol[mid+1:2*mid]
T_1 = sol[2*mid+1:end]

# Reshape les solutions en matrices
T_2_matrix = reshape(T_1, (nx, ny))
T_1_g_matrix = reshape(T_1_g, (nx, ny))
T_1_matrix = reshape(T_2, (nx, ny))

# Créer les heatmaps
p1 = heatmap(T_1_matrix, title="Poisson Equation - T_1", aspect_ratio=1)
readline()
p2 = heatmap(T_1_g_matrix, title="Poisson Equation - T_1_g", aspect_ratio=1)
readline() 
p3 = heatmap(T_2_matrix, title="Poisson Equation - T_2", aspect_ratio=1)
readline()
# Afficher les heatmaps sur le même graphique
plot(p1, p2, p3, layout=(3, 1))
readline()
