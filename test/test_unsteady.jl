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

# Cut cells Circle Init:
values = evaluate_levelset(circle.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

cut_cells_boundary = create_boundary(cut_cells, nx, ny, 0.0)

# Border Init : 
border_cells = get_border_cells(mesh)
boundary_conditions = Dict(
    "left" => DirichletCondition(0.0),  # Remplacer par la condition de bord gauche
    "right" => DirichletCondition(0.0),  # Remplacer par la condition de bord droite
    "top" => DirichletCondition(0.0),  # Remplacer par la condition de bord supérieure
    "bottom" => DirichletCondition(0.0)  # Remplacer par la condition de bord inférieure
)

# calculate first and second order moments for the circle
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

#Dirichlet Condition
# Initial Condition
using SpecialFunctions
using FunctionZeros

function Bessel(x, y)
    x= x-1.5
    y= y-1.5
    # Convertir les coordonnées cartésiennes en coordonnées polaires
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)

    # Ordres et racines des fonctions de Bessel
    m1, n1 = 8, 2
    m2, n2 = 4, 4
    ω1 = besselj_zero(m1, n1)
    ω2 = besselj_zero(m2, n2)

    # Facteurs de normalisation
    A1 = 1.0  # A remplacer par la valeur correcte
    A2 = 1.0  # A remplacer par la valeur correcte

    # Calculer la somme des fonctions propres
    return A1 * besselj(m1, ω1 * r) * cos(m1 * θ) + A2 * besselj(m2, ω2 * r) * cos(m2 * θ)
end

function Tw0(x, y)
    x = x-1.5
    y = y-1.5
    return 1/(4*pi*0.01)*exp(-(x^2+y^2)/(4*0.01))
end
function Tg0(x, y)
    return 0.0
end

# Analytical Solution
function Bessel_true(t, x, y)
    x= x-1.5
    y= y-1.5
    # Racines des fonctions de Bessel
    lam1 = besselj_zero(8, 2)
    lam2 = besselj_zero(4, 4)

    # Alpha
    alpha = 1 / (lam1^2 + lam2^2)

    # Harmoniques
    u1 = besselj(8, lam1 * sqrt(x^2 + y^2)) * cos(8 * atan(y, x))
    u2 = besselj(4, lam2 * sqrt(x^2 + y^2)) * cos(4 * atan(y, x))
    return exp(-lam1^2 * alpha * t) * u1 + exp(-lam2^2 * alpha * t) * u2
end

function u_true(t,x,y)
    return 1/(4*pi*(0.01+t))*exp(-(x-1.5)^2-(y-1.5)^2/(4*(0.01+t)))
end

T_w_0 = [Tw0(x, y) for (x, y) in bary]
T_g_0 = [Tg0(x, y) for (x, y) in bary]

heatmap(x, y, reshape(T_w_0, nx, ny), c=:bluesreds, aspect_ratio=:equal, title = "T_w_0", xlabel = "x", ylabel = "y")
readline()

# Time parameters
delta_t = 0.01
t_end = 1.0

T_w, T_g = solve_system(v_diag, G, GT, Wdagger, H, T_w_0, T_g_0, delta_t, t_end)

# Plot
heatmap(x, y, reshape(T_w, nx, ny), c=:bluesreds, aspect_ratio=:equal, title = "T_w", xlabel = "x", ylabel = "y")
readline()
heatmap(x, y, reshape(T_g, nx, ny), c=:bluesreds, aspect_ratio=:equal, title = "T_g", xlabel = "x", ylabel = "y")
readline()

# Comparaison
T_w_true = [u_true(t_end, x, y) for (x, y) in bary]
T_g_true = [0.0 for (x, y) in bary]

diff_T_w = T_w - T_w_true
diff_T_g = T_g - T_g_true

heatmap(x, y, reshape(diff_T_w, nx, ny), c=:bluesreds, aspect_ratio=:equal, title = "Diff T_w", xlabel = "x", ylabel = "y")
readline()

