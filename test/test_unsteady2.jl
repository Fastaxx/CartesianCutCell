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

grid = CartesianGrid((80, 80) , (4.0, 4.0))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage 
x, y = mesh
nx,ny = length(x), length(y)

# Border Init : 
border_cells = get_border_cells(mesh)

# Constants
k1, k2 = 1.0, 1.0

# Geometry
a, b = 2.0, 2.0
domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))
sdf = SignedDistanceFunction((x, y, _=0) -> sqrt((x-a)^2+(y-b)^2) - 1 , domain)
sdf_complement = CartesianLevelSet.complement(sdf)

# Définir les conditions de bord
boundary_conditions = (
    left = DirichletCondition(0.0),  # Remplacer par la condition de bord gauche
    right = DirichletCondition(0.0),  # Remplacer par la condition de bord droite
    top = DirichletCondition(0.0),  # Remplacer par la condition de bord supérieure
    bottom = DirichletCondition(0.0)  # Remplacer par la condition de bord inférieure
)

# calculate first and second order moments for the circle
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(sdf.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(sdf.sdf_function, mesh, bary)

# calculate first and second order moments for the circle complement
V_complement, v_diag_complement, bary_complement, ax_diag_complement, ay_diag_complement = calculate_first_order_moments(sdf_complement.sdf_function, mesh)
w_diag_complement, bx_diag_complement, by_diag_complement, border_cells_wx_complement, border_cells_wy_complement = calculate_second_order_moments(sdf_complement.sdf_function, mesh, bary_complement)

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

# Initial condition
function Tw0(x, y)
    return 0.0
end

function Tg0(x, y)
    return 0.0
end

T_w_0 = [Tw0(x, y) for (x, y) in bary]
T_w_0_c = [Tw0(x, y) for (x, y) in bary_complement]
T_g_0 = [Tg0(x, y) for (x, y) in bary]
T_g_0_c = [Tg0(x, y) for (x, y) in bary_complement]

# Time parameters
delta_t = 0.01
t_end = 1.0

function solve_unsteady_diph(T_w_0, T_g_0, T_w_0_c, T_g_0_c, delta_t, t_end, G, GT, H, Wdagger, G_complement, GT_complement, H_complement, Wdagger_complement, v_diag, v_diag_complement, boundary_conditions)
    # Initialiser les solutions
    T_w = T_w_0
    T_g = T_g_0
    T_w_c = T_w_0_c
    T_g_c = T_g_0_c

    # Construire la matrice A
    block1 = v_diag + delta_t/2*GT*Wdagger*G
    block2 = delta_t/2*GT*Wdagger*H
    block3 = v_diag_complement + delta_t/2*GT_complement*Wdagger_complement*G_complement
    block4 = delta_t/2*GT_complement*Wdagger_complement*H_complement

    zero_block = zeros(size(block1))
    one_block = sparse(Matrix{Float64}(I, size(block1)))
    minus_one_block = -1.0 .* one_block

    b1 = [block3 block4 zero_block zero_block]
    b2 = [zero_block one_block zero_block minus_one_block]
    b3 = [zero_block zero_block block1 block2]
    b4 = []
end 