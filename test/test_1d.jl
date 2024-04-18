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

# Grille 1D
grid = CartesianGrid((160,) , (1.0,))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage
@show mesh
x = mesh[1]
nx = length(x)
dx = 1/nx
@show nx, dx

x_front = 0.01
domain = ((minimum(x),), (maximum(x),))
front = SignedDistanceFunction((x, _=0) -> x - x_front , domain)
front_complement = CartesianLevelSet.complement(front)
barycentres = vec([(xi + dx/2,) for xi in x[1:end]])

# Phase 1
values = evaluate_levelset(front.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(front.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(front.sdf_function, mesh, bary)

# Définir les conditions de bord
boundary_conditions = (
    left = DirichletCondition(1.0),  # Remplacer par la condition de bord gauche
    right = NeumannCondition(0.0),  # Remplacer par la condition de bord droite
)

# Operators Global
Delta_x_minus = backward_difference_matrix_sparse(nx)
Delta_x_plus = forward_difference_matrix_sparse(nx)

# Operators Phase 1
G = Delta_x_minus*bx_diag
GT = G'
minus_GTHT = Delta_x_plus*ax_diag
H = ax_diag*Delta_x_minus - Delta_x_minus*bx_diag
HT = H'
Wdagger = sparse_inverse(w_diag)

# Initial condition
function Tw0(x)
    return 0.0
end
function Tg0(x)
    return 0.0
end

T_w_0 = [Tw0(x) for (x) in bary]
T_g_0 = [Tg0(x) for (x) in bary]

# Time parameters
delta_t = 1e-2
t_end = 10.0

# Solve
function solve_unsteady(T_w_0, T_g_0, delta_t, t_end, G, GT, H, Wdagger, v_diag, boundary_conditions)
    T_w = T_w_0
    T_g = T_g_0
    t = 0.0
    x_front = 0.0
    x_front_values = [] # Tableau pour stocker les valeurs de x_front

    while t < t_end
        block1 = v_diag + delta_t/2*GT*Wdagger*G
        block2 = delta_t/2*GT*Wdagger*H
        block3 = zeros(size(block2))
        block4 = ones(size(block3))

        A = [block1 block2; block3 block4]
        b = [(v_diag - delta_t/2 * GT * Wdagger * G) * T_w - delta_t/2 * GT * Wdagger * H * T_g; T_g]

        # Boundary
        A[1,:] .= 0.0
        A[1,1] = 1.0
        b[1] = 1.0

        # Résoudre le système Ax = b
        T = cg(A, b)

        # Mettre à jour les solutions
        T_w = T[1:size(T_w_0)[1]]
        T_g = T[(size(T_w_0)[1] + 1):end]

        # Avance le front : v_front = saut gradient de T_w
        v_front = -Wdagger*(G*T_w - H*T_g)
        x_front += delta_t*v_front[1]
        push!(x_front_values, x_front) # Ajouter la valeur actuelle de x_front au tableau

        @show x_front
        # Remettre à jour le front
        front = SignedDistanceFunction((x, _=0) -> x - x_front , domain)

        # Remettre à jour les G, GT , ...
        values = evaluate_levelset(front.sdf_function, mesh)
        cut_cells = CartesianLevelSet.get_cut_cells(values)
        intersection_points = get_intersection_points(values, cut_cells)
        midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

        V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(front.sdf_function, mesh)
        w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(front.sdf_function, mesh, bary)

        G = Delta_x_minus*bx_diag
        GT = G'
        minus_GTHT = Delta_x_plus*ax_diag
        H = ax_diag*Delta_x_minus - Delta_x_minus*bx_diag
        HT = H'
        Wdagger = sparse_inverse(w_diag)

        # Mettre à jour le temps
        t += delta_t
    end
    return T_w, T_g, x_front_values
end

T_w, T_g, x_front_values = solve_unsteady(T_w_0, T_g_0, delta_t, t_end, G, GT, H, Wdagger, v_diag, boundary_conditions)

# Tracer les valeurs de x_front
scatter(x_front_values, title = "Evolution de x_front", xlabel = "Iteration", ylabel = "x_front")
readline()

# Tracer les solutions
plot(x, T_w, title = "T_w", xlabel = "x", ylabel = "T_w")
readline()