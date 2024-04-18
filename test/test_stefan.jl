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
grid = CartesianGrid((80, 80) , (2.0, 2.0))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage 
x, y = mesh
nx, ny = length(x), length(y)
dx, dy = 1/nx, 1/ny
@show nx*ny

a,b = 1.0, 1.0
domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))
circle = SignedDistanceFunction((x, y, _=0) -> sqrt((x-a)^2+(y-b)^2) - 0.5 , domain)
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

# Définir les conditions de bord
boundary_conditions = (
    left = NeumannCondition(0.0),  # Remplacer par la condition de bord gauche
    right = NeumannCondition(0.0),  # Remplacer par la condition de bord droite
    top = NeumannCondition(0.0),  # Remplacer par la condition de bord supérieure
    bottom = NeumannCondition(0.0)  # Remplacer par la condition de bord inférieure
)

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
function build_block_matrix_system2(G1, GT1, Wdagger1, H1, HT1, G2, GT2, Wdagger2, H2, HT2)
    # Construire les blocs de la matrice
    block1 = GT1 * Wdagger1 * G1
    block2 = GT1 * Wdagger1 * H1
    block3 = HT1 * Wdagger1 * G1
    block4 = HT1 * Wdagger1 * H1 
    block5 = HT2 * Wdagger2 * H2
    block6 = HT2 * Wdagger2 * G2
    block7 = GT2 * Wdagger2 * H2
    block8 = GT2 * Wdagger2 * G2

    # Créer deux blocs de zéros de la même taille que block1 et block2
    zero_block = zeros(size(block1))

    # Créer une matrice de uns de la même taille que block1
    one_block = sparse(Matrix{Float64}(I, size(block1)))
    minus_one_block = -one_block

    line1 = hcat(block1, zero_block, block2, zero_block)
    line2 = hcat(zero_block, block8, zero_block, block7)
    line3 = hcat(zero_block, zero_block, one_block, minus_one_block)
    line4 = hcat(block3, block6, block4, block5) 

    # Construire la matrice en blocs
    A = vcat(line1, line2, line3, line4)

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
function build_rhs_vector2(V1, f_omega1, V2, f_omega2)
    # Construire les blocs du vecteur de droite
    block1 = V1 * f_omega1
    block2 = V2 * f_omega2 
    block3 = zeros(size(block1))
    block4 = zeros(size(block1))

    # Construire le vecteur de droite
    b = vcat(block1, block2, block3, block4)

    return b
end

function solve_block_matrix_system(A, b, border_cells, boundary_conditions)
    left_cells = [cell for cell in border_cells if cell[2] == 1]
    right_cells = [cell for cell in border_cells if cell[2] == nx-1]
    top_cells = [cell for cell in border_cells if cell[1] == ny-1]
    bottom_cells = [cell for cell in border_cells if cell[1] == 1]

    # Modify A and b for each border cell
    for (i, cell) in enumerate(border_cells)
        linear_index = LinearIndices((nx, ny))[cell]

        # Apply boundary conditions
        if cell in left_cells
            condition = boundary_conditions.left
        elseif cell in right_cells
            condition = boundary_conditions.right
        elseif cell in top_cells
            condition = boundary_conditions.top
        elseif cell in bottom_cells
            condition = boundary_conditions.bottom
        end

        if condition isa DirichletCondition
            A[linear_index, :] .= 0
            A[linear_index, linear_index] = 1
            b[linear_index] = isa(condition.value, Function) ? condition.value(cell...) : condition.value
        elseif condition isa NeumannCondition
            if cell in left_cells
                A[linear_index, linear_index] = -1
                A[linear_index, linear_index + 1] = 1
            elseif cell in right_cells
                A[linear_index, linear_index] = -1
                A[linear_index, linear_index - 1] = 1
            elseif cell in top_cells
                A[linear_index, linear_index] = -1
                A[linear_index, linear_index - nx] = 1  # if row-major
                # A[linear_index, linear_index - 1] = 1  # if column-major
            elseif cell in bottom_cells
                A[linear_index, linear_index] = -1
                A[linear_index, linear_index + nx] = 1  # if row-major
                # A[linear_index, linear_index + 1] = 1  # if column-major
            end
            b[linear_index] -= isa(condition.value, Function) ? condition.value(cell...) : condition.value
        elseif condition isa PeriodicCondition
            # Implement Periodic condition
        elseif condition isa RobinCondition
            # Implement Robin condition
        end
    end

    x = bicgstabl(A, b) # Solve Ax = b
    return x
end


# Interface condition #Inutile si T1g=T2g dans le cas equilibre
g_gamma = cut_cells_boundary
g_gamma_complement = cut_cells_boundary_complement

# Forcing function
function f_omega_1(x, y)
    return 4*pi^4*cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2) 
end

function f_omega_2(x, y)
    return 4*pi^4*cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2) 
end

f_omega_values_1 = [f_omega_1(x, y) for (x, y) in bary]
f_omega_values_2 = [f_omega_2(x, y) for (x, y) in bary_complement]

# Blocs de la matrice
A = build_block_matrix_system2(G, GT, Wdagger, H, HT, G_complement, GT_complement, Wdagger_complement, H_complement, HT_complement)
rhs = build_rhs_vector2(v_diag, f_omega_values_1, v_diag_complement, f_omega_values_2)
sol = solve_block_matrix_system(A, rhs, border_cells, boundary_conditions)

# Extraire T_2_w, T_2_g, T_1_w, T_1_g du vecteur solution
T_2_w = sol[1:nx*ny]
T_1_w = sol[nx*ny+1:2*nx*ny]
T_1_g = sol[2*nx*ny+1:3*nx*ny]
T_2_g = sol[3*nx*ny+1:4*nx*ny]

# Reshape les solutions en matrices
T_2_matrix = reshape(T_2_w, (nx, ny))
T_1_matrix = reshape(T_1_w, (nx, ny))
T_1_g_matrix = reshape(T_1_g, (nx, ny))
T_2_g_matrix = reshape(T_2_g, (nx, ny))

# Créer les heatmaps
p1 = heatmap(T_2_matrix, title="T_2_w", aspect_ratio=1)
readline()
p2 = heatmap(T_1_matrix, title="T_1_w", aspect_ratio=1)
readline()
p3 = heatmap(T_1_g_matrix, title="T_1_g", aspect_ratio=1)
readline()
p4 = heatmap(T_2_g_matrix, title="T_2_g", aspect_ratio=1)
readline()

plot(p1, p2, p3, p4, layout = (1, 4), size = (1200, 400), title = ["T_2_w" "T_1_w" "T_1_g" "T_1_g"], aspect_ratio = 1)
readline()

# Comparaison Carrée
carre = SignedDistanceFunction((x, y, _=0) -> (x-a)^2 + (y-b)^2 - 25.0, domain)
values_carre = evaluate_levelset(carre.sdf_function, mesh)
cut_cells_carre = CartesianLevelSet.get_cut_cells(values_carre)
intersection_points_carre = get_intersection_points(values_carre, cut_cells_carre)
midpoints_carre = get_segment_midpoints(values_carre, cut_cells_carre, intersection_points_carre)

cut_cells_boundary_carre = create_boundary(cut_cells_carre, nx, ny, 0.0)

# calculate first and second order moments for the carre
V_carre, v_diag_carre, bary_carre, ax_diag_carre, ay_diag_carre = calculate_first_order_moments(carre.sdf_function, mesh)
w_diag_carre, bx_diag_carre, by_diag_carre, border_cells_wx_carre, border_cells_wy_carre = calculate_second_order_moments(carre.sdf_function, mesh, bary_carre)

# Operators Carre
G_carre = build_matrix_G(nx, ny, Delta_x_minus, Delta_y_minus, bx_diag_carre, by_diag_carre)
GT_carre = build_matrix_G_T(nx, ny, Delta_x_plus, Delta_y_plus, bx_diag_carre, by_diag_carre)
minus_GTHT_carre = build_matrix_minus_GTHT(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag_carre, ay_diag_carre)
H_carre = build_matrix_H(nx, ny, Delta_x_minus, Delta_y_minus, ax_diag_carre, ay_diag_carre, bx_diag_carre, by_diag_carre)
HT_carre = build_matrix_H_T(nx, ny, Delta_x_plus, Delta_y_plus, ax_diag_carre, ay_diag_carre, bx_diag_carre, by_diag_carre)
Wdagger_carre = sparse_inverse(w_diag_carre)

# Dirichlet boundary conditions
# Forcing function
function f_omega_carre(x, y)
    return 4*pi^4*cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))*((x-a)^2+(y-b)^2)  
end

# Analytic solution
function p_carre(x, y)
    return cos(pi^2*(x-a)*(y-b))*sin(pi^2*(x-a)*(y-b))  
end

f_omega_values_carre = [f_omega_carre(x, y) for (x, y) in bary_carre]
g_gamma_carre = cut_cells_boundary_carre

p_omega_carre = [carre.sdf_function(x, y) > 0 ? 0 : p_carre(x, y) for (x, y) in bary_carre]

# Blocs de la matrice
x_carre = solve_Ax_b_poisson(nx, ny, G_carre, GT_carre, Wdagger_carre, H_carre, v_diag_carre, f_omega_values_carre, g_gamma_carre, border_cells, boundary_conditions)

x_carre_matrix = reshape(x_carre, (nx, ny))
heatmap(x_carre_matrix, title = "Heatmap of Carree solution", aspect_ratio = 1)
readline()

# Comparaison Phases
diff = x_carre - (T_2_w + T_1_w)+ T_1_g
diff_matrix = reshape(diff, (nx, ny))
heatmap(diff_matrix, title = "Difference between Carre and Stefan solution", aspect_ratio = 1)
readline()

p_omega = [p_carre(x, y) for (x, y) in bary_carre]

error = volume_integrated_p_norm(p_omega, T_2_w + T_1_w - T_1_g, V_carre, Inf)
@show error

solid_indices, fluid_indices, cut_cells_indices = get_volume_indices(V)
V_cut_cells = [V[index] for index in cut_cells_indices]
p_omega_cut_cells = [p_omega[index] for index in cut_cells_indices]
T_2_w_cut_cells = [T_2_w[index] for index in cut_cells_indices]
T_1_w_cut_cells = [T_1_w[index] for index in cut_cells_indices]
T_1_g_cut_cells = [T_1_g[index] for index in cut_cells_indices]
error = volume_integrated_p_norm(p_omega_cut_cells, T_2_w_cut_cells + T_1_w_cut_cells - T_1_g_cut_cells, V_cut_cells, Inf)
@show error


angles_deg = calculate_angles(bary, cut_cells, nx, ny, [a,b])

linear_indices = [LinearIndices((nx, ny))[i] for i in cut_cells]
diff_cut = abs.(diff[linear_indices])

scatter(angles_deg, diff_cut, xlabel="Angle (degrees)", ylabel="Diff", title="Diff vs Angle")
readline()
