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

function build_block_matrix_system(G1, GT1, Wdagger1, H1, HT1, G2, GT2, Wdagger2, H2, HT2, k1, k2)
    # Construire les blocs de la matrice
    block1 = k1 .* GT1 * Wdagger1 * G1
    block2 = k1 .* GT1 * Wdagger1 * H1
    block3 = k1 .* HT1 * Wdagger1 * G1
    block4 = k1 .* HT1 * Wdagger1 * H1 
    block5 = k2 .* HT2 * Wdagger2 * H2
    block6 = k2 .* HT2 * Wdagger2 * G2
    block7 = k2 .* GT2 * Wdagger2 * H2
    block8 = k2 .* GT2 * Wdagger2 * G2

    # Créer deux blocs de zéros de la même taille que block1 et block2
    zero_block = zeros(size(block1))

    # Créer une matrice de uns de la même taille que block1
    one_block = sparse(Matrix{Float64}(I, size(block1)))
    minus_one_block = -2.0 .* one_block

    line1 = hcat(block8, block7, zero_block, zero_block)
    line2 = hcat(zero_block, one_block, zero_block, minus_one_block)
    line3 = hcat(zero_block, zero_block, block1, block2)
    line4 = hcat(block6, block5, block3, block4) 

    # Construire la matrice en blocs
    A = vcat(line1, line2, line3, line4)
end

function build_rhs_vector(V1, f_omega1, V2, f_omega2)
    # Construire les blocs du vecteur de droite
    block1 = V1 * f_omega1
    block2 = zeros(size(block1))
    block3 = V2 * f_omega2

    zero_block = zeros(size(block1))

    # Construire le vecteur de droite
    b = vcat(block3, zero_block, block1, zero_block) # T_2_w, T_2_g, T_1_w, T_1_g

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

    x = gmres(A, b) # Solve Ax = b
    return x
end

# Forcing function
function f_omega_1(x, y)
    return 4.0
end

function f_omega_2(x, y)
    return 4.0
end

f_omega_values_1 = [f_omega_1(x, y) for (x, y) in bary]
f_omega_values_2 = [f_omega_2(x, y) for (x, y) in bary_complement]

# Blocs de la matrice
A = build_block_matrix_system(G, GT, Wdagger, H, HT, G_complement, GT_complement, Wdagger_complement, H_complement, HT_complement, k1, k2)
rhs = build_rhs_vector(v_diag, f_omega_values_1, v_diag_complement, f_omega_values_2)
sol = solve_block_matrix_system(A, rhs, border_cells, boundary_conditions)

# Plot
T_2_w = sol[1:nx*ny]
T_2_g = sol[nx*ny+1:2*nx*ny]
T_1_w = sol[2*nx*ny+1:3*nx*ny]
T_1_g = sol[3*nx*ny+1:4*nx*ny]

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

plot(p1, p2, p3, p4, layout = (1, 4), size = (1200, 400), title = ["T_2_w" "T_1_w" "T_1_g" "T_2_g"], aspect_ratio = 1)
readline()

# Plot 3D
for i in 1:nx
    for j in 1:ny
        if T_2_matrix[i, j] == 0
            T_2_matrix[i, j] = T_1_matrix[i, j]
        end
    end
end

p1 = surface(x, y, T_2_matrix, title="T_2_w", xlabel="x", ylabel="y", zlabel="T_2_w", camera=(50, 50))
readline()