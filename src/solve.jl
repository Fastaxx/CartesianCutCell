## Dirichlet Condition
# Function to solve G^T W^T G pw = Vfw - G^T W^T H gg - Poisson Equation Dirichlet
function solve_Ax_b_poisson(nx::Int, ny::Int, G, GT, Wdagger, H, V, f_omega, g_gamma, border_cells, boundary_conditions)
    A = GT * Wdagger * G  # Construct the matrix A
    b = V * f_omega - GT * Wdagger * H * g_gamma # Construct the vector b
    
    left_cells = [cell for cell in border_cells if cell[2] == 1]
    right_cells = [cell for cell in border_cells if cell[2] == nx-1]
    top_cells = [cell for cell in border_cells if cell[1] == ny-1]
    bottom_cells = [cell for cell in border_cells if cell[1] == 1]
    top_left_corner = [cell for cell in border_cells if cell[1] == ny-1 && cell[2] == 1]
    top_right_corner = [cell for cell in border_cells if cell[1] == ny-1 && cell[2] == nx-1]
    bottom_left_corner = [cell for cell in border_cells if cell[1] == 1 && cell[2] == 1]
    bottom_right_corner = [cell for cell in border_cells if cell[1] == 1 && cell[2] == nx-1]

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

## Neumann Condition
# Function to construct the block matrix Neumann
function construct_block_matrix_neumann(G, GT, Wdagger, H, HT)
    # Constructing the upper left block
    upper_left = GT * Wdagger * G
    
    # Constructing the upper right block
    upper_right = GT * Wdagger * H
    
    # Constructing the lower left block
    lower_left = HT * Wdagger * G
    
    # Constructing the lower right block
    lower_right = HT * Wdagger * H
    
    # Concatenating the blocks horizontally and vertically
    upper_block = hcat(upper_left, upper_right)
    lower_block = hcat(lower_left, lower_right)
    A = vcat(upper_block, lower_block)
    
    return A
end

# Function to construct the RHS Vector Neumann
function construct_rhs_vector_neumann(V, f_omega, IGamma, g_gamma)
    # Concatenate the vectors vertically
    b = vcat(V*f_omega, IGamma * g_gamma)
    return b
end

using Statistics

# Function to solve the Bloc Matrix System Neumann
function solve_Ax_b_neumann(G, GT, Wdagger, H, HT, V, f_omega, IGamma, g_gamma)
    A = construct_block_matrix_neumann(G, GT, Wdagger, H, HT)
    b = construct_rhs_vector_neumann(V, f_omega, IGamma, g_gamma) 

    # Solving the linear system
    x = bicgstabl(A, b) 

    mid = length(x) ÷ 2
    # Extracting p_ω and p_γ from the solution vector
    p_ω = x[1:mid]
    p_γ = x[(mid + 1):end]

    return p_ω, p_γ
end

## Robin Condition
# Function to construct the block matrix Robin
function construct_block_matrix_robin(G, GT, Wdagger, H, HT, Ib, Ia, IGamma)
    # Constructing the upper left block
    upper_left = GT * Wdagger * G
    
    # Constructing the upper right block
    upper_right = GT * Wdagger * H
    
    # Constructing the lower left block
    lower_left = Ib * HT * Wdagger * G
    
    # Constructing the lower right block
    lower_right = Ib * HT * Wdagger * H + Ia * IGamma
    
    # Concatenating the blocks horizontally and vertically
    upper_block = hcat(upper_left, upper_right)
    lower_block = hcat(lower_left, lower_right)
    A = vcat(upper_block, lower_block)
    
    return A
end

# Function to construct the RHS Vector Robin
function construct_rhs_vector_robin(V, f_omega, IGamma, g_gamma)
    # Concatenate the vectors vertically
    b = vcat(V*f_omega, IGamma * g_gamma)
    return b
end

# Function to solve the Bloc Matrix System Robin
function solve_Ax_b_robin(G, GT, Wdagger, H, HT, Ib, Ia, V, f_omega, IGamma, g_gamma)
    A = construct_block_matrix_robin(G, GT, Wdagger, H, HT, Ib, Ia, IGamma)
    b = construct_rhs_vector_robin(V, f_omega, IGamma, g_gamma)

    # Solving the linear system
    x = cg(A, b)

    mid = length(x) ÷ 2
    # Extracting p_ω and p_γ from the solution vector
    p_ω = x[1:mid]
    p_γ = x[(mid + 1):end]

    return p_ω, p_γ
end

function solve_system(V, G, GT, Wdagger, H, T_w_0, T_g_0, delta_t, t_end)
    # Initialiser les solutions
    T_w = T_w_0
    T_g = T_g_0

    # Construire la matrice A
    block1 = V + delta_t/2 * GT * Wdagger * G
    block2 = delta_t/2 * GT * Wdagger * H
    block3 = zeros(size(block2))
    block4 = ones(size(block2))

    A = [block1 block2; block3 block4]

    # Boucle sur le temps
    t = 0.0
    while t < t_end
        @show t
        # Construire le vecteur b
        b = [(V - delta_t/2 * GT * Wdagger * G) * T_w - delta_t/2 * GT * Wdagger * H * T_g; 
             T_g]

        # Résoudre le système Ax = b
        T = cg(A, b)

        # Mettre à jour les solutions
        T_w = T[1:size(T_w_0)[1]]
        T_g = T[(size(T_w_0)[1] + 1):end]
        # Mettre à jour le temps
        t += delta_t
    end

    return T_w, T_g
end