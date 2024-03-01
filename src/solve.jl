include("utils.jl")
## Dirichlet Condition
# Function to solve G^T W^T G pw = Vfw - G^T W^T H gg - Poisson Equation Dirichlet
function solve_Ax_b_poisson(nx::Int, ny::Int, G, GT, Wdagger, H, V, f_omega, g_gamma)
    A = GT * Wdagger * G  # Construct the matrix A
    b = V * f_omega - GT * Wdagger * H * g_gamma # Construct the vector b
    x = cg(A, b) # Solve Ax = b
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

# Function to solve the Bloc Matrix System Neumann
function solve_Ax_b_neumann(G, GT, Wdagger, H, HT, V, f_omega, IGamma, g_gamma)
    A = construct_block_matrix_neumann(G, GT, Wdagger, H, HT)
    b = construct_rhs_vector_neumann(V, f_omega, IGamma, g_gamma)

    # Solving the linear system
    x = cg(A, b)

    # Extracting p_ω and p_γ from the solution vector
    p_ω = x[1:size(G, 2)]
    p_γ = x[(size(G, 2) + 1):end]

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

    # Extracting p_ω and p_γ from the solution vector
    p_ω = x[1:size(G, 2)]
    p_γ = x[(size(G, 2) + 1):end]

    return p_ω, p_γ
end



