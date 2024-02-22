using SparseArrays

# Function to build the sparse diagonal matrix Ax
function build_diagonal_matrix_Ax(Ax_values::Vector{Float64})
    n = length(Ax_values)
    return sparse(1:n, 1:n, Ax_values)
end

# Function to build the sparse diagonal matrix Ay
function build_diagonal_matrix_Ay(Ay_values::Vector{Float64})
    n = length(Ay_values)
    return sparse(1:n, 1:n, Ay_values)
end

# Function to build the sparse diagonal matrix V
function build_diagonal_matrix_V(V_values::Vector{Float64})
    n = length(V_values)
    return sparse(1:n, 1:n, V_values)
end

# Example usage
Ax_values = [1.0, 2.0, 3.0]  # Example values for Ax
Ay_values = [0.5, 1.5, 2.5]  # Example values for Ay
V_values = [0.1, 0.2, 0.3]   # Example values for V

Ax = build_diagonal_matrix_Ax(Ax_values)
Ay = build_diagonal_matrix_Ay(Ay_values)
V = build_diagonal_matrix_V(V_values)

# Print the matrices
println("Ax:")
println(Ax)
println("Ay:")
println(Ay)
println("V:")
println(V)


# Function to build the sparse diagonal matrix Bx
function build_diagonal_matrix_Bx(Bx_values::Vector{Float64})
    n = length(Bx_values)
    return sparse(1:n, 1:n, Bx_values)
end

# Function to build the sparse diagonal matrix By
function build_diagonal_matrix_By(By_values::Vector{Float64})
    n = length(By_values)
    return sparse(1:n, 1:n, By_values)
end

# Function to build the block diagonal matrix W
function build_block_diagonal_W(Wx::Vector{T}, Wy::Vector{T}) where T
    n = length(Wx)
    Wx_diag = spdiagm(0 => Wx)
    Wy_diag = spdiagm(0 => Wy)
    return blockdiag(Wx_diag, Wy_diag)
end

# Example vectors
Wx = [1.0, 2.0, 3.0]
Wy = [4.0, 5.0, 6.0]

# Build the block diagonal matrix W
W = build_block_diagonal_W(Wx, Wy)

# Display the matrix
println("W:")
println(W)
