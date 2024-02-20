using SparseArrays

function backward_difference_matrix_sparse(n::Int)
    # Initialize arrays to store non-zero entries and their corresponding indices
    row_indices = Int[]
    col_indices = Int[]
    values = Float64[]
    
    # Fill the interior rows of the matrix
    for i in 2:n-1
        push!(row_indices, i)
        push!(col_indices, i)
        push!(values, 1.0)
        
        push!(row_indices, i)
        push!(col_indices, i-1)
        push!(values, -1.0)
    end
    
    # Add the last row entries
    push!(row_indices, n)
    push!(col_indices, n-1)
    push!(values, -1.0)
    push!(row_indices, n)
    push!(col_indices, n)
    push!(values, 1.0)
    
    # Create a sparse matrix using the collected non-zero entries
    D_sparse = sparse(row_indices, col_indices, values, n, n)
    
    # Set the last element in the diagonal to 0
    D_sparse[n, n] = 0.0

    return D_sparse
end

# Test the function
n = 5
D_sparse = backward_difference_matrix_sparse(n)
println("Backward Difference Matrix:")
println(D_sparse)

function forward_difference_matrix_sparse(n::Int)
    # Initialize arrays to store non-zero entries and their corresponding indices
    row_indices = Int[]
    col_indices = Int[]
    values = Float64[]
    
    # Fill the interior rows of the matrix
    for i in 1:n-1
        push!(row_indices, i)
        push!(col_indices, i)
        push!(values, -1.0)
        
        push!(row_indices, i)
        push!(col_indices, i+1)
        push!(values, 1.0)
    end
    
    # Add the last row entries
    push!(row_indices, n)
    push!(col_indices, n)
    push!(values, -1.0)
    push!(row_indices, n)
    push!(col_indices, n-1)
    push!(values, 1.0)
    
    # Create a sparse matrix using the collected non-zero entries
    D_sparse = sparse(row_indices, col_indices, values, n, n)
    
    # Set the last element in the diagonal to 0
    D_sparse[n, n] = 0.0

    return D_sparse
end

# Test the function
n = 5
D_sparse = forward_difference_matrix_sparse(n)
println("Forward Difference Matrix:")
println(D_sparse)


function backward_interpolation_matrix_sparse(n::Int)
    # Initialize arrays to store non-zero entries and their corresponding indices
    row_indices = Int[]
    col_indices = Int[]
    values = Float64[]
    
    # Fill the interior rows of the matrix
    for i in 2:n
        push!(row_indices, i)
        push!(col_indices, i)
        push!(values, 1.0)
        
        push!(row_indices, i)
        push!(col_indices, i-1)
        push!(values, 1.0)
    end
    
    # Add the last row entries
    push!(row_indices, 1)
    push!(col_indices, 1)
    push!(values, 1.0)
    
    # Create a sparse matrix using the collected non-zero entries
    D_sparse = 0.5*sparse(row_indices, col_indices, values, n, n)
    
    # Set the last element in the diagonal to 0
    D_sparse[n, n] = 0.0
    
    return D_sparse
end

# Test the function
n = 5
D_sparse = backward_interpolation_matrix_sparse(n)
println("Backward Interpolation Matrix:")
println(D_sparse)

function forward_interpolation_matrix_sparse(n::Int)
    # Initialize arrays to store non-zero entries and their corresponding indices
    row_indices = Int[]
    col_indices = Int[]
    values = Float64[]
    
    # Fill the interior rows of the matrix
    for i in 1:n-1
        push!(row_indices, i)
        push!(col_indices, i)
        push!(values, 1.0)
        
        push!(row_indices, i)
        push!(col_indices, i+1)
        push!(values, 1.0)
    end
    
    # Add the last row entries
    push!(row_indices, n)
    push!(col_indices, n)
    push!(values, 1.0)
    
    # Create a sparse matrix using the collected non-zero entries
    D_sparse = 0.5*sparse(row_indices, col_indices, values, n, n)
    
    # Set the last element in the diagonal to 0
    D_sparse[n, n] = 0.0
    
    return D_sparse
end

# Test the function
n = 5
D_sparse = forward_interpolation_matrix_sparse(n)
println("Forward Interpolation Matrix:")
println(D_sparse)

function identity_matrix_sparse(n::Int)
    # Create arrays to store non-zero entries and their corresponding indices
    row_indices = Int[]
    col_indices = Int[]
    values = Float64[]
    
    # Fill the arrays to construct the identity matrix
    for i in 1:n
        push!(row_indices, i)
        push!(col_indices, i)
        push!(values, 1.0)
    end
    
    # Create a sparse matrix using the collected non-zero entries
    I_sparse = sparse(row_indices, col_indices, values, n, n)
    
    return I_sparse
end

# Test the function
n = 5
I_sparse = identity_matrix_sparse(n)
println("Identity Matrix:")
println(I_sparse)

function kronecker_product_sparse(A::SparseMatrixCSC, B::SparseMatrixCSC)
    # Compute the Kronecker product using Kronecker.jl
    C = kron(A, B)
    
    return C
end

# Test the function
A = sparse([1 0; 0 1])
B = sparse([2 0; 0 2])
C_sparse = kronecker_product_sparse(I_sparse, D_sparse)
println("Kronecker Product of A and B:")
println(C_sparse)
