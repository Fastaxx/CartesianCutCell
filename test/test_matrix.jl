using CartesianCutCell

# Test the function backward difference
n = 5
D_sparse = backward_difference_matrix_sparse(n)
println("Backward Difference Matrix:")
println(D_sparse)

# Test the function forward difference
n = 5
D_sparse = forward_difference_matrix_sparse(n)
println("Forward Difference Matrix:")
println(D_sparse)

# Test the function backward interpolation
n = 5
D_sparse = backward_interpolation_matrix_sparse(n)
println("Backward Interpolation Matrix:")
println(D_sparse)

# Test the function forward interpolation
n = 5
D_sparse = forward_interpolation_matrix_sparse(n)
println("Forward Interpolation Matrix:")
println(D_sparse)

# Test the function identity matrix
n = 5
I_sparse = identity_matrix_sparse(n)
println("Identity Matrix:")
println(I_sparse)

# Test the function kronecker
C_sparse = kronecker_product_sparse(I_sparse, D_sparse)
println("Kronecker Product of A and B:")
println(C_sparse)