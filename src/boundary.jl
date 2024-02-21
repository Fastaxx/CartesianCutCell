using SparseArrays

# Function to build the diagonal matrices Ia and Ib
function build_diagonal_matrix_sparse(ag::Vector{T}) where T
    n = length(ag)
    Ia = spdiagm(0 => ag)
    return Ia
end

# Example usage:
ag = [1.0, 2.0, 3.0, 4.0]  # Example vector
Ia_sparse = build_diagonal_matrix_sparse(ag)
println(Ia_sparse)

