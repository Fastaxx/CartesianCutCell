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


"""
    clip_volume_moments(V, dx, dy, epsilon)

Clip volume moments of a SparseArrayCSC matrix `V` based on the specified threshold `epsilon`.
The clipped volume moments are returned as a new SparseArrayCSC matrix.

# Arguments
- `V::SparseMatrixCSC`: SparseArrayCSC matrix representing volume moments.
- `dx::Float64`: Cell width.
- `dy::Float64`: Cell height.
- `epsilon::Float64`: Threshold value for clipping.

# Returns
- `clipped_V::SparseMatrixCSC`: SparseArrayCSC matrix with clipped volume moments.
"""
function clip_volume_moments(V, dx, dy, epsilon)
    clipped_V_values = similar(V.nzval)
    for i in 1:length(V.nzval)
        if V.nzval[i] < epsilon
            clipped_V_values[i] = 0.0
        elseif V.nzval[i] > dx * dy - epsilon
            clipped_V_values[i] = dx * dy
        else
            clipped_V_values[i] = V.nzval[i]
        end
    end
    clipped_V = sparse(V.rowval, V.colptr, clipped_V_values)
    return clipped_V
end



"""
    clip_surface_moments(A_alpha, d_alpha, epsilon)

Clip surface moments of a SparseArrayCSC matrix `A_alpha` based on the specified threshold `epsilon`.
The clipped surface moments are returned as a new SparseArrayCSC matrix.

# Arguments
- `A_alpha::SparseMatrixCSC`: SparseArrayCSC matrix representing surface moments.
- `d_alpha::Float64`: Cell dimension along the specified direction (x or y).
- `epsilon::Float64`: Threshold value for clipping.

# Returns
- `clipped_A::SparseMatrixCSC`: SparseArrayCSC matrix with clipped surface moments.
"""
function clip_surface_moments(A_alpha, delta_alpha, epsilon)
    clipped_A_values = similar(A_alpha.nzval)
    for i in 1:length(A_alpha.nzval)
        if A_alpha.nzval[i] < sqrt(epsilon)
            clipped_A_values[i] = 0.0
        elseif A_alpha.nzval[i] > delta_alpha - sqrt(epsilon)
            clipped_A_values[i] = delta_alpha
        else
            clipped_A_values[i] = A_alpha.nzval[i]
        end
    end
    clipped_A = sparse(A_alpha.rowval, A_alpha.colptr, clipped_A_values)
    return clipped_A
end

