using SparseArrays

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

using Test

function test_clip_volume_moments()
    # Créer un cas de test
    V = sparse([1, 2, 3], [1, 2, 3], [0.1, 0.5, 0.9])
    dx = dy = 1.0
    epsilon = 0.2

    # Appeler la fonction avec le cas de test
    clipped_V = clip_volume_moments(V, dx, dy, epsilon)

    # Vérifier que les valeurs ont été coupées correctement
    @test clipped_V[1, 1] == 0.0
    @test clipped_V[2, 2] == 0.5
    @test clipped_V[3, 3] == 1.0
end

function test_clip_surface_moments()
    # Créer un cas de test
    A_alpha = sparse([1, 2, 3], [1, 2, 3], [0.1, 0.5, 0.9])
    d_alpha = 1.0
    epsilon = 0.2

    # Appeler la fonction avec le cas de test
    clipped_A_alpha = clip_surface_moments(A_alpha, d_alpha, epsilon)

    # Vérifier que les valeurs ont été coupées correctement
    @test clipped_A_alpha[1, 1] == 0.0
    @test clipped_A_alpha[2, 2] == 0.5
    @test clipped_A_alpha[3, 3] == 0.9
end

# Exécuter les tests
test_clip_volume_moments()
test_clip_surface_moments()
