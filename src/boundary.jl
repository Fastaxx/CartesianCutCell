using SparseArrays
using Test
# Function to build the diagonal matrices Ia and Ib
function build_diagonal_matrix_sparse(ag::Vector{T}) where T
    n = length(ag)
    Ia = spdiagm(0 => ag)
    return Ia
end

"""
# Example usage:
ag = [1.0, 2.0, 3.0, 4.0]  # Example vector
Ia_sparse = build_diagonal_matrix_sparse(ag)
println(Ia_sparse)
"""
function build_igamma(HT::SparseMatrixCSC)
    # Calculer la somme de chaque ligne de HT
    row_sums = sum(HT, dims=2)

    # Prendre la valeur absolue de chaque somme
    abs_row_sums = abs.(row_sums)

    # Créer une matrice diagonale creuse avec les sommes de ligne comme valeurs diagonales
    Igamma = spdiagm(0 => vec(abs_row_sums))

    return Igamma
end

"""
# Test pour la fonction Igamma
@testset "Igamma" begin
    HT = sparse([1.0 2.0; 3.0 4.0])
    Igamma_matrix = build_igamma(HT)
    @test Igamma_matrix == spdiagm(0 => vec(abs.(sum(HT, dims=2))))
end
"""