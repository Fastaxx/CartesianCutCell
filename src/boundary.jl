using SparseArrays
using Test

function get_cut_cells(levelset, xyz)
    nx = length(xyz[1])
    ny = length(xyz[2])
    cut_cells = []

    for i in 1:nx-1
        for j in 1:ny-1
            # Get the values of the level set at the corners of the cell
            values = [levelset(xyz[1][i], xyz[2][j]), levelset(xyz[1][i+1], xyz[2][j]), levelset(xyz[1][i], xyz[2][j+1]), levelset(xyz[1][i+1], xyz[2][j+1])]

            # If the level set changes sign across the cell, add it to the list of cut cells
            if any(x -> x < 0, values) && any(x -> x > 0, values)
                push!(cut_cells, (i, j))
            end
        end
    end

    return cut_cells
end

function create_boundary(cut_cells, nx, ny, value)
    # Initialize a vector of zeros
    vector = zeros(nx * ny)

    # Assign the given value to the cut cells
    for (i, j) in cut_cells
        index = (j-1) * nx + i
        vector[index] = value
    end

    return vector
end

function get_border_cells(xyz)
    nx = length(xyz[1])
    ny = length(xyz[2])
    border_cells = []

    for i in 1:nx-1
        for j in 1:ny-1
            # If the cell is on the border of the mesh, add it to the list of border cells
            if i == 1 || i == nx-1 || j == 1 || j == ny-1
                push!(border_cells, (i, j))
            end
        end
    end

    return border_cells
end


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