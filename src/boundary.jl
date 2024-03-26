function get_cut_cells(levelset, xyz)
    nx = length(xyz[1])
    ny = length(xyz[2])
    cut_cells = CartesianIndex[]

    for i in 1:nx-1
        for j in 1:ny-1
            # Get the values of the level set at the corners of the cell
            values = [levelset(xyz[1][i], xyz[2][j]), levelset(xyz[1][i+1], xyz[2][j]), levelset(xyz[1][i], xyz[2][j+1]), levelset(xyz[1][i+1], xyz[2][j+1])]

            # If the level set changes sign across the cell, add it to the list of cut cells
            if any(x -> x < 0, values) && any(x -> x > 0, values)
                push!(cut_cells, CartesianIndex(i, j))
            end
        end
    end

    return cut_cells
end

function create_boundary(cut_cells, nx, ny, value)
    # Initialize a matrix of zeros
    matrix = zeros(nx, ny)

    # Assign the given value to the cut cells
    for index in cut_cells
        matrix[index] = value
    end

    return vec(matrix)
end

function get_border_cells(xyz)
    nx = length(xyz[1])
    ny = length(xyz[2])
    border_cells = CartesianIndex[]

    for i in 1:nx-1
        for j in 1:ny-1
            # If the cell is on the border of the mesh, add it to the list of border cells
            if i == 1 || i == nx-1 || j == 1 || j == ny-1
                push!(border_cells, CartesianIndex(i, j))
            end
        end
    end

    return border_cells
end

function get_volume_indices(V, max_volume)
    solid_indices = CartesianIndex[]
    fluid_indices = CartesianIndex[]
    cut_cell_indices = CartesianIndex[]

    for index in CartesianIndices(V)
        if V[index] == 0
            push!(solid_indices, index)
        elseif V[index] == max_volume
            push!(fluid_indices, index)
        else
            push!(cut_cell_indices, index)
        end
    end

    return solid_indices, fluid_indices, cut_cell_indices
end

# Function to build the diagonal matrices Ia and Ib
function build_diagonal_matrix_sparse(ag::Vector{T}) where T
    n = length(ag)
    Ia = spdiagm(0 => ag)
    return Ia
end

function build_igamma(HT::SparseMatrixCSC)
    vec_1 = [1 for i in 1:size(HT, 2)]

    # Calculer le produit
    row_sums = HT*vec_1

    # Prendre la valeur absolue de chaque élément
    abs_row_sums = abs.(row_sums)

    # Créer une matrice diagonale creuse avec elements ligne comme valeurs diagonales
    Igamma = spdiagm(0 => abs_row_sums)

    return Igamma
end

# Définir les structures pour chaque type de condition limite
struct DirichletCondition
    value::Union{Function, Float64}
end

struct NeumannCondition
    value::Union{Function, Float64}
end

struct RobinCondition
    alpha::Union{Function, Float64}
    beta::Union{Function, Float64}
    gamma::Union{Function, Float64}
end