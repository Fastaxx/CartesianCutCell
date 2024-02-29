using SparseArrays
function backward_difference_matrix_sparse(n::Int)
    D_sparse = spdiagm(0 => ones(n), -1 => -ones(n-1))
    D_sparse[n, n] = 0.0
    return D_sparse
end

function forward_difference_matrix_sparse(n::Int)
    D_sparse = spdiagm(0 => -ones(n), 1 => ones(n-1))
    D_sparse[n, n] = 0.0
    return D_sparse
end

function backward_interpolation_matrix_sparse(n::Int)
    D_sparse = 0.5 * spdiagm(0 => ones(n), -1 => ones(n-1))
    D_sparse[n, n] = 0.0
    return D_sparse
end

function forward_interpolation_matrix_sparse(n::Int)
    D_sparse = 0.5 * spdiagm(0 => ones(n), 1 => ones(n-1))
    D_sparse[n, n] = 0.0
    return D_sparse
end
 
function identity_matrix_sparse(n::Int)
    I_sparse = spdiagm(0 => ones(n))
    return I_sparse
end

function backward_difference_matrix_sparse_2D_x(nx::Int, ny::Int)
    I_ny = identity_matrix_sparse(ny)
    D_nx = backward_difference_matrix_sparse(nx)
    return kron(I_ny, D_nx)
end

function forward_difference_matrix_sparse_2D_x(nx::Int, ny::Int)
    I_ny = identity_matrix_sparse(ny)
    D_nx = forward_difference_matrix_sparse(nx)
    return kron(I_ny, D_nx)
end

function backward_difference_matrix_sparse_2D_y(nx::Int, ny::Int)
    D_ny = backward_difference_matrix_sparse(ny)
    I_nx = identity_matrix_sparse(nx)
    return kron(D_ny, I_nx)
end

function forward_difference_matrix_sparse_2D_y(nx::Int, ny::Int)
    D_ny = forward_difference_matrix_sparse(ny)
    I_nx = identity_matrix_sparse(nx)
    return kron(D_ny, I_nx)
end

function backward_interpolation_matrix_sparse_2D_x(nx::Int, ny::Int)
    I_ny = identity_matrix_sparse(ny)
    S_nx = backward_interpolation_matrix_sparse(nx)
    return kron(I_ny, S_nx)
end

function forward_interpolation_matrix_sparse_2D_x(nx::Int, ny::Int)
    I_ny = identity_matrix_sparse(ny)
    S_nx = forward_interpolation_matrix_sparse(nx)
    return kron(I_ny, S_nx)
end

function backward_interpolation_matrix_sparse_2D_y(nx::Int, ny::Int)
    S_ny = backward_interpolation_matrix_sparse(ny)
    I_nx = identity_matrix_sparse(nx)
    return kron(S_ny, I_nx)
end

function forward_interpolation_matrix_sparse_2D_y(nx::Int, ny::Int)
    S_ny = forward_interpolation_matrix_sparse(ny)
    I_nx = identity_matrix_sparse(nx)
    return kron(S_ny, I_nx)
end

function backward_difference_matrix_sparse_3D_x(nx::Int, ny::Int, nz::Int)
    I_ny = identity_matrix_sparse(ny)
    I_nz = identity_matrix_sparse(nz)
    D_nx = backward_difference_matrix_sparse(nx)
    return kron(kron(I_nz, I_ny), D_nx)
end

function forward_difference_matrix_sparse_3D_x(nx::Int, ny::Int, nz::Int)
    I_ny = identity_matrix_sparse(ny)
    I_nz = identity_matrix_sparse(nz)
    D_nx = forward_difference_matrix_sparse(nx)
    return kron(kron(I_nz, I_ny), D_nx)
end

function backward_difference_matrix_sparse_3D_y(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_nz = identity_matrix_sparse(nz)
    D_ny = backward_difference_matrix_sparse(ny)
    return kron(kron(I_nz, D_ny), I_nx)
end

function forward_difference_matrix_sparse_3D_y(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_nz = identity_matrix_sparse(nz)
    D_ny = forward_difference_matrix_sparse(ny)
    return kron(kron(I_nz, D_ny), I_nx)
end

function backward_difference_matrix_sparse_3D_z(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_ny = identity_matrix_sparse(ny)
    D_nz = backward_difference_matrix_sparse(nz)
    return kron(kron(D_nz, I_ny), I_nx)
end

function forward_difference_matrix_sparse_3D_z(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_ny = identity_matrix_sparse(ny)
    D_nz = forward_difference_matrix_sparse(nz)
    return kron(kron(D_nz, I_ny), I_nx)
end

function backward_interpolation_matrix_sparse_3D_x(nx::Int, ny::Int, nz::Int)
    I_ny = identity_matrix_sparse(ny)
    I_nz = identity_matrix_sparse(nz)
    S_nx = backward_interpolation_matrix_sparse(nx)
    return kron(kron(I_nz, I_ny), S_nx)
end

function forward_interpolation_matrix_sparse_3D_x(nx::Int, ny::Int, nz::Int)
    I_ny = identity_matrix_sparse(ny)
    I_nz = identity_matrix_sparse(nz)
    S_nx = forward_interpolation_matrix_sparse(nx)
    return kron(kron(I_nz, I_ny), S_nx)
end

function backward_interpolation_matrix_sparse_3D_y(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_nz = identity_matrix_sparse(nz)
    S_ny = backward_interpolation_matrix_sparse(ny)
    return kron(kron(I_nz, S_ny), I_nx)
end

function forward_interpolation_matrix_sparse_3D_y(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_nz = identity_matrix_sparse(nz)
    S_ny = forward_interpolation_matrix_sparse(ny)
    return kron(kron(I_nz, S_ny), I_nx)
end

function backward_interpolation_matrix_sparse_3D_z(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_ny = identity_matrix_sparse(ny)
    S_nz = backward_interpolation_matrix_sparse(nz)
    return kron(kron(S_nz, I_ny), I_nx)
end

function forward_interpolation_matrix_sparse_3D_z(nx::Int, ny::Int, nz::Int)
    I_nx = identity_matrix_sparse(nx)
    I_ny = identity_matrix_sparse(ny)
    S_nz = forward_interpolation_matrix_sparse(nz)
    return kron(kron(S_nz, I_ny), I_nx)
end