using Test
using CartesianCutCell
include("../src/matrix.jl")

@testset "forward_difference_matrix_sparse Tests" begin
    # Test 1: Check if the matrix is sparse
    n = 5
    D_sparse = forward_difference_matrix_sparse(n)
    @test issparse(D_sparse)

    # Test 2: Check if the matrix is of size n x n
    @test size(D_sparse) == (n, n)

    # Test 3: Check if the last element in the diagonal is 0
    @test D_sparse[n, n] == 0.0

    # Test 4: Check if the other diagonal elements are -1
    for i in 1:n-1
        @test D_sparse[i, i] == -1.0
    end

    # Test 5: Check if the elements above the diagonal are 1
    for i in 1:n-1
        @test D_sparse[i, i+1] == 1.0
    end

    # Test 6: Check if the rest of the elements are 0
    for i in 1:n
        for j in 1:n
            if j != i && j != i+1
                @test D_sparse[i, j] == 0.0
            end
        end
    end
end

@testset "Interpolation Matrix Sparse Tests" begin
    n = 5

    # Test for backward_interpolation_matrix_sparse
    D_sparse = backward_interpolation_matrix_sparse(n)
    @test issparse(D_sparse)
    @test size(D_sparse) == (n, n)
    @test D_sparse[n, n] == 0.0
    for i in 1:n-1
        @test D_sparse[i, i] == 0.5
        @test D_sparse[i+1, i] == 0.5
    end

    # Test for forward_interpolation_matrix_sparse
    D_sparse = forward_interpolation_matrix_sparse(n)
    @test issparse(D_sparse)
    @test size(D_sparse) == (n, n)
    @test D_sparse[n, n] == 0.0
    for i in 1:n-1
        @test D_sparse[i, i] == 0.5
        @test D_sparse[i, i+1] == 0.5
    end
end

@testset "identity_matrix_sparse Tests" begin
    # Test 1: Check if the matrix is sparse
    n = 5
    I_sparse = identity_matrix_sparse(n)
    @test issparse(I_sparse)

    # Test 2: Check if the matrix is of size n x n
    @test size(I_sparse) == (n, n)

    # Test 3: Check if the diagonal elements are 1
    for i in 1:n
        @test I_sparse[i, i] == 1.0
    end

    # Test 4: Check if the off-diagonal elements are 0
    for i in 1:n
        for j in 1:n
            if i != j
                @test I_sparse[i, j] == 0.0
            end
        end
    end
end

@testset "Interpolation Matrix Sparse 2D Tests" begin
    nx = 5
    ny = 5

    # Test for backward_interpolation_matrix_sparse_2D_x
    D_sparse = backward_interpolation_matrix_sparse_2D_x(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)

    # Test for forward_interpolation_matrix_sparse_2D_x
    D_sparse = forward_interpolation_matrix_sparse_2D_x(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)

    # Test for backward_interpolation_matrix_sparse_2D_y
    D_sparse = backward_interpolation_matrix_sparse_2D_y(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)

    # Test for forward_interpolation_matrix_sparse_2D_y
    D_sparse = forward_interpolation_matrix_sparse_2D_y(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)
end

@testset "Difference Matrix Sparse 3D Tests" begin
    nx = 5
    ny = 5
    nz = 5

    # Test for backward_difference_matrix_sparse_3D_x
    D_sparse = backward_difference_matrix_sparse_3D_x(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)
end

@testset "Difference Matrix Sparse 2D Tests" begin
    nx = 5
    ny = 5

    # Test for backward_difference_matrix_sparse_2D_x
    D_sparse = backward_difference_matrix_sparse_2D_x(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)

    # Test for forward_difference_matrix_sparse_2D_x
    D_sparse = forward_difference_matrix_sparse_2D_x(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)

    # Test for backward_difference_matrix_sparse_2D_y
    D_sparse = backward_difference_matrix_sparse_2D_y(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)

    # Test for forward_difference_matrix_sparse_2D_y
    D_sparse = forward_difference_matrix_sparse_2D_y(nx, ny)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny, nx*ny)
end

@testset "Difference Matrix Sparse 3D Tests" begin
    nx = 5
    ny = 5
    nz = 5

    # Test for backward_difference_matrix_sparse_3D_x
    D_sparse = backward_difference_matrix_sparse_3D_x(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for forward_difference_matrix_sparse_3D_x
    D_sparse = forward_difference_matrix_sparse_3D_x(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for backward_difference_matrix_sparse_3D_y
    D_sparse = backward_difference_matrix_sparse_3D_y(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for forward_difference_matrix_sparse_3D_y
    D_sparse = forward_difference_matrix_sparse_3D_y(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)
end

@testset "Interpolation Matrix Sparse 3D Tests" begin
    nx = 5
    ny = 5
    nz = 5

    # Test for backward_interpolation_matrix_sparse_3D_x
    D_sparse = backward_interpolation_matrix_sparse_3D_x(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for forward_interpolation_matrix_sparse_3D_x
    D_sparse = forward_interpolation_matrix_sparse_3D_x(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for backward_interpolation_matrix_sparse_3D_y
    D_sparse = backward_interpolation_matrix_sparse_3D_y(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for forward_interpolation_matrix_sparse_3D_y
    D_sparse = forward_interpolation_matrix_sparse_3D_y(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for backward_interpolation_matrix_sparse_3D_z
    D_sparse = backward_interpolation_matrix_sparse_3D_z(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)

    # Test for forward_interpolation_matrix_sparse_3D_z
    D_sparse = forward_interpolation_matrix_sparse_3D_z(nx, ny, nz)
    @test issparse(D_sparse)
    @test size(D_sparse) == (nx*ny*nz, nx*ny*nz)
end