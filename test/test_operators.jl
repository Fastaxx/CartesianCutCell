using Test

@testset "build_matrix_G Tests" begin
    # Test 1: Check if the matrix is of size 2*nx*ny x nx*ny
    nx, ny = 5, 5
    Dx_minus, Dy_minus = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Bx, By = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    G = build_matrix_G(nx, ny, Dx_minus, Dy_minus, Bx, By)
    @test size(G) == (2*nx*ny, nx*ny)

    # Test 2: Check if the first half of G is equal to Dx_minus * Bx
    @test G[1:nx*ny, :] == Dx_minus * Bx

    # Test 3: Check if the second half of G is equal to Dy_minus * By
    @test G[nx*ny+1:end, :] == Dy_minus * By
end


@testset "build_matrix_G_T Tests" begin
    # Test 1: Check if the matrix is of size nx*ny x 2*nx*ny
    nx, ny = 5, 5
    Dx_plus, Dy_plus = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Bx, By = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    G = build_matrix_G_T(nx, ny, Dx_plus, Dy_plus, Bx, By)
    @test size(G) == (nx*ny, 2*nx*ny)

    # Test 2: Check if the first half of G is equal to -Bx * Dx_plus
    @test G[:, 1:nx*ny] == -Bx * Dx_plus

    # Test 3: Check if the second half of G is equal to -By * Dy_plus
    @test G[:, nx*ny+1:end] == -By * Dy_plus
end

@testset "build_matrix_H Tests" begin
    # Test 1: Check if the matrix is of size 2*nx*ny x nx*ny
    nx, ny = 5, 5
    Dx_minus, Dy_minus = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Ax, Ay = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Bx, By = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    H = build_matrix_H(nx, ny, Dx_minus, Dy_minus, Ax, Ay, Bx, By)
    @test size(H) == (2*nx*ny, nx*ny)

    # Test 2: Check if the first half of H is equal to Ax*Dx_minus - Dx_minus*Bx
    @test H[1:nx*ny, :] == Ax*Dx_minus - Dx_minus*Bx

    # Test 3: Check if the second half of H is equal to Ay*Dy_minus - Dy_minus*By
    @test H[nx*ny+1:end, :] == Ay*Dy_minus - Dy_minus*By
end

@testset "build_matrix_H_T Tests" begin
    # Test 1: Check if the matrix is of size nx*ny x 2*nx*ny
    nx, ny = 5, 5
    Dx_plus, Dy_plus = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Ax, Ay = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Bx, By = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Ht = build_matrix_H_T(nx, ny, Dx_plus, Dy_plus, Ax, Ay, Bx, By)
    @test size(Ht) == (nx*ny, 2*nx*ny)

    # Test 2: Check if the first half of Ht is equal to Bx*Dx_plus - Dx_plus*Ax
    @test Ht[:, 1:nx*ny] == Bx*Dx_plus - Dx_plus*Ax

    # Test 3: Check if the second half of Ht is equal to By*Dy_plus - Dy_plus*Ay
    @test Ht[:, nx*ny+1:end] == By*Dy_plus - Dy_plus*Ay
end

@testset "build_matrix_GTHT Tests" begin
    # Test 1: Check if the matrix is of size nx*ny x 2*nx*ny
    nx, ny = 5, 5
    Dx_plus, Dy_plus = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    Ax, Ay = rand(nx*ny, nx*ny), rand(nx*ny, nx*ny)
    minus_GTHT = build_matrix_GTHT(nx, ny, Dx_plus, Dy_plus, Ax, Ay)
    @test size(minus_GTHT) == (nx*ny, 2*nx*ny)

    # Test 2: Check if the first half of minus_GTHT is equal to Dx_plus * Ax
    @test minus_GTHT[:, 1:nx*ny] == Dx_plus * Ax

    # Test 3: Check if the second half of minus_GTHT is equal to Dy_plus * Ay
    @test minus_GTHT[:, nx*ny+1:end] == Dy_plus * Ay
end


@testset "compute_grad_operator Tests" begin
    # Test 1: Check if the output is a vector of size nx*ny
    nx, ny = 5, 5
    p_omega, p_gamma = rand(nx*ny), rand(nx*ny)
    Wdagger, G, H = rand(nx*ny, 2*nx*ny), rand(2*nx*ny, nx*ny), rand(2*nx*ny, nx*ny)
    grad = compute_grad_operator(p_omega, p_gamma, Wdagger, G, H)
    @test size(grad) == (nx*ny,)

    # Test 2: Check if the output is equal to Wdagger * (G * p_omega + H * p_gamma)
    @test grad == Wdagger * (G * p_omega + H * p_gamma)
end