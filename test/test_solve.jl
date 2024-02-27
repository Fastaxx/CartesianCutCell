using Test
using LinearAlgebra
using SparseArrays
include("../src/solve.jl")

@testset "solve_Ax_b_poisson Tests" begin
    # Test 1: Check if the output is the solution of Ax = b
    nx, ny = 5, 5
    G = rand(2*nx*ny, nx*ny)
    GT = transpose(G)
    Wdagger = rand(2*nx*ny, 2*nx*ny)
    H = rand(2*nx*ny, nx*ny)
    V = rand(nx*ny, nx*ny)
    f_omega = rand(nx*ny, nx*ny)
    g_gamma = rand(nx*ny, nx*ny)
    x = solve_Ax_b_poisson(nx, ny, G, GT, Wdagger, H, V, f_omega, g_gamma)
    A = GT * Wdagger * G
    b = V * f_omega - GT * Wdagger * H * g_gamma
    @test A * x ≈ b
end