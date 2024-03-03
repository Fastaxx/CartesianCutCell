# Solve.jl

## Overview
This file contains functions for solving linear systems with different boundary conditions: Dirichlet, Neumann, and Robin conditions.

## Functions Provided

`solve_Ax_b_poisson`: Solves the Poisson equation with Dirichlet boundary conditions using the linear system \(G^T W^T G p_ω = V f_ω - G^T W^T H g_γ\). It constructs the matrix \(A = G^T W^T G\) and the vector \(b = V f_ω - G^T W^T H g_γ\), then solves the system \(Ax = b\) and returns the solution \(x\).

`construct_block_matrix_neumann`: Constructs the block matrix \(A\) for solving the linear system with Neumann boundary conditions. It combines the matrices \(G^T W^T G\), \(G^T W^T H\), \(H^T W^T G\), and \(H^T W^T H\) into a single block matrix.

`construct_rhs_vector_neumann`: Constructs the right-hand side vector \(b\) for the linear system with Neumann boundary conditions.

`solve_Ax_b_neumann`: Solves the linear system with Neumann boundary conditions by constructing the block matrix \(A\) using `construct_block_matrix_neumann`, the right-hand side vector \(b\) using `construct_rhs_vector_neumann`, and then solving the system \(Ax = b\). It returns the solution \(p_ω\) and \(p_γ\).

`construct_block_matrix_robin`: Constructs the block matrix \(A\) for solving the linear system with Robin boundary conditions. It combines the matrices \(G^T W^T G\), \(G^T W^T H\), \(I_b H^T W^T G\), \(I_b H^T W^T H\), and \(I_a I_Γ\) into a single block matrix.

`construct_rhs_vector_robin`: Constructs the right-hand side vector \(b\) for the linear system with Robin boundary conditions.

`solve_Ax_b_robin`: Solves the linear system with Robin boundary conditions by constructing the block matrix \(A\) using `construct_block_matrix_robin`, the right-hand side vector \(b\) using `construct_rhs_vector_robin`, and then solving the system \(Ax = b\). It returns the solution \(p_ω\) and \(p_γ\).

