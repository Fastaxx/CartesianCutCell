# Cartesian Cut Cell method

## Geometry

## Mesh

## Matrix.jl
### Introduction
This Julia file provides functions to create elementary matrices commonly used in numerical computations, particularly in the cut-cell method. These matrices include backward and forward differentiation matrices, as well as backward and forward interpolation matrices.

### Functions Provided
`backward_difference_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the backward differentiation matrix using the SparseMatrixCSC format from the SparseArrays package. The matrix has 1 on the diagonal and -1 below the diagonal, with the last coefficient in the diagonal being 0.

`forward_difference_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the forward differentiation matrix using the SparseMatrixCSC format. The matrix has -1 on the diagonal and 1 above the diagonal, with the last coefficient in the diagonal being 0.

`backward_interpolation_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the backward interpolation matrix using the SparseMatrixCSC format. The matrix has 1 on the diagonal and 1 below the diagonal, with the last coefficient in the diagonal being 0.

`forward_interpolation_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the forward interpolation matrix using the SparseMatrixCSC format. The matrix has 1 on the diagonal and 1 above the diagonal, with the last coefficient in the diagonal being 0.

### Usage
To use these functions, simply include the Julia file in your project and call the desired function with the appropriate parameters. Each function returns a sparse matrix representing the requested elementary matrix.

### Dependencies
These functions rely on the SparseArrays package for handling sparse matrices. Make sure to have this package installed before using the functions.

## Operators.jl

### Overview
This Julia file contains functions to create various operators including the gradient, divergence, and matrix calculations.

### Functions
`build_matrix_G`: This function constructs the matrix G, which is used to compute the gradient of a scalar field in a fluid domain. It takes as input the dimensions of the mesh (nx and ny), along with certain difference operators (Dx_minus, Dy_minus, Bx, By), and returns the constructed matrix G.

`build_matrix_G_T`: Similar to the previous function, this one constructs the transpose of the matrix G, denoted as G^T, used in certain computations involving gradients.

`build_matrix_H`: Constructs the matrix H, which plays a role in computing the gradient of a scalar field. It also takes difference operators (Dx_minus, Dy_minus, Ax, Ay, Bx, By) as inputs.

`build_matrix_H_T`: Constructs the transpose of the matrix H, denoted as H^T, used in certain computations involving gradients.

`build_matrix_GTHT`: Constructs the matrix -(G^T + H^T), used in the computation of the divergence operator. It takes difference operators (Dx_plus, Dy_plus, Ax, Ay) as inputs.

`compute_grad_operator`: Computes the gradient of a scalar field using the matrices G, H, and their transpose counterparts. It takes the scalar field values (p_omega and p_gamma) and the transpose of the weight matrix (Wdagger) as inputs.

`compute_divergence`: Computes the divergence of a vector field using the matrices G^T and H^T. It takes the vector field components (q_omega and q_gamma) as inputs. Additionally, it has a boolean parameter gradient which, if set to true, computes only the gradient without involving H^T.

## Solve.jl

### Overview
This file contains functions for solving linear systems with different boundary conditions: Dirichlet, Neumann, and Robin conditions.

### Functions Provided

`solve_Ax_b_poisson`: Solves the Poisson equation with Dirichlet boundary conditions using the linear system \(G^T W^T G p_ω = V f_ω - G^T W^T H g_γ\). It constructs the matrix \(A = G^T W^T G\) and the vector \(b = V f_ω - G^T W^T H g_γ\), then solves the system \(Ax = b\) and returns the solution \(x\).

`construct_block_matrix_neumann`: Constructs the block matrix \(A\) for solving the linear system with Neumann boundary conditions. It combines the matrices \(G^T W^T G\), \(G^T W^T H\), \(H^T W^T G\), and \(H^T W^T H\) into a single block matrix.

`construct_rhs_vector_neumann`: Constructs the right-hand side vector \(b\) for the linear system with Neumann boundary conditions.

`solve_Ax_b_neumann`: Solves the linear system with Neumann boundary conditions by constructing the block matrix \(A\) using `construct_block_matrix_neumann`, the right-hand side vector \(b\) using `construct_rhs_vector_neumann`, and then solving the system \(Ax = b\). It returns the solution \(p_ω\) and \(p_γ\).

`construct_block_matrix_robin`: Constructs the block matrix \(A\) for solving the linear system with Robin boundary conditions. It combines the matrices \(G^T W^T G\), \(G^T W^T H\), \(I_b H^T W^T G\), \(I_b H^T W^T H\), and \(I_a I_Γ\) into a single block matrix.

`construct_rhs_vector_robin`: Constructs the right-hand side vector \(b\) for the linear system with Robin boundary conditions.

`solve_Ax_b_robin`: Solves the linear system with Robin boundary conditions by constructing the block matrix \(A\) using `construct_block_matrix_robin`, the right-hand side vector \(b\) using `construct_rhs_vector_robin`, and then solving the system \(Ax = b\). It returns the solution \(p_ω\) and \(p_γ\).

These functions provide a comprehensive set of tools for solving linear systems associated with different boundary conditions encountered in fluid dynamics simulations. They enable efficient and accurate numerical computations in fluid flow problems.
