# Matrix.jl
## Introduction
This Julia file provides functions to create elementary matrices commonly used in numerical computations, particularly in the cut-cell method. These matrices include backward and forward differentiation matrices, as well as backward and forward interpolation matrices.

## Functions Provided
`backward_difference_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the backward differentiation matrix using the SparseMatrixCSC format from the SparseArrays package. The matrix has 1 on the diagonal and -1 below the diagonal, with the last coefficient in the diagonal being 0.

`forward_difference_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the forward differentiation matrix using the SparseMatrixCSC format. The matrix has -1 on the diagonal and 1 above the diagonal, with the last coefficient in the diagonal being 0.

`backward_interpolation_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the backward interpolation matrix using the SparseMatrixCSC format. The matrix has 1 on the diagonal and 1 below the diagonal, with the last coefficient in the diagonal being 0.

`forward_interpolation_matrix_sparse(n::Int)`: This function constructs a sparse matrix representing the forward interpolation matrix using the SparseMatrixCSC format. The matrix has 1 on the diagonal and 1 above the diagonal, with the last coefficient in the diagonal being 0.

## Usage
To use these functions, simply include the Julia file in your project and call the desired function with the appropriate parameters. Each function returns a sparse matrix representing the requested elementary matrix.

## Dependencies
These functions rely on the SparseArrays package for handling sparse matrices. Make sure to have this package installed before using the functions.