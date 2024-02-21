
# Function to solve G^T W^T G pw = Vfw - G^T W^T H gg - Poisson Equation Dirichlet
function solve_Ax_b_poisson(nx::Int, ny::Int, G, W, V, H, fw, gg)
    A = transpose(G) * W * G  # Construct the matrix A
    b = V * fw * - transpose(G) * W * H * gg # Construct the vector b
    x = A \ b       # Solve Ax = b
    return x
end

