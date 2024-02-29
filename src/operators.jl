# Function to build the matrix G
function build_matrix_G(nx::Int, ny::Int, Dx_minus, Dy_minus, Bx, By)    
    # Compute matrix products
    Dx_minus_Bx = Dx_minus * Bx
    Dy_minus_By = Dy_minus * By
    
    # Concatenate vertically to obtain G
    G = vcat(Dx_minus_Bx, Dy_minus_By)
    
    return G
end

# Function to build the matrix Gt
function build_matrix_G_T(nx::Int, ny::Int, Dx_plus, Dy_plus, Bx, By)    
    # Compute matrix products
    Dx_plus_Bx = Bx * Dx_plus
    Dy_plus_By = By * Dy_plus
    
    # Concatenate vertically to obtain G
    G = - hcat(Dx_plus_Bx, Dy_plus_By)
    
    return G
end

# Function to create the matrix H
function build_matrix_H(nx::Int, ny::Int, Dx_minus, Dy_minus, Ax, Ay, Bx, By)
    # Compute matrix products
    Ax_Dx_minus = Ax * Dx_minus
    Ay_Dy_minus = Ay * Dy_minus
    Dx_minus_Bx = Dx_minus * Bx
    Dy_minus_By = Dy_minus * By
    
    # Compute blocks of H
    block1 = Ax_Dx_minus - Dx_minus_Bx
    block2 = Ay_Dy_minus - Dy_minus_By
    
    # Concatenate vertically to obtain H
    H = vcat(block1, block2)
    
    return H
end

# Function to create the matrix Ht
function build_matrix_H_T(nx::Int, ny::Int, Dx_plus, Dy_plus, Ax, Ay, Bx, By)
    # Compute matrix products
    Dx_plus_Ax = Dx_plus * Ax
    Dy_plus_Ay = Dy_plus * Ay 
    Bx_Dx_plus = Bx * Dx_plus
    By_Dy_plus = By * Dy_plus
    
    # Compute blocks of H^T
    block1 = Bx_Dx_plus - Dx_plus_Ax
    block2 = By_Dy_plus - Dy_plus_Ay
    
    # Concatenate horizontally to obtain H
    Ht = hcat(block1, block2)
    
    return Ht
end

# Function to create the matrix -(G^T+H^T)
function build_matrix_GTHT(nx::Int, ny::Int, Dx_plus, Dy_plus, Ax, Ay)
    # Compute matrix products
    Dx_plus_Ax = Dx_plus * Ax
    Dy_plus_Ay = Dy_plus * Ay
    
    # Compute blocks of -(G^T+H^T)
    block1 = Dx_plus_Ax
    block2 = Dy_plus_Ay
    
    # Concatenate vertically to obtain H
    minus_GTHT = hcat(block1, block2)
    
    return minus_GTHT
end

# Function to compute the grad operator
function compute_grad_operator(p_omega, p_gamma, Wdagger, G, H)
    # Compute G * pw
    G_pw = G * p_omega
    
    # Compute H * pg
    H_pgamma = H * p_gamma
    
    # Compute grad = Wdagger * (G * p_omega + H * p_gamma)
    grad = Wdagger * (G_pw + H_pgamma)
    
    return grad
end

# Function to compute the div operator
function compute_divergence(q_omega, q_gamma, GT, minus_GTHT, HT, gradient::Bool=false)
    if gradient
        divergence = - GT * q_omega
    else
        # Compute -(G^T + H^T) * q_omega
        div_qw = minus_GTHT * q_omega
    
        # Compute H^T * q_gamma
        div_qg = HT * q_gamma
    
        # Compute divergence
        divergence = div_qw + div_qg
    end
    return divergence
end

function sparse_inverse(W::SparseMatrixCSC)
    # Extraire les valeurs diagonales
    diag_values = diag(W)

    # Inverser les valeurs non nulles, remplacer les zéros par des uns
    diag_values = [val != 0 ? 1/val : 1 for val in diag_values]

    diag_values = float.(diag_values)

    # Créer une nouvelle matrice diagonale creuse avec les nouvelles valeurs
    W_inv = spdiagm(0 => diag_values)

    return W_inv
end

function Cx(ux_ω, uy_ω,  Dx_minus, Sx_plus, Ax, Dy_minus, Sy_plus, Ay)
    term1 = diagm(0 => Dx_minus * Sx_plus * Ax * ux_ω) * Sx_plus
    term2 = diagm(0 => Dy_minus * Sx_plus * Ay * uy_ω) * Sy_plus
    return term1 + term2
end

function Kx(u_γ, Sx_minus, HT)
    return Sx_minus * diagm(0 => (HT * u_γ))
end

function Cy(ux_ω, uy_ω,  Dx_minus, Sx_plus, Ax, Dy_minus, Sy_plus, Ay)
    term1 = diagm(0 => Dx_minus * Sy_plus * Ax * ux_ω) * Sx_plus
    term2 = diagm(0 => Dy_minus * Sy_plus * Ay * uy_ω) * Sy_plus
    return term1 + term2
end

function Ky(u_γ, Sy_minus, HT)
    return Sy_minus * diagm(0 => (HT * u_γ))
end

function convx(Dx_minus, Dy_minus, Sx_plus, Sy_plus, Sx_minus, Ax, Ay, HT, u_γ, qγ_x, qγ_y, ux_ω, uy_ω)
    term1 = Cx(ux_ω, uy_ω,  Dx_minus, Sx_plus, Ax, Dy_minus, Sy_plus, Ay) * qγ_x
    term2 = Kx(u_γ, Sx_minus, HT) * (qγ_x+qγ_y)/2
    return term1 + term2 
end

function convy(Dx_minus, Dy_minus, Sx_plus, Sy_plus, Sx_minus, Ax, Ay, HT, u_γ, qγ_x, qγ_y, ux_ω, uy_ω)
    term1 = Cy(ux_ω, uy_ω,  Dx_minus, Sx_plus, Ax, Dy_minus, Sy_plus, Ay) * qγ_y
    term2 = Ky(u_γ, Sy_minus, HT) * (qγ_x+qγ_y)/2
    return term1 + term2 
end