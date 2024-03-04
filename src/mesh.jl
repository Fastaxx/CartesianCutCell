import Base: OneTo
const T = Float64

function generate_mesh(outer, inner)
    xy_collocated = collocated.(Ref(identity), outer, inner) # Generate a collocated mesh
    xy_staggered = staggered.(Ref(identity), outer, inner) # Generate a staggered mesh

    return xy_collocated, xy_staggered
end

function calculate_first_order_moments(levelset, xyz)
    # first-kind moments
    V, bary = integrate(Tuple{0}, levelset, xyz, T, zero)
    As = integrate(Tuple{1}, levelset, xyz, T, zero)
    mid = length(bary) ÷ 2
    barx = bary[mid] 
    @show  barx  
    v_diag = spdiagm(0 => V) # Diagonal matrix of Volumes : V

    Ax = As[1] # Surface in x : Ax
    Ay = As[2] # Surface in y : Ay

    ax_diag = spdiagm(0 => Ax) # Diagonal matrix of Ax
    ay_diag = spdiagm(0 => Ay) # Diagonal matrix of Ay

    return V, v_diag, bary, ax_diag, ay_diag
end 

function calculate_second_order_moments(levelset, xyz, bary)
    # Moments (2nd order)
    Ws = integrate(Tuple{0}, levelset, xyz, T, zero, bary)
    Wx = Ws[1] # Surface in x : Wx
    Wy = Ws[2] # Surface in y : Wy

    # Premiere facon de trouver les frontieres. Pb : Pas de distinction entre l'interface et la bordure du domaine
    border_cells_wx = [cell for (cell, value) in enumerate(Wx) if value == 0]
    border_cells_wy = [cell for (cell, value) in enumerate(Wy) if value == 0]

    wx_diag = spdiagm(0 => Wx) # Diagonal matrix of Wx
    wy_diag = spdiagm(0 => Wy) # Diagonal matrix of Wy

    w_diag = blockdiag(wx_diag, wy_diag)

    # Surface
    Bs = integrate(Tuple{1}, levelset, xyz, T, zero, bary)
    Bx = Bs[1] # Surface in x : Bx
    By = Bs[2] # Surface in y : By
    
    bx_diag = spdiagm(0 => Bx) # Diagonal matrix of Bx
    by_diag = spdiagm(0 => By) # Diagonal matrix of By

    return w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy
end



"""
# -1 : =0 (interface)
# 0 : >0 (inside)
# 1 : <0 (outside)
"""