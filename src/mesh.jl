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
    
    v_diag = spdiagm(0 => V) # Diagonal matrix of Volumes : V

    Ax = As[1] # Surface in x : Ax

    ax_diag = spdiagm(0 => Ax) # Diagonal matrix of Ax

    if length(xyz) == 1
        # If xyz is 1D, there is no Ay
        return V, v_diag, bary, ax_diag, nothing
    else
        Ay = As[2] # Surface in y : Ay
        ay_diag = spdiagm(0 => Ay) # Diagonal matrix of Ay
        return V, v_diag, bary, ax_diag, ay_diag
    end
end

function calculate_second_order_moments(levelset, xyz, bary)
    # Moments (2nd order)
    Ws = integrate(Tuple{0}, levelset, xyz, T, zero, bary)
    Wx = Ws[1] # Surface in x : Wx

    # Premiere facon de trouver les frontieres. Pb : Pas de distinction entre l'interface et la bordure du domaine
    border_cells_wx = [cell for (cell, value) in enumerate(Wx) if value == 0]

    wx_diag = spdiagm(0 => Wx) # Diagonal matrix of Wx

    # Surface
    Bs = integrate(Tuple{1}, levelset, xyz, T, zero, bary)
    Bx = Bs[1] # Surface in x : Bx
    
    bx_diag = spdiagm(0 => Bx) # Diagonal matrix of Bx

    if length(xyz) == 1
        # If xyz is 1D, there is no Wy, By
        return wx_diag, bx_diag, nothing, border_cells_wx, nothing
    else
        Wy = Ws[2] # Surface in y : Wy
        border_cells_wy = [cell for (cell, value) in enumerate(Wy) if value == 0]
        wy_diag = spdiagm(0 => Wy) # Diagonal matrix of Wy
        w_diag = blockdiag(wx_diag, wy_diag)

        By = Bs[2] # Surface in y : By
        by_diag = spdiagm(0 => By) # Diagonal matrix of By

        return w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy
    end
end



"""
# -1 : =0 (interface)
# 0 : >0 (inside)
# 1 : <0 (outside)
"""