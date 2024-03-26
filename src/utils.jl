function volume_integrated_p_norm(e1, e2, V, p)
    if p == Inf
        total_volume = sum(V)
        abs_error = abs.(e1 - e2) .* V
        integral = maximum(abs_error)
        return integral
    else
        total_volume = sum(V)
        abs_error = abs.(e1 - e2) .^p
        integral = sum(abs_error .* V)/ total_volume
        return (integral)^(1/p)
    end
end

function get_condition_number_L2_svd(A)
    matrix = Array(A)
    val = svdvals(matrix)
    return val[1]/val[end]
end

function cut_cell_statistics(V, cut_cell_indices)
    # Nombre total de cellules
    total_cells = length(V)

    # Nombre de cut-cells
    num_cut_cells = length(cut_cell_indices)

    # Pourcentage de cut-cells
    percentage_cut_cells = (num_cut_cells / total_cells) * 100

    # Pourcentage de cellules non coupées
    percentage_uncut_cells = 100 - percentage_cut_cells

    # Volumes des cut-cells
    cut_cell_volumes = [V[i] for i in cut_cell_indices]

    # Volume minimal et maximal des cut-cells
    min_volume = minimum(cut_cell_volumes)
    max_volume = maximum(cut_cell_volumes)
 
    # Écart type du volume des cut-cells
    std_dev_volume = std(cut_cell_volumes)
 
    # Médiane du volume des cut-cells
    median_volume = median(cut_cell_volumes)
    
    # Volume total des cut-cells
    total_volume = sum(cut_cell_volumes)

    # Volume moyen des cut-cells
    mean_volume = total_volume / num_cut_cells

    return num_cut_cells, percentage_cut_cells, percentage_uncut_cells, total_volume, mean_volume, min_volume, max_volume, std_dev_volume, median_volume
end

function calculate_angles(bary, cut_cells, nx, ny, centre_cercle)
    # Convertir les indices CartesianIndex en indices linéaires
    linear_indices = [LinearIndices((nx, ny))[i] for i in cut_cells]

    # Récupérer les valeurs correspondantes de bary
    bary_cut_cells = bary[linear_indices]

    # Calculer les angles
    angles = [Base.atan(bary[2] - centre_cercle[2], bary[1] - centre_cercle[1]) for bary in bary_cut_cells]

    # Convertir en degrés
    angles_deg = Base.rad2deg.(angles)

    return angles_deg
end