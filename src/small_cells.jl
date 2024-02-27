using SparseArrays

function clip_volume_moments(V, dx, dy, epsilon)
    clipped_V_values = similar(V.nzval)
    for i in 1:length(V.nzval)
        if V.nzval[i] < epsilon
            clipped_V_values[i] = 0.0
        elseif V.nzval[i] > dx * dy - epsilon
            clipped_V_values[i] = dx * dy
        else
            clipped_V_values[i] = V.nzval[i]
        end
    end
    clipped_V = sparse(V.rowval, V.colptr, clipped_V_values)
    return clipped_V
end

function clip_surface_moments(A_alpha, delta_alpha, epsilon)
    clipped_A_values = similar(A_alpha.nzval)
    for i in 1:length(A_alpha.nzval)
        if A_alpha.nzval[i] < sqrt(epsilon)
            clipped_A_values[i] = 0.0
        elseif A_alpha.nzval[i] > delta_alpha - sqrt(epsilon)
            clipped_A_values[i] = delta_alpha
        else
            clipped_A_values[i] = A_alpha.nzval[i]
        end
    end
    clipped_A = sparse(A_alpha.rowval, A_alpha.colptr, clipped_A_values)
    return clipped_A
end