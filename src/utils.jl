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

