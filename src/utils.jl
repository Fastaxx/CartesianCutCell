function volume_integrated_p_norm(e1, e2, V, p)
    total_volume = sum(V)
    abs_error = abs.(e1 - e2) .^p
    integral = sum(abs_error .* V)/ total_volume
    return (integral)^(1/p)
end

function get_condition_number_L2_svd(A)
    # Get the maximum singular value
    singular_values_list = svds(A, nsv=6)[1].S
    max_singular_value = maximum(singular_values_list)

    # Get the minimum singular value
    min_singular_value = minimum(singular_values_list)

    return max_singular_value/min_singular_value
end

