# Utils.jl
## Overview
This Julia file contains functions to create various operations : Statistical calculations, norm, Condition number.

## Functions Provided

`volume_integrated_p_norm` : This function calculates the p-norm integrated over the volume of a given difference. This function is useful for quantifying the error between two numerical solutions, for example.

`get_condition_number_L2_svd`: The `get_condition_number_L2_svd` function in Julia calculates the conditioning number L2 of an A matrix using singular value decomposition (SVD).