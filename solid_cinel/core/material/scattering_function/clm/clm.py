from numba import cuda
if cuda.is_available():
    from .clm_cpu import get_scatfunc_pdos_row_cpu, get_scatfunc_pdos_cpu, scatfunc_values_alpha_vec_cpu
    get_scatfunc_pdos = get_scatfunc_pdos_cpu
    get_scatfunc_pdos_row = get_scatfunc_pdos_row_cpu
    scatfunc_values_alpha_vec = scatfunc_values_alpha_vec_cpu
else:
    from .clm_cpu import get_scatfunc_pdos_row_cpu, get_scatfunc_pdos_cpu, scatfunc_values_alpha_vec_cpu
    get_scatfunc_pdos = get_scatfunc_pdos_cpu
    get_scatfunc_pdos_row = get_scatfunc_pdos_row_cpu
    scatfunc_values_alpha_vec = scatfunc_values_alpha_vec_cpu