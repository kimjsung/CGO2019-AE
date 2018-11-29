#
#
#
import src.codes.helper_base    as helper_base

#
#   used variables
#   : gridsize_#, blocksize_#
#   : parameters (C, A, B)
#   : size_idx
#   : SLICE_SIZE_1_IDX
#   : dev_internal_offset_t2|v2
#   : stride_reg_x, stride_reg_y
#   : size_internal
#

def call_kernel(f, steps_tab, str_kernel_name, int_inner_group_idx,
                l_combined_parameters_C, l_combined_parameters_A, l_combined_parameters_B, l_combined_addr_internal,
                l_external_index, l_internal_index,
                opt_internal, opt_pre_computed):
    #
    #v
    #
    helper_base.codes_a_line(f, steps_tab, str_kernel_name, 0)
    helper_base.codes_a_line(f,         0, "<<<gridsize_" + str(int_inner_group_idx) + ", blocksize_" + str(int_inner_group_idx) + ">>>", 0)
    helper_base.codes_a_line(f,         0, "(", 0)

    #   parameters: Output Tensor (C)
    par_count = 0
    for each_par in l_combined_parameters_C[int_inner_group_idx - 1]:
        if opt_pre_computed == -1:
            if "range"  in each_par[0]:
                continue
            if "base"   in each_par[0]:
                continue
            if "offset" in each_par[0]:
                continue
            if "addr"   in each_par[0]:
                continue

            #
            if par_count == 0:
                helper_base.codes_a_line(f, 0,        each_par[0], 0)
            else:
                helper_base.codes_a_line(f, 0, ", " + each_par[0], 0)
            par_count += 1
        else:
            #
            if par_count == 0:
                helper_base.codes_a_line(f, 0,        each_par[0], 0)
            else:
                helper_base.codes_a_line(f, 0, ", " + each_par[0], 0)
            par_count += 1

    #   parameters: Input Tensor (A)
    for each_par in l_combined_parameters_A[int_inner_group_idx - 1]:
        if opt_pre_computed == -1:
            if "range"  in each_par[0]:
                continue
            if "base"   in each_par[0]:
                continue
            if "offset" in each_par[0]:
                continue
            if "addr"   in each_par[0]:
                continue
            #
            helper_base.codes_a_line(f, 0, ", " + each_par[0], 0)
        else:
            #
            helper_base.codes_a_line(f, 0, ", " + each_par[0], 0)

    #   parameters: Input Tensor (B)
    for each_par in l_combined_parameters_B[int_inner_group_idx - 1]:
        if opt_pre_computed == -1:
            if "range"  in each_par[0]:
                continue
            if "base"   in each_par[0]:
                continue
            if "offset" in each_par[0]:
                continue
            if "addr"   in each_par[0]:
                continue
            #
            helper_base.codes_a_line(f, 0, ", " + each_par[0], 0)
        else:
            #
            helper_base.codes_a_line(f, 0, ", " + each_par[0], 0)

    #   w/o pre-computed arrays 
    if opt_pre_computed == -1:
        #   external
        for each_idx in l_external_index:
            helper_base.codes_a_line(f, 0, ", size_" + each_idx, 0)
        #   internal
        for each_idx in l_internal_index:
            helper_base.codes_a_line(f, 0, ", size_" + each_idx, 0)
        #   # of tiles for external
        for each_idx in l_external_index:
            helper_base.codes_a_line(f, 0, ", CEIL(size_" + each_idx + ", SIZE_SLICE_1_" + each_idx.capitalize() + ")", 0)

    #   strides for inputs (temporal)
    for each_par in l_combined_addr_internal[int_inner_group_idx - 1]:
        helper_base.codes_a_line(f, 0, ", " + each_par[0], 0)
        helper_base.codes_a_line(f, 0, ", " + each_par[2], 0)

    #
    if len(l_internal_index) > 1 and opt_internal == 1:
        #
        #
        #
        helper_base.codes_a_line(f, 0, ", dev_internal_offset_t2", 0)
        helper_base.codes_a_line(f, 0, ", dev_internal_offset_v2", 0)


    #   strides for output
    helper_base.codes_a_line(f, 0, ", stride_reg_x_" + str(int_inner_group_idx), 0)
    helper_base.codes_a_line(f, 0, ", stride_reg_y_" + str(int_inner_group_idx), 0)

    #   size_internal
    helper_base.codes_a_line(f, 0, ", size_internal", 0)

    #
    helper_base.codes_a_line(f,         0, ");", 1)