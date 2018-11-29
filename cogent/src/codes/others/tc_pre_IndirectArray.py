import src.generators.tc_helper     as tc_helper

#
def tc_gen_code_driver_PreComputedArray(f, kernel_number,   l_interface_info,   l_var_outputs_helpers,  l_var_input_left,   l_var_input_right,  l_var_tensor_block, 
                                                            l_input_tensors,    l_host_dynamic,         l_external_idx,     l_t3_idx,           l_t3_mapping_reg,   l_t3_mapping_tb):
    #
    tc_gen_code_driver_PreComputedArray_Head(f, kernel_number, l_interface_info, l_var_outputs_helpers, l_var_input_left, l_var_input_right, l_var_tensor_block)

    #
    tc_gen_code_driver_PreComputed_Init(f, kernel_number, l_input_tensors, l_host_dynamic, l_external_idx)
    
    #
    tc_gen_code_driver_PreComputed_wo_TB(f, kernel_number, l_input_tensors, l_t3_idx, l_external_idx, l_t3_mapping_reg)

    #
    tc_gen_code_driver_PreComputed_w_TB(f, kernel_number, l_input_tensors, l_t3_idx, l_t3_mapping_tb, l_external_idx, l_t3_mapping_reg, l_host_dynamic)

    #
    f.write("}\n")

#
def tc_gen_code_pre_IndirectArray(f,    l_t3_idx,       l_t3_mapping_tb,    l_input_tensors,    l_t3_mapping_reg,
                                        l_external_idx, l_internal_idx,     l_host_dynamic,     possible_diff,      idx_kernel):

    #
    tc_gen_code_pre_IndirectArray_Head(f, idx_kernel)

    #
    tc_gen_code_pre_IndirectArray_Init(f, l_input_tensors, l_host_dynamic, l_external_idx, possible_diff, idx_kernel)

    #
    tc_gen_code_pre_IndirectArray_wo_TB(f, l_input_tensors, l_t3_idx, l_external_idx, l_t3_mapping_reg, idx_kernel)

    #
    #
    #
    if possible_diff == 1:
        f.write("\n")
        tc_gen_code_pre_IndirectArray_wo_TB_twoDiff(f, l_input_tensors, l_t3_idx, l_external_idx, l_t3_mapping_reg, idx_kernel)

    #
    tc_gen_code_pre_IndirectArray_w_TB(f, l_input_tensors, l_t3_idx, l_t3_mapping_tb, l_external_idx, l_t3_mapping_reg, l_host_dynamic, idx_kernel)

    #
    #   tc_gen_code_pre_IndirectArray_Internal_Indices()
    #
    if len(l_internal_idx) > 1:
        tc_gen_code_pre_IndirectArray_Internal_Indices(f, l_internal_idx)
    else:
        f.write("\t// Do not need Indirect Arrays for Internal Indices\n")

    f.write("}\n")      # End of "pre_IndirectArray()"

#
def tc_gen_code_pre_IndirectArray_Internal_Indices(f, l_internal_idx):
    #
    f.write("\t// tc_gen_code_pre_IndirectArray_Internal_Indices()\n")
    #
    idx_count                   = 0
    str_size_internal_indices   = ""
    for each_idx in l_internal_idx:
        if idx_count == 0:
            str_size_internal_indices = "SIZE_IDX_" + each_idx.capitalize()
        else:
            str_size_internal_indices = str_size_internal_indices + " * SIZE_IDX_" + each_idx.capitalize()
        idx_count = idx_count + 1

    f.write("\n")
    f.write("\t// For Internal Indices,\n")
    f.write("\th_internal_t2_1_offset = (int*)malloc(sizeof(int) * " + str_size_internal_indices + ");\n")
    f.write("\th_internal_v2_1_offset = (int*)malloc(sizeof(int) * " + str_size_internal_indices + ");\n")
    f.write("\n")

    #
    str_idx         = ""
    str_stride_t2   = ""
    str_stride_v2   = ""
    idx_count       = 0
    for each_idx in l_internal_idx:
        #   For "For-Statement",
        f.write("\tfor (int idx_" + each_idx + " = 0; idx_" + each_idx + " < SIZE_IDX_" + each_idx.capitalize() + "; idx_" + each_idx + "++)\n")

        #   For a linearized index,
        if idx_count == 0:
            str_idx         = "idx_" + each_idx
            str_stride_t2   = "(" + "idx_" + each_idx + " * STR_SD2_T2_1_" + each_idx.capitalize() + ")"
            str_stride_v2   = "(" + "idx_" + each_idx + " * STR_SD2_V2_1_" + each_idx.capitalize() + ")"
        else:
            str_idx         = "idx_" + each_idx + " + (" + str_idx + ") * SIZE_IDX_" + each_idx.capitalize()
            str_stride_t2   = str_stride_t2 + " + " + "(" + "idx_" + each_idx + " * STR_SD2_T2_1_" + each_idx.capitalize() + ")"
            str_stride_v2   = str_stride_v2 + " + " + "(" + "idx_" + each_idx + " * STR_SD2_V2_1_" + each_idx.capitalize() + ")"
        idx_count = idx_count + 1

    #
    f.write("\t{\n")

    #
    f.write("\t\th_internal_t2_1_offset[" + str_idx + "] = " + str_stride_t2 + ";\n")
    f.write("\t\th_internal_v2_1_offset[" + str_idx + "] = " + str_stride_v2 + ";\n")

    #
    f.write("\t}\n")

#
def tc_gen_code_driver_PreComputed_Init(f, idx_kernel, l_input_tensors, l_host_dynamic, l_external_idx):
    #
    f.write("\t// tc_gen_code_pre_IndirectArray_Init()\n")

    #
    str_prev_idx = "1"
    for each_idx in l_external_idx:
        f.write("\tint str_sd2_t3_" + each_idx + " = " + str_prev_idx + ";\n")
        str_prev_idx = "str_sd2_t3_" + each_idx + " * size_" + each_idx

    #
    f.write("\n")
    f.write("\thost_t3_output_base_" + str(idx_kernel) + " = (int*)malloc(sizeof(int) * (num_thread_blocks_kernel_" + str(idx_kernel) + "));\n")
    l_host_dynamic.append("host_t3_output_base_" + str(idx_kernel))

    #
    for sd2_func in l_input_tensors:
        #
        #   LEFT
        #
        f.write("\thost_" + sd2_func[0][0] + "_addr = (int*)malloc(sizeof(int) * ")
        l_host_dynamic.append("host_" + sd2_func[0][0] + "_addr")

        #
        idx_count = 0
        for func_idx in sd2_func[0][1]:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                f.write(" * ")
            idx_count = idx_count + 1

        f.write("num_thread_blocks_kernel_" + str(idx_kernel))
        f.write(");\n")

        #
        #   RIGHT
        #
        f.write("\thost_" + sd2_func[1][0] + "_addr = (int*)malloc(sizeof(int) * ")
        l_host_dynamic.append("host_" + sd2_func[1][0] + "_addr")

        #
        idx_count = 0
        for func_idx in sd2_func[1][1]:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                f.write(" * ")
            idx_count = idx_count + 1
        f.write("num_thread_blocks_kernel_" + str(idx_kernel))
        f.write(");\n")

#
def tc_gen_code_pre_IndirectArray_Init(f, l_input_tensors, l_host_dynamic, l_external_idx, possible_diff, idx_kernel):
    #
    f.write("\t// tc_gen_code_pre_IndirectArray_Init()\n")
    f.write("\th_t3_output_base_" + str(idx_kernel) + "             = (int*)malloc(sizeof(int) * (n_blks_" + str(idx_kernel) + "));\n")
    l_host_dynamic.append("h_t3_output_base_" + str(idx_kernel))

    #
    if possible_diff == 1:
        f.write("\th_t3_output_base_full_" + str(idx_kernel) + "        = (int*)malloc(sizeof(int) * (num_blk_full_" + str(idx_kernel) + "));\n")
        f.write("\th_t3_output_base_non_full_" + str(idx_kernel) + "    = (int*)malloc(sizeof(int) * (num_blk_non_full_" + str(idx_kernel) + "));\n")
        l_host_dynamic.append("h_t3_output_base_full_" + str(idx_kernel))
        l_host_dynamic.append("h_t3_output_base_non_full_" + str(idx_kernel))
        f.write("\n")

    #
    for sd2_func in l_input_tensors:
        #
        #   LEFT
        #
        f.write("\th_" + sd2_func[0][0] + "_addr = (int*)malloc(sizeof(int) * ")
        l_host_dynamic.append("h_" + sd2_func[0][0] + "_addr")

        #
        idx_count = 0
        for func_idx in sd2_func[0][1]:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                f.write(" * ")
            idx_count = idx_count + 1

        f.write("n_blks_" + str(idx_kernel))
        f.write(");\n")

        if possible_diff == 1:
            f.write("\th_" + sd2_func[0][0] + "_addr_full = (int*)malloc(sizeof(int) * ")
            l_host_dynamic.append("h_" + sd2_func[0][0] + "_addr_full")

            #
            idx_count = 0
            for func_idx in sd2_func[0][1]:
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                    f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                    f.write(" * ")
                idx_count = idx_count + 1

            f.write("num_blk_full_" + str(idx_kernel))
            f.write(");\n")

            #
            f.write("\th_" + sd2_func[0][0] + "_addr_non_full = (int*)malloc(sizeof(int) * ")
            l_host_dynamic.append("h_" + sd2_func[0][0] + "_addr_non_full")

            #
            idx_count = 0
            for func_idx in sd2_func[0][1]:
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                    f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                    f.write(" * ")
                idx_count = idx_count + 1

            f.write("num_blk_non_full_" + str(idx_kernel))
            f.write(");\n")
            f.write("\n")

        #
        #   RIGHT
        #
        f.write("\th_" + sd2_func[1][0] + "_addr = (int*)malloc(sizeof(int) * ")
        l_host_dynamic.append("h_" + sd2_func[1][0] + "_addr")

        #
        idx_count = 0
        for func_idx in sd2_func[1][1]:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                f.write(" * ")
            idx_count = idx_count + 1
        f.write("n_blks_" + str(idx_kernel))
        f.write(");\n")

        if possible_diff == 1:
            #
            f.write("\th_" + sd2_func[1][0] + "_addr_full = (int*)malloc(sizeof(int) * ")
            l_host_dynamic.append("h_" + sd2_func[1][0] + "_addr_full")

            #
            idx_count = 0
            for func_idx in sd2_func[1][1]:
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                    f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                    f.write(" * ")
                idx_count = idx_count + 1
            f.write("num_blk_full_" + str(idx_kernel))
            f.write(");\n")

            #
            f.write("\th_" + sd2_func[1][0] + "_addr_non_full = (int*)malloc(sizeof(int) * ")
            l_host_dynamic.append("h_" + sd2_func[1][0] + "_addr_non_full")

            #
            idx_count = 0
            for func_idx in sd2_func[1][1]:
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, func_idx) != -1:
                    f.write("SIZE_SLICE_" + str(idx_kernel) + "_" + func_idx.capitalize())
                    f.write(" * ")
                idx_count = idx_count + 1
            f.write("num_blk_non_full_" + str(idx_kernel))
            f.write(");\n")
        f.write("\n")

#
#
#
def tc_gen_code_driver_PreComputed_wo_TB(f, idx_kernel, l_input_tensors, l_t3_idx, l_external_idx, l_t3_mapping_reg):
    #
    #
    #
    f.write("\t// tc_gen_code_pre_IndirectArray_wo_TB()\n")
    f.write("\t// For Each Thread Block\n")
    f.write("\tfor (int i = 0; i < num_thread_blocks_kernel_" + str(idx_kernel) + "; i++)\n")
    f.write("\t{\n")

    # Getting indices
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("\t\tint blk_idx_" + t3_idx + " = host_t3_block_index_" + str(idx_kernel) + "[i * NUM_INDEX + " + str(idx_count) + "];\n")
        idx_count = idx_count + 1
    f.write("\n")

    # Getting t3_output_base
    f.write("\t\t// Calculating t3\n")
    f.write("\t\thost_t3_output_base_" + str(idx_kernel) + "[i] = ")
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("blk_idx_" + t3_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize() + " * str_sd2_t3_" + t3_idx)
        if idx_count == len(l_t3_idx) - 1:
            f.write(";\n")
        else:
            f.write(" + ")
        idx_count = idx_count + 1
    f.write("\n")

    # Calculating Actual Address for Each Block based on t3's offset.
    # For t2,
    sd2_idx             = 1
    base_h_t2_1_addr    = ""
    base_l_offset       = ""
    actual_h_t2_1_addr  = ""

    # Per A Single Tensor Contraction
    for sd2_func in l_input_tensors:
        f.write("\t\t// Calculating Actual Address for " + sd2_func[0][0] + " and " + sd2_func[1][0] +  "\n")
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""

        #print ("reg-tile: ", l_t3_mapping_reg)
        rev_left = list(reversed(sd2_func[0][1]))

        # For-Statement
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) != -1:
                    base_l_offset   = "idx_" + left_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) == -1:
                    base_l_offset   = "idx_" + left_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:    # external index
                str_idx             = "idx_" + left_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * size_" + left_idx
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * size_" + left_idx
            idx_count = idx_count + 1

        # [a, e, b, f]  >>  [f, b, e, a]
        # a + (e + (b + (f) * SIZE_B * SIZE_E) * SIZE_A
        # a + ((b) * SIZE_E) * SIZE_A
        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\thost_" + sd2_func[0][0] + "_addr[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
        f.write("\n")

        #
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""

        # RIGHT
        rev_right = list(reversed(sd2_func[1][1]))

        # For-Statement
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) != -1:
                    base_l_offset   = "idx_" + right_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) == -1:
                    base_l_offset   = "idx_" + right_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:    # external index
                str_idx             = "idx_" + right_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * size_" + right_idx
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * size_" + right_idx
            idx_count = idx_count + 1

        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\thost_" + sd2_func[1][0] + "_addr[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
    f.write("\t}\n")

#
def tc_gen_code_pre_IndirectArray_wo_TB(f, l_input_tensors, l_t3_idx, l_external_idx, l_t3_mapping_reg, idx_kernel):
    #
    #
    #
    f.write("\t// tc_gen_code_pre_IndirectArray_wo_TB()\n")
    f.write("\t// For Each Thread Block\n")
    f.write("\tfor (int i = 0; i < n_blks_" + str(idx_kernel) + "; i++)\n")
    f.write("\t{\n")

    # Getting indices
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("\t\tint blk_idx_" + t3_idx + " = h_t3_blk_idx_" + str(idx_kernel) + "[i * NUM_INDEX + " + str(idx_count) + "];\n")
        idx_count = idx_count + 1
    f.write("\n")

    # Getting t3_output_base
    f.write("\t\t// Calculating t3\n")
    f.write("\t\th_t3_output_base_" + str(idx_kernel) + "[i] = ")
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("blk_idx_" + t3_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize() + " * STR_SD2_T3_" + t3_idx.capitalize())
        if idx_count == len(l_t3_idx) - 1:
            f.write(";\n")
        else:
            f.write(" + ")
        idx_count = idx_count + 1
    f.write("\n")

    # Calculating Actual Address for Each Block based on t3's offset.
    # For t2,
    sd2_idx             = 1
    base_h_t2_1_addr    = ""
    base_l_offset       = ""
    actual_h_t2_1_addr  = ""

    # Per A Single Tensor Contraction
    for sd2_func in l_input_tensors:
        f.write("\t\t// Calculating Actual Address for " + sd2_func[0][0] + " and " + sd2_func[1][0] +  "\n")
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""

        #print ("reg-tile: ", l_t3_mapping_reg)
        rev_left = list(reversed(sd2_func[0][1]))

        # For-Statement
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) != -1:
                    base_l_offset   = "idx_" + left_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) == -1:
                    base_l_offset   = "idx_" + left_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:    # external index
                str_idx             = "idx_" + left_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
            idx_count = idx_count + 1


        # [a, e, b, f]  >>  [f, b, e, a]
        # a + (e + (b + (f) * SIZE_B * SIZE_E) * SIZE_A
        # a + ((b) * SIZE_E) * SIZE_A
        '''
        idx_count = 0
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1: # if left_idx is an external index,
                str_idx         = "idx_" + left_idx
                f.write("\t\tfor (int " + str_idx + " = 0; " + str_idx + " < SIZE_SLICE_" + left_idx.capitalize() + "; " + str_idx + "++)\n")
                #
                if idx_count == 0:
                    base_l_offset   = str_idx
                    base_h_addr     = "SIZE_SLICE_" + left_idx.capitalize()
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + left_idx.capitalize() + " + " + str_idx
                else:
                    base_l_offset   = str_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + left_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + left_idx.capitalize()
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + left_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
                idx_count = idx_count + 1
            else:
                actual_h_addr = "(" + actual_h_addr +  ") * SIZE_IDX_" + left_idx.capitalize()
        '''
        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\th_" + sd2_func[0][0] + "_addr[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
        f.write("\n")

        #
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""

        # RIGHT
        rev_right = list(reversed(sd2_func[1][1]))

        # For-Statement
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) != -1:
                    base_l_offset   = "idx_" + right_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) == -1:
                    base_l_offset   = "idx_" + right_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:    # external index
                str_idx             = "idx_" + right_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
            idx_count = idx_count + 1

        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\th_" + sd2_func[1][0] + "_addr[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
    f.write("\t}\n")

#
#
#
def tc_gen_code_pre_IndirectArray_wo_TB_twoDiff(f, l_input_tensors, l_t3_idx, l_external_idx, l_t3_mapping_reg, idx_kernel):
    #
    #   non_full
    #
    f.write("\t// For Each Thread Block\n")
    f.write("\tfor (int i = 0; i < num_blk_non_full_" + str(idx_kernel) + "; i++)\n")
    f.write("\t{\n")

    #   base_index
    f.write("\t\tint base = t3_blk_idx_non_full_" + str(idx_kernel) + "[i];\n")

    # Getting indices
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("\t\tint blk_idx_" + t3_idx + " = h_t3_blk_idx_" + str(idx_kernel) + "[base + " + str(idx_count) + "];\n")
        idx_count = idx_count + 1
    f.write("\n")

    # Getting t3_output_base
    f.write("\t\t// Calculating t3\n")
    f.write("\t\th_t3_output_base_non_full_" + str(idx_kernel) + "[i] = ")
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("blk_idx_" + t3_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize() + " * STR_SD2_T3_" + t3_idx.capitalize())
        if idx_count == len(l_t3_idx) - 1:
            f.write(";\n")
        else:
            f.write(" + ")
        idx_count = idx_count + 1
    f.write("\n")

    # Calculating Actual Address for Each Block based on t3's offset.
    # For t2,
    sd2_idx             = 1
    base_h_t2_1_addr    = ""
    base_l_offset       = ""
    actual_h_t2_1_addr  = ""

    # Per A Single Tensor Contraction
    for sd2_func in l_input_tensors:
        f.write("\t\t// Calculating Actual Address for " + sd2_func[0][0] + " and " + sd2_func[1][0] +  "\n")
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""
        '''
        # LEFT
        rev_left = list(reversed(sd2_func[0][1]))
        for left_idx in rev_left:
            #
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                str_idx = "idx_" + left_idx
                f.write("\t\tfor (int " + str_idx + " = 0; " + str_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; " + str_idx + "++)\n")
                base_h_addr = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                if count == 0:
                    actual_h_addr = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx
                    base_l_offset = str_idx
                else:
                    actual_h_addr = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " +str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
                    base_l_offset = str_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                count = count + 1
            else:
                actual_h_addr = "(" +  actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
        '''
        rev_left = list(reversed(sd2_func[0][1]))
        # For-Statement
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) != -1:
                    base_l_offset   = "idx_" + left_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) == -1:
                    base_l_offset   = "idx_" + left_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:    # external index
                str_idx             = "idx_" + left_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
            idx_count = idx_count + 1    

        #
        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\th_" + sd2_func[0][0] + "_addr_non_full[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
        f.write("\n")

        #
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""
        '''
        # RIGHT
        rev_right = list(reversed(sd2_func[1][1]))
        for right_idx in rev_right:
            #
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                str_idx = "idx_" + right_idx
                f.write("\t\tfor (int " + str_idx + " = 0; " + str_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; " + str_idx + "++)\n")
                base_h_addr = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                if count == 0:
                    actual_h_addr = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx
                    base_l_offset = str_idx
                else:
                    actual_h_addr = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
                    base_l_offset =  str_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                count = count + 1
            else:
                actual_h_addr = "(" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
        '''
        rev_right = list(reversed(sd2_func[1][1]))
        # For-Statement
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) != -1:
                    base_l_offset   = "idx_" + right_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) == -1:
                    base_l_offset   = "idx_" + right_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:    # external index
                str_idx             = "idx_" + right_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
            idx_count = idx_count + 1

        #
        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\th_" + sd2_func[1][0] + "_addr_non_full[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
    f.write("\t}\n")

    #
    #   full
    #
    f.write("\n")
    f.write("\t// For Each Thread Block\n")
    f.write("\tfor (int i = 0; i < num_blk_full_" + str(idx_kernel) + "; i++)\n")
    f.write("\t{\n")

    #   base_index
    f.write("\t\tint base = t3_blk_idx_full_" + str(idx_kernel) + "[i];\n")

    # Getting indices
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("\t\tint blk_idx_" + t3_idx + " = h_t3_blk_idx_" + str(idx_kernel) + "[base + " + str(idx_count) + "];\n")
        idx_count = idx_count + 1
    f.write("\n")

    # Getting t3_output_base
    f.write("\t\t// Calculating t3\n")
    f.write("\t\th_t3_output_base_full_" + str(idx_kernel) + "[i] = ")
    idx_count = 0
    for t3_idx in l_t3_idx:
        f.write("blk_idx_" + t3_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize() + " * STR_SD2_T3_" + t3_idx.capitalize())
        if idx_count == len(l_t3_idx) - 1:
            f.write(";\n")
        else:
            f.write(" + ")
        idx_count = idx_count + 1
    f.write("\n")

    # Calculating Actual Address for Each Block based on t3's offset.
    # For t2,
    sd2_idx             = 1
    base_h_t2_1_addr    = ""
    base_l_offset       = ""
    actual_h_t2_1_addr  = ""

    # Per A Single Tensor Contraction
    for sd2_func in l_input_tensors:
        f.write("\t\t// Calculating Actual Address for " + sd2_func[0][0] + " and " + sd2_func[1][0] +  "\n")
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""

        rev_left = list(reversed(sd2_func[0][1]))
        # For-Statement
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) != -1:
                    base_l_offset   = "idx_" + left_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) == -1:
                    base_l_offset   = "idx_" + left_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    f.write("\t\tfor (int idx_" + left_idx + " = 0; idx_" + left_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + "; idx_" + left_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:    # external index
                str_idx             = "idx_" + left_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + left_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * SIZE_IDX_" + left_idx.capitalize()
            idx_count = idx_count + 1    

        #
        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\th_" + sd2_func[0][0] + "_addr_full[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
        f.write("\n")

        #
        count               = 0
        base_h_addr         = ""
        actual_h_addr       = ""
        base_l_offset       = ""

        rev_right = list(reversed(sd2_func[1][1]))
        # For-Statement
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) != -1:
                    base_l_offset   = "idx_" + right_idx
                    base_h_addr     = "SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) == -1:
                    base_l_offset   = "idx_" + right_idx + " + (" + base_l_offset + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    base_h_addr     = base_h_addr + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    f.write("\t\tfor (int idx_" + right_idx + " = 0; idx_" + right_idx + " < SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + "; idx_" + right_idx + "++)\n")

        # [f, d, e, c] >> [c, e, d, f]
        # f + (d + (e + (c) * SIZE_E) * SIZE_D) * SIZE_F
        # (d + ((c) * SIZE_E) * SIZE_D) * SIZE_F
        idx_count = 0
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:    # external index
                str_idx             = "idx_" + right_idx
                if count == 0:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx
                else:
                    actual_h_addr   = "blk_idx_" + right_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize() + " + " + str_idx + " + (" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
                count = count + 1
            else:                                                                   # internal index
                if idx_count != 0:
                    actual_h_addr = "(" + actual_h_addr + ") * SIZE_IDX_" + right_idx.capitalize()
            idx_count = idx_count + 1

        #
        f.write("\t\t{\n")
        f.write("\t\t\tint l_offset = " + base_l_offset + ";\n")
        f.write("\t\t\th_" + sd2_func[1][0] + "_addr_full[l_offset + i * " + base_h_addr + "] = " + actual_h_addr + ";\n")
        f.write("\t\t}\n")
    f.write("\t}\n")

#
def tc_gen_code_driver_PreComputed_w_TB(f, idx_kernel, l_input_tensors, l_t3_idx, l_t3_mapping_tb, l_external_idx, l_t3_mapping_reg, l_host_dynamic):
    #
    #
    #
    f.write("\n")
    f.write("\t// tc_gen_code_pre_IndirectArray_w_TB()\n")
    f.write("\t// Within a Thread Block\n")
    f.write("\thost_t3_output_offset_" + str(idx_kernel) + " = (int*)malloc(sizeof(int) * (SIZE_TB_" + str(idx_kernel) + "_X * SIZE_TB_" + str(idx_kernel) + "_Y));\n")
    l_host_dynamic.append("host_t3_output_offset_" + str(idx_kernel) + "")

    for sd2_func in l_input_tensors:
        f.write("\thost_" + sd2_func[0][0] + "_offset = (int*)malloc(sizeof(int) * (SIZE_TB_" + str(idx_kernel) + "_X * SIZE_TB_" + str(idx_kernel) + "_Y));\n")
        l_host_dynamic.append("host_" + sd2_func[0][0] + "_offset")
        f.write("\thost_" + sd2_func[1][0] + "_offset = (int*)malloc(sizeof(int) * (SIZE_TB_" + str(idx_kernel) + "_X * SIZE_TB_" + str(idx_kernel) + "_Y));\n")
        l_host_dynamic.append("host_" + sd2_func[1][0] + "_offset")

    f.write("\n")

    #
    l_t3_mapping            = list(reversed(l_t3_mapping_tb))
    l_t3_mapping_idx_off    = ""
    l_t3_mapping_idx_addr   = ""
    l_t3_mapping_t2_1       = ""
    l_t3_mapping_v2_1       = ""

    #
    #print ("l_t3_mapping: ", l_t3_mapping)

    #
    idx_count = 0
    for mapping in l_t3_mapping:
        f.write("\tfor (int idx_" + mapping + " = 0; idx_" + mapping + " < SIZE_SLICE_" + str(idx_kernel) + "_" + mapping.capitalize() + "; idx_" + mapping + "++)\n")
        if idx_count == 0:
            l_t3_mapping_idx_off    = "idx_" + mapping
            l_t3_mapping_idx_addr   = "idx_" + mapping
        else:
            l_t3_mapping_idx_off    = "idx_" + mapping + " + (" + l_t3_mapping_idx_off  + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + mapping.capitalize()
            l_t3_mapping_idx_addr   = "idx_" + mapping + " + (" + l_t3_mapping_idx_addr + ") * size_"   + mapping
        idx_count = idx_count + 1

    #
    rev_l_t3_idx        = list(reversed(l_t3_idx))
    idx_count           = 0
    idx_actual_count    = 0
    for mapping in rev_l_t3_idx:
        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping, mapping) != -1:
            if idx_count == 0:
                l_t3_mapping_idx_addr = "idx_" + mapping
            else:
                l_t3_mapping_idx_addr = "idx_" + mapping + " + (" + l_t3_mapping_idx_addr + ") * size_" + mapping
            idx_count = idx_count + 1
        else:
            if idx_count == 0:
                l_t3_mapping_idx_addr = l_t3_mapping_idx_addr
            else:
                l_t3_mapping_idx_addr = "(" + l_t3_mapping_idx_addr + ") * size_" + mapping

    f.write("\t{\n")
    f.write("\t\tint t3_index_1 = " + l_t3_mapping_idx_off + ";\n")
    f.write("\n")
    f.write("\t\thost_t3_output_offset_" + str(idx_kernel) + "[t3_index_1] = " + l_t3_mapping_idx_addr + ";\n")

    #
    #   Offsets for Two Input Tensors--- t2 and v2.
    #
    for sd2_func in l_input_tensors:
        #
        #   LEFT
        #
        l_t3_mapping_left   = ""
        rev_left            = list(reversed(sd2_func[0][1]))
        count               = 0
        idx_count           = 0        
        #
        #
        #
        for left_idx in rev_left:
            # exclude external index for reg_tile
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, left_idx) == -1:
                # exclude internal index
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:    # external index
                    if count == 0:
                        l_t3_mapping_left   = "idx_" + left_idx
                    else:
                        l_t3_mapping_left   = "idx_" + left_idx + " + (" + l_t3_mapping_left + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    count = count + 1
                #   To-Do
                #else:                                                                   # internal index
                #    if idx_count != 0:
                #        l_t3_mapping_left = "(" + l_t3_mapping_left + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                #    idx_count = idx_count + 1

        #
        #   RIGHT
        #
        l_t3_mapping_right  = ""
        rev_right           = list(reversed(sd2_func[1][1]))
        count               = 0
        idx_count           = 0
        #
        #
        for right_idx in rev_right:
            # exclude external index for reg_tile
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg , right_idx) == -1:
                # exclude internal index
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:    # external index
                    if count == 0:
                        l_t3_mapping_right   = "idx_" + right_idx
                    else:
                        l_t3_mapping_right   = "idx_" + right_idx + " + (" + l_t3_mapping_right + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    count = count + 1
                #   To-Do
                #else:                                                                   # internal index
                #    if idx_count != 0:
                #        l_t3_mapping_right = "(" + l_t3_mapping_right + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                #    idx_count = idx_count + 1

        f.write("\t\thost_" + sd2_func[0][0] + "_offset[t3_index_1] = " + l_t3_mapping_left + ";\n")
        f.write("\t\thost_" + sd2_func[1][0] + "_offset[t3_index_1] = " + l_t3_mapping_right + ";\n")

    f.write("\t}\n")    # End of "For"

#
def tc_gen_code_pre_IndirectArray_w_TB(f, l_input_tensors, l_t3_idx, l_t3_mapping_tb, l_external_idx, l_t3_mapping_reg, l_host_dynamic, idx_kernel):
    #
    #
    #
    f.write("\n")
    f.write("\t// tc_gen_code_pre_IndirectArray_w_TB()\n")
    f.write("\t// Within a Thread Block\n")
    f.write("\th_t3_output_offset_" + str(idx_kernel) + " = (int*)malloc(sizeof(int) * (SIZE_TB_" + str(idx_kernel) + "_X * SIZE_TB_" + str(idx_kernel) + "_Y));\n")
    l_host_dynamic.append("h_t3_output_offset_" + str(idx_kernel) + "")

    for sd2_func in l_input_tensors:
        f.write("\th_" + sd2_func[0][0] + "_offset = (int*)malloc(sizeof(int) * (SIZE_TB_" + str(idx_kernel) + "_X * SIZE_TB_" + str(idx_kernel) + "_Y));\n")
        l_host_dynamic.append("h_" + sd2_func[0][0] + "_offset")
        f.write("\th_" + sd2_func[1][0] + "_offset = (int*)malloc(sizeof(int) * (SIZE_TB_" + str(idx_kernel) + "_X * SIZE_TB_" + str(idx_kernel) + "_Y));\n")
        l_host_dynamic.append("h_" + sd2_func[1][0] + "_offset")

    f.write("\n")

    #
    l_t3_mapping            = list(reversed(l_t3_mapping_tb))
    l_t3_mapping_idx_off    = ""
    l_t3_mapping_idx_addr   = ""
    l_t3_mapping_t2_1       = ""
    l_t3_mapping_v2_1       = ""

    #
    idx_count = 0
    for mapping in l_t3_mapping:
        f.write("\tfor (int idx_" + mapping + " = 0; idx_" + mapping + " < SIZE_SLICE_" + str(idx_kernel) + "_" + mapping.capitalize() + "; idx_" + mapping + "++)\n")
        if idx_count == 0:
            l_t3_mapping_idx_off    = "idx_" + mapping
            l_t3_mapping_idx_addr   = "idx_" + mapping
        else:
            l_t3_mapping_idx_off    = "idx_" + mapping + " + (" + l_t3_mapping_idx_off  + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + mapping.capitalize()
            l_t3_mapping_idx_addr   = "idx_" + mapping + " + (" + l_t3_mapping_idx_addr + ") * SIZE_IDX_"   + mapping.capitalize()
        idx_count = idx_count + 1

    #
    rev_l_t3_idx        = list(reversed(l_t3_idx))
    idx_count           = 0
    idx_actual_count    = 0
    for mapping in rev_l_t3_idx:
        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping, mapping) != -1:
            if idx_count == 0:
                l_t3_mapping_idx_addr = "idx_" + mapping
            else:
                l_t3_mapping_idx_addr = "idx_" + mapping + " + (" + l_t3_mapping_idx_addr + ") * SIZE_IDX_" + mapping.capitalize()
            idx_count = idx_count + 1
        else:
            if idx_count == 0:
                l_t3_mapping_idx_addr = l_t3_mapping_idx_addr
            else:
                l_t3_mapping_idx_addr = "(" + l_t3_mapping_idx_addr + ") * SIZE_IDX_" + mapping.capitalize()

    f.write("\t{\n")
    f.write("\t\tint t3_index_1 = " + l_t3_mapping_idx_off + ";\n")
    f.write("\n")
    f.write("\t\th_t3_output_offset_" + str(idx_kernel) + "[t3_index_1] = " + l_t3_mapping_idx_addr + ";\n")

    #
    #   Offsets for Two Input Tensors--- t2 and v2.
    #
    for sd2_func in l_input_tensors:
        #
        #   LEFT
        #
        l_t3_mapping_left   = ""
        rev_left            = list(reversed(sd2_func[0][1]))
        count               = 0
        idx_count           = 0
        #
        '''
        for left_idx in rev_left:
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping, left_idx) != -1:
                if idx_count == 0:
                    l_t3_mapping_left = "idx_" + left_idx
                else:
                    l_t3_mapping_left = "idx_" + left_idx + " + (" + l_t3_mapping_left + ") * SIZE_SLICE_" + left_idx.capitalize()
                idx_count = idx_count + 1
            else:
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:
                    l_t3_mapping_left = "(" + l_t3_mapping_left + ") * SIZE_SLICE_" + left_idx.capitalize()
        '''
        #
        for left_idx in rev_left:
            # exclude external index for reg_tile
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg , left_idx) == -1:
                # exclude internal index
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, left_idx) != -1:    # external index
                    if count == 0:
                        l_t3_mapping_left   = "idx_" + left_idx
                    else:
                        l_t3_mapping_left   = "idx_" + left_idx + " + (" + l_t3_mapping_left + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    count = count + 1
                else:                                                                   # internal index
                    if idx_count != 0:
                        l_t3_mapping_left = "(" + l_t3_mapping_left + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + left_idx.capitalize()
                    idx_count = idx_count + 1

        #
        #   RIGHT
        #
        l_t3_mapping_right  = ""
        rev_right           = list(reversed(sd2_func[1][1]))
        count               = 0
        idx_count           = 0
        #
        '''
        for right_idx in rev_right:
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping, right_idx) != -1:
                if idx_count == 0:
                    l_t3_mapping_right = "idx_" + right_idx
                else:
                    l_t3_mapping_right = "idx_" + right_idx + " + (" + l_t3_mapping_right + ") * SIZE_SLICE_" + right_idx.capitalize()
                idx_count = idx_count + 1
            else:
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:
                    if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, right_idx) == -1:
                        l_t3_mapping_right = "(" + l_t3_mapping_right + ") * SIZE_SLICE_" + right_idx.capitalize()
        '''
        #
        for right_idx in rev_right:
            # exclude external index for reg_tile
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg , right_idx) == -1:
                # exclude internal index
                if tc_helper.tc_gen_helper_find_1d(l_external_idx, right_idx) != -1:    # external index
                    if count == 0:
                        l_t3_mapping_right   = "idx_" + right_idx
                    else:
                        l_t3_mapping_right   = "idx_" + right_idx + " + (" + l_t3_mapping_right + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    count = count + 1
                else:                                                                   # internal index
                    if idx_count != 0:
                        l_t3_mapping_right = "(" + l_t3_mapping_right + ") * SIZE_SLICE_" + str(idx_kernel) + "_" + right_idx.capitalize()
                    idx_count = idx_count + 1

        f.write("\t\th_" + sd2_func[0][0] + "_offset[t3_index_1] = " + l_t3_mapping_left + ";\n")
        f.write("\t\th_" + sd2_func[1][0] + "_offset[t3_index_1] = " + l_t3_mapping_right + ";\n")

    f.write("\t}\n")    # End of "For"

#
def tc_gen_code_pre_IndirectArray_Head(f, idx_kernel):
    f.write("\n")
    f.write("// created by tc_gen_code_pre_IndirectArray_" + str(idx_kernel) + "()\n")
    f.write("void pre_IndirectArray_" + str(idx_kernel) + "()\n")
    f.write("{\n")
    f.write("\tprintf (\" >>> %s <<<\\n\", __func__);\n")
    f.write("\n")

#
def tc_gen_code_driver_PreComputedArray_Head(f, kernel_number, l_interface_info, l_var_outputs_helpers, l_var_input_left, l_var_input_right, l_var_tensor_block):
    f.write("\n")
    f.write("// created by tc_gen_code_pre_PreComputedArray_" + str(kernel_number) + "()\n")
    f.write("void pre_PreComputedArray_" + str(kernel_number) + "(")

    #
    #   Related Outputs-Helpers
    #
    idx_count = 0
    for each_var in l_var_outputs_helpers:
        if "host" in each_var[1]:
            if not "range" in each_var[1]:
                str_type = each_var[0]

                if not "index" in each_var[1]:
                    str_type = str_type + "&"
                                
                if idx_count == 0:
                    f.write(str_type + " " + each_var[1])
                else:
                    f.write(", " + str_type + " " + each_var[1])
                idx_count = idx_count + 1

    #
    #   
    #
    for each_var in l_var_tensor_block:
        f.write(", " + each_var[0] + " " + each_var[1])

    #
    #
    #
    for each_var in l_var_input_left:
        if "host" in each_var[1]:
            if "addr" in each_var[1]:
                f.write(", " + each_var[0] + "& " + each_var[1])
            if "offset" in each_var[1]:
                f.write(", " + each_var[0] + "& " + each_var[1])

    for each_var in l_var_input_right:
        if "host" in each_var[1]:
            if "addr" in each_var[1]:
                f.write(", " + each_var[0] + "& " + each_var[1])
            if "offset" in each_var[1]:
                f.write(", " + each_var[0] + "& " + each_var[1])

    #
    for each_idx in l_interface_info[0][0]:
        f.write(", int size_" + each_idx)

    f.write(")\n")
    f.write("{\n")
    f.write("\tprintf (\" >>> %s <<<\\n\", __func__);\n")
    f.write("\n")

