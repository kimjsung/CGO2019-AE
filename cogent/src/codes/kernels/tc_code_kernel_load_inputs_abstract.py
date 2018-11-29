import src.generators.tc_helper     as tc_helper

#
#   1. The FVI is an internal or an external index?
#       1.1. Internal
#           : TB_X should load elements along the internal index.
#           : TB_Y should load elements along the external index.
#       1.2. External
#           : TB_X should load elements along the external index.
#           : TB_Y should load elements along the internal index.
#
#   2. For 1.1. case, TB_X -(loads)-> K && TB_Y -(loads)-> E, where E \ REG
#       2.1. |TB_X| = |K|
#       2.2. |TB_X| > |K|
#       2.3. |TB_X| < |K|
#       2.4. |TB_Y| = |E|
#       2.5. |TB_Y| > |E|
#       2.6. |TB_Y| < |E|
#
#   3. For 1.2. case, TB_X -(loads)-> E && TB_Y -(loads)-> K, where E \ REG
#       3.1. |TB_X| = |E|
#       3.2. |TB_X| > |E|
#       3.3. |TB_X| < |E|
#       3.4. |TB_Y| = |K|
#       3.5. |TB_Y| > |K|
#       3.6. |TB_Y| < |K|
#

#
#
#
def tc_gen_code_Kernel_Load_Inputs_Abstracts(f, num_code_tabs, 
                                                #
                                                info_input_tensor,
                                                #   options
                                                opt_load_ext_int, 
                                                opt_smem_ab,        opt_input_ab,
                                                opt_gen_ext,        opt_gen_int,
                                                opt_axis_reg,
                                                opt_internal,
                                                #   lists
                                                l_tile_sizes,
                                                l_internal_idx,
                                                l_mapping_tb,
                                                l_mapping_reg,
                                                #   sizes
                                                size_ext_tb,        size_ext_reg,
                                                size_smem_k,        size_tb_x,              size_tb_y,
                                                idx_kernel):
    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// tc_gen_code_Kernel_Load_Inputs_Abstracts()", 1)

    #
    #   >>> Base Form <<<
    #   (Optional: Not Matched Indices: Temporal Indices)
    #   if ([1] idx_a < rng_c2 && [2] threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)  // boundary: external-smem, internal
    #   if ([3] threadIdx.x < 8 && [4] threadIdx.y < #)                                     // boundary: |TB| and |K|
    #   {
    #       for (int ll = 0; ll < [5] rng_c1; ll++)                                         // related to |REG| && boundary: external0-reg
    #       {
    #           sm_b[[6] columns][[7] rows] = dev_v2[[8] ext_addr + [9] int_addr];
    #       }
    #   }
    #
    #print ("sizes: size_ext_tb: ", size_ext_tb, ", size_ext_reg: ", size_ext_reg, ", size_smem_k: ", size_smem_k, ", size_tb_x: ", size_tb_x, ", size_tb_y: ", size_tb_y)

    #
    #   Overall: 
    #
    str_boundary_before_for_statements = ""

    #
    #   [Option]
    #   1: SMEM_A
    #   2: SMEM_B
    #
    if opt_smem_ab == 1:
        str_smem_array = "sm_a"
    else:
        str_smem_array = "sm_b"

    #
    #   [Option]
    #   1: Input-Left
    #   2: Input-Right
    #
    str_input_array     = "dev_" + info_input_tensor[3]
    
    #
    #print ("info_input_tensor: ", info_input_tensor)
    l_input_tensor      = info_input_tensor[4]
    l_input_tensor_name = info_input_tensor[3]

    #
    #   Partial
    #
    str_boundary_ext_smem   = ""
    str_boundary_ext_reg    = ""
    str_boundary_int        = ""
    str_load_smem_column    = ""
    str_load_smem_row       = ""
    str_load_global_ext     = ""
    str_load_global_int     = ""

    #
    #   Secondary-Options
    #
    opt_boundary_if_before_for  = -1
    opt_boundary_inner_internal = -1
    opt_boundary_inner_external = -1

    #
    #   [Check][Options-Secondary]
    #
    if opt_gen_ext == 1 or opt_gen_int == 1:
        opt_boundary_if_before_for = 1



    #
    #   [Boundary][External] Register Tiling
    #
    str_boundary_ext_reg = tc_gen_code_Kernel_Load_Inputs_Boundary_External_REG(opt_gen_ext, opt_axis_reg, l_tile_sizes, l_mapping_reg)

    #   2. For 1.1. case, TB_X -(loads)-> K && TB_Y -(loads)-> E, where E \ REG
    #       2.1. |TB_X| = |K|: nothing to do
    #       2.2. |TB_X| > |K|: boundary case such as threadIdx.x < |K|
    #       2.3. |TB_X| < |K|: need multiple load-instructions
    #
    #       2.4. |TB_Y| = |E|: nothing to do
    #       2.5. |TB_Y| > |E|: boundary case such as threadidx.y < |E|
    #       2.6. |TB_Y| < |E|: need multiple load-instructions
    opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, num_inner_inst_tb_x, num_inner_inst_tb_y = tc_gen_code_Kernel_Load_Inputs_Check_Inner_For_Statements(opt_load_ext_int, size_smem_k, size_ext_tb, size_ext_reg, size_tb_x, size_tb_y) 

    #
    #   [Option][Load][Input][External][Address][Global] Index-Matching Problem
    #
    l_info_matching_indices, opt_matching_index_fully = tc_gen_code_Kernel_Load_Inputs_Addr_Global_External_Matching_Index(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y,
                                                                                                l_input_tensor, l_mapping_tb, l_mapping_reg, l_internal_idx, l_tile_sizes,
                                                                                                num_inner_inst_tb_x, num_inner_inst_tb_y,
                                                                                                size_tb_x, size_tb_y)
    #
    print ("[DEBUG] l_info_matching_indices: ", l_info_matching_indices, ", opt_matching_index_fully: ", opt_matching_index_fully)

    #
    #   [Boundary][External][SMEM]
    #
    if opt_gen_ext == 1:
        str_boundary_ext_smem   = tc_gen_code_Kernel_Load_Inputs_Boundary_Exteranl_TB(opt_load_ext_int, l_input_tensor, l_mapping_reg, l_internal_idx, l_info_matching_indices)

    #
    #   [Boundary][Overall] |TB| vs |SMEM|
    #
    str_boundary_tb_smem = tc_gen_code_Kernel_Load_Input_Boundary_TB_SMEM(opt_load_ext_int, size_ext_tb, size_smem_k, size_tb_x, size_tb_y)
    print ("str_boundary_tb_smem: ", str_boundary_tb_smem)
    if str_boundary_tb_smem != "":
        opt_boundary_if_before_for = 1

    #
    #   [Boundary][Internal]
    #
    if opt_gen_int == 1:
        str_boundary_int        = tc_gen_code_Kernel_Load_Inputs_Boundary_Internal(opt_load_ext_int, idx_kernel)

    #
    #   [Boundary][Overall][SMEM]
    #
    if opt_gen_ext == 1:
        #
        str_boundary_before_for_statements = str_boundary_ext_smem
        #
        #   [Internal]
        #
        if opt_gen_int == 1:
            str_boundary_before_for_statements += " && " + str_boundary_int
        #
        #
        #
        if str_boundary_tb_smem != "":
            if str_boundary_before_for_statements != "":
                str_boundary_before_for_statements = str_boundary_before_for_statements + " && " + str_boundary_tb_smem
            else:
                str_boundary_before_for_statements = str_boundary_tb_smem
    else:
        #
        #   [Internal]
        #
        if opt_gen_int == 1:
            str_boundary_before_for_statements += str_boundary_int
        #
        #
        #
        if str_boundary_tb_smem != "":
            if str_boundary_before_for_statements != "":
                str_boundary_before_for_statements = str_boundary_before_for_statements + " && " + str_boundary_tb_smem
            else:
                str_boundary_before_for_statements = str_boundary_tb_smem

    #
    #   [Boundary][Inner-For-Statement]
    #
    str_boundary_int_smem = ""
    if opt_load_ext_int == -1:
        #
        if opt_inner_load_input_tb_x == 1:
            print ("|TB_X| > |K|")
            str_boundary_int_smem = "threadIdx.x < " + str(size_smem_k)
            str_boundary_before_for_statements = str_boundary_before_for_statements + " && " + str_boundary_int_smem
        
        #
        #   It depends on "Matching Index"
        #
        if opt_inner_load_input_tb_y == 1:
            print ("|TB_Y| > |E|")
    else:
        #
        #   It depends on "Matching Index"
        #
        if opt_inner_load_input_tb_x == 1:
            print ("|TB_X| > |E|")
        
        #
        if opt_inner_load_input_tb_y == 1:
            print ("|TB_Y| > |K|")
            str_boundary_int_smem = "threadIdx.y < " + str(size_smem_k)
            str_boundary_before_for_statements = str_boundary_before_for_statements + " && " + str_boundary_int_smem
    
    #
    #   [Code][Load][Input][Boundary-Checks] Before For-Statements
    #
    if opt_boundary_if_before_for == 1:
        tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs,   "if (",                             -1)
        tc_helper.tc_gen_helper_code_a_line(f, 0,               str_boundary_before_for_statements, -1)
        tc_helper.tc_gen_helper_code_a_line(f, 0,               ")",                                 1)
    else:
        tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// No Need to Put Boundary-Checks before For-Statement: " + str_boundary_before_for_statements + ": " + str_boundary_tb_smem, 1)

    #
    #   [Code][Load][Input][For-Statements] Open
    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs,   "for (int ll = 0; ",    -1)
    tc_helper.tc_gen_helper_code_a_line(f, 0,               "ll < ", -1)
    tc_helper.tc_gen_helper_code_a_line(f, 0,               str_boundary_ext_reg,   -1)
    tc_helper.tc_gen_helper_code_a_line(f, 0,               "; ll++)",               1)
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs,   "{",                     1)
    num_code_tabs += 1

    #
    #
    #
    str_load_smem_column        = tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Column(opt_load_ext_int)
    str_load_smem_row           = tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Row(opt_load_ext_int)


    #
    #   [Code][Load][Input] To Load Inputs
    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// " + str(l_input_tensor), 1)
    for each_inner_load_inst_x in range(num_inner_inst_tb_x):
        for each_inner_load_inst_y in range(num_inner_inst_tb_y):
            #
            #
            #
            str_load_smem_column_inner  = tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Column_Inner(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, size_tb_x, size_tb_y, each_inner_load_inst_x, each_inner_load_inst_y)
            str_load_smem_row_inner     = tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Row_Inner(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, size_tb_x, size_tb_y, size_ext_tb, each_inner_load_inst_x, each_inner_load_inst_y)
            str_load_global_addr_ext, str_load_global_addr_ext_idx = tc_gen_code_Kernel_Load_Inputs_Addr_Global_External(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, l_internal_idx, l_input_tensor, l_mapping_tb, l_mapping_reg, l_tile_sizes, l_info_matching_indices, opt_matching_index_fully, idx_kernel, each_inner_load_inst_x, each_inner_load_inst_y, size_tb_x, size_tb_y)
            str_load_global_addr_int, str_load_global_addr_int_idx = tc_gen_code_Kernel_Load_Inputs_Addr_Global_Internal(opt_load_ext_int, opt_input_ab, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, opt_internal, l_input_tensor_name, l_internal_idx, l_input_tensor, size_smem_k, size_tb_x, size_tb_y, each_inner_load_inst_x, each_inner_load_inst_y)

            #
            #   [Boundary] 
            #
            tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// Exception: Temp. version!: " + str_load_global_addr_int_idx, 1)
            tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// Exception: Temp. version!: " + str_load_global_addr_ext_idx, 1)
            
            #opt_gen_ext
            #opt_gen_int
            str_boundary_inner = ""
            if each_inner_load_inst_x > 0 or each_inner_load_inst_y > 0:
                if opt_gen_int == 1:
                    str_boundary_inner = str_load_global_addr_int_idx + " < size_internal"
                    if opt_gen_ext == 1:
                        str_boundary_inner = str_boundary_inner + " && " + str_load_global_addr_ext_idx
                        tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "if (" + str_boundary_inner + ") ", 1)
                    else:
                        tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "if (" + str_boundary_inner + ") ", 1)
                else:
                    if opt_gen_ext == 1:
                        str_boundary_inner = str_load_global_addr_ext_idx
                        tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "if (" + str_boundary_inner + ") ", 1)
                    else:
                        #
                        tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// Exception: Full-Full", 1)
                #tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "if (" + str_boundary_inner + ") ", 1)
            #
            #   [Code][Write]
            #
            tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs,   str_smem_array,                 -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               "[",                            -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               str_load_smem_column,           -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               str_load_smem_column_inner,     -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               "]",                            -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               "[",                            -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               str_load_smem_row,              -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               str_load_smem_row_inner,        -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               "] = ",                         -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               str_input_array,                -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               "[",                            -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               str_load_global_addr_ext,       -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               " + ",                          -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               str_load_global_addr_int,       -1)
            tc_helper.tc_gen_helper_code_a_line(f, 0,               "];",                            1)

    #
    #   [Code][Load][Input][For-Statements] Close
    #
    num_code_tabs -= 1
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "}", 1)

    #
    #   >>> Base Form <<<
    #   (Optional: Not Matched Indices: Temporal Indices)
    #   if ([1] idx_a < rng_c2 && [2] threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)  // boundary: external-smem, internal
    #   if ([3] threadIdx.x < 8 && [4] threadIdx.y < #)                                     // boundary: |TB| and |K|
    #   {
    #       for (int ll = 0; ll < [5] rng_c1; ll++)                                         // related to |REG| && boundary: external0-reg
    #       {
    #           sm_b[[6] columns][[7] rows] = dev_v2[[8] ext_addr + [9] int_addr];
    #       }
    #   }
    #

#
#   [Option][External][Thread-Block]
#
def tc_gen_code_Kernel_Load_Inputs_Addr_Global_External_Matching_Index(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y,
                                                                       l_input_tensor, l_mapping_tb, l_mapping_reg, l_internal_idx, l_tile_sizes,
                                                                       num_inner_inst_tb_x, num_inner_inst_tb_y,
                                                                       size_tb_x, size_tb_y):
    #
    #   [Option]
    #
    #
    l_info_matching_indices     = []
    l_info_pruned_indices       = []
    l_info_idx_tb               = []
    l_info_pruned_indices_tb    = []
    opt_matching_index_fully    = -1

    #
    #   |TB_X| -(loads)-> K 
    #   |TB_Y| -(loads)-> E ***
    #
    if opt_load_ext_int == -1:
        l_idx_tb = l_mapping_tb[1]
    else:
        l_idx_tb = l_mapping_tb[0]

    #
    #   [Option] Indices mapped on TB except for them having tile-size == 1.
    #
    for each_idx in l_idx_tb:
        if tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx) != 1:
            l_info_pruned_indices_tb.append([each_idx, tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)])

    #
    for each_idx in l_idx_tb:
        l_info_idx_tb.append([each_idx, tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)])

    #
    print ("[Code Generator][Load][Input][Addr][Global][External][Matching-Index] l_info_idx_tb: ", l_info_idx_tb)

    #
    #   [Option] Check if indices mapped on TB_X | TB_Y can be used for the external indices mapped on 
    #
    for each_idx in l_input_tensor:
        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
            if tc_helper.tc_gen_helper_find_1d(l_mapping_reg, each_idx) == -1:
                print ("[Code Generator][Load][Input][Addr][Global][External][Matching-Index] ext. mapped on TB: ", each_idx, ", size: ", tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx))
                #
                #   To Prune Indices with Tile-Size = 1.
                #
                if tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx) == 1:
                    l_info_matching_indices.append([each_idx, "0"])
                else:
                    l_info_pruned_indices.append([each_idx, tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)])

    #
    #   Target Indices should be handled.
    #
    print ("[DEBUG] l_info_pruned_indices: ",       l_info_pruned_indices)
    print ("[DEBUG] l_info_pruned_indices_tb: ",    l_info_pruned_indices_tb)

    #
    #       
    #
    print ("[Code Generator][Load][Input][Addr][Global][External][Matching-Index] # of Pruned Indices mapped on TB: ",  len(l_info_pruned_indices_tb))
    print ("[Code Generator][Load][Input][Addr][Global][External][Matching-Index] # of Pruned Indices: ",               len(l_info_pruned_indices))
    if len(l_info_pruned_indices_tb) == len(l_info_pruned_indices) == 1:
        #print ("len(l_info_pruned_indices_tb) == len(l_info_pruned_indices) == 1")
        len_pruned_indices_tb   = 1
        len_pruned_indice       = 1
        #
        #
        #
        for each_idx_info in l_info_pruned_indices_tb:
            len_pruned_indices_tb *= tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx_info[0])

        #
        for each_idx_info in l_info_pruned_indices:
            len_pruned_indice *= tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx_info[0])

        #
        l_info_matching_indices.append([l_info_pruned_indices[0][0], l_info_pruned_indices_tb[0][0]])
        opt_matching_index_fully = 1
    else:
        #
        #   Miserable Case:
        #
        print ("Miserable Case: Should Handle Manually!")
        len_pruned_indices_tb   = 1
        len_pruned_indice       = 1
        #
        #
        #
        for each_idx_info in l_info_pruned_indices_tb:
            len_pruned_indices_tb *= tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx_info[0])

        #
        for each_idx_info in l_info_pruned_indices:
            len_pruned_indice *= tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx_info[0])

        print ("len_pruned_indices_tb: ", len_pruned_indices_tb)
        print ("len_pruned_indice: ", len_pruned_indice)


    #
    return l_info_matching_indices, opt_matching_index_fully

#
#   [Boundary][External][Thread-Block]
#
def tc_gen_code_Kernel_Load_Inputs_Boundary_Exteranl_TB(opt_load_ext_int, l_input_tensor, l_mapping_reg, l_internal_idx, l_info_matching_indices):
    #
    print ("[Code Generator][Load][Input][Boundary][External][TB]")
    print ("[Code Generator][Load][Input][Boundary][External][TB] l_input_tensor: ", l_input_tensor)
    str_boundary_external_tb = ""

    #
    #   External Indices Mapped on TB
    #
    idx_count = 0
    for each_idx in l_input_tensor:
        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
            if tc_helper.tc_gen_helper_find_1d(l_mapping_reg, each_idx) == -1:
                #
                #
                #
                if tc_helper.tc_gen_helper_find(l_info_matching_indices, each_idx) == "0":
                    str_input_specific_idx = "0"
                else:
                    str_input_specific_idx = "idx_" + tc_helper.tc_gen_helper_find(l_info_matching_indices, each_idx)

                #
                if idx_count == 0:
                    #str_boundary_external_tb = "idx_" + each_idx + " < " + "rng_" + each_idx   # should be fixed.
                    str_boundary_external_tb = str_input_specific_idx + " < " + "rng_" + each_idx   # should be fixed.
                else:
                    #str_boundary_external_tb = str_boundary_external_tb + " && idx_" + each_idx + " < " + "rng_" + each_idx   # should be fixed.
                    str_boundary_external_tb = str_boundary_external_tb + " && " + str_input_specific_idx + " < " + "rng_" + each_idx   # should be fixed.
                #
                idx_count += 1
        
    #
    return str_boundary_external_tb

#
#   [Boundary][External][Register Tiling]
#
def tc_gen_code_Kernel_Load_Inputs_Boundary_External_REG(opt_gen_ext, opt_axis, l_tile_sizes, l_mapping_reg):
    #
    print ("[Code Generator][Load][Input][Boundary][External][REG]")
    str_boundary_external_reg = ""

    #
    #   opt_axis: 0 (x), 1 (y)
    #
    str_reg_idx = ""
    if opt_axis == 0:
        str_reg_idx = l_mapping_reg[0]
    else:
        str_reg_idx = l_mapping_reg[1]

    #
    #
    #
    if opt_gen_ext == 1:
        str_boundary_external_reg = "rng_" + str_reg_idx
    else:
        str_boundary_external_reg = str(tc_helper.tc_gen_helper_find(l_tile_sizes, str_reg_idx))

    #
    return str_boundary_external_reg

#
#   [Boundary][TB][SMEM] |TB| vs |SMEM|
#
def tc_gen_code_Kernel_Load_Input_Boundary_TB_SMEM(opt_load_ext_int, size_ext_tb, size_smem_k, size_tb_x, size_tb_y):
    #
    print ("[Code Generator][Load][Input][Boundary][TB vs SMEM]")
    str_results = ""
    str_result_ext = ""
    str_result_int = ""

    #
    #   |TB_X| -(load)-> K          && |TB_Y| -(load)-> E
    #   |TB_Y| -(load)-> E ***      && |TB_X| -(load)-> K   // (-1)
    #
    #print ("opt_load_ext_int: ", opt_load_ext_int)
    #print ("size_smem_k: ", size_smem_k)
    if opt_load_ext_int == -1:
        #print ("size_tb_y >= size_ext_tb: ", size_tb_y, ", ", size_ext_tb)
        if size_tb_y > size_ext_tb:
            str_result_ext = "threadIdx.y < " + str(size_ext_tb)
        if size_tb_x > size_smem_k:
            str_result_int = "threadIdx.x < " + str(size_smem_k)
    else:
        #print ("size_tb_x >= size_ext_tb: ", size_tb_x, ", ", size_ext_tb)
        if size_tb_x > size_ext_tb:
            str_result_ext = "threadIdx.x < " + str(size_ext_tb)
        if size_tb_y > size_smem_k:
            str_result_int = "threadIdx.y < " + str(size_smem_k)

    #
    if str_result_ext == "":
        str_results = str_result_int
    else:
        str_results = str_result_ext
        if str_result_int != "":
            str_results = str_results + " && " + str_result_int
    
    #
    print ("str_results: ", str_results)
    return str_results

#
#   [Boundary][Internal]
#
def tc_gen_code_Kernel_Load_Inputs_Boundary_Internal(opt_load_ext_int, idx_kernel):
    #
    print ("[Code Generator][Load][Input][Boundary][Internal]")
    str_boundary_internal = ""

    #
    #   |TB_X| -(loads)-> K
    #
    if opt_load_ext_int == -1:
        str_boundary_internal = "threadIdx.x"
    else:
        str_boundary_internal = "threadIdx.y" 

    #
    #
    #
    str_boundary_internal += " < SIZE_INT_UNIT_" + str(idx_kernel) + " - internal_upperbound"

    #
    return str_boundary_internal

#                 v
#   DEFAULT: SMEM[K][E]
#
def tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Column(opt_load_ext_int):
    #
    print ("[Code Generator][Load][Input][Addr][SMEM][Column]")
    str_smem_column_addr = ""

    #
    #   |TB_X| -(loads)-> K ***
    #   |TB_Y| -(loads)-> E
    #
    if opt_load_ext_int == -1:
        str_smem_column_addr = "threadIdx.x"
    #
    #   |TB_X| -(loads)-> E
    #   |TB_Y| -(loads)-> K ***
    #
    else:
        str_smem_column_addr = "threadIdx.y"
    
    #
    return str_smem_column_addr

#                 v
#   DEFAULT: SMEM[K][E] (depending on Inner-Load-Instructions)
#
def tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Column_Inner(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, 
                                                            size_tb_x, size_tb_y, 
                                                            idx_inner_tb_x, idx_inner_tb_y):
    #
    str_smem_column_inner_addr = ""

    #
    #   |TB_X| -(loads)-> K ***
    #   |TB_Y| -(loads)-> E
    #
    if opt_load_ext_int == -1:
        #
        #   [Option] |TB_X| vs |K|
        #
        if opt_inner_load_input_tb_x == 2:
            str_smem_column_inner_addr = " + " + str(size_tb_x * idx_inner_tb_x)
        
    #
    #   |TB_X| -(loads)-> E
    #   |TB_Y| -(loads)-> K ***
    #
    else:
        #
        #   [Option] |TB_Y| vs |K|
        #
        if opt_inner_load_input_tb_y == 2:
            str_smem_column_inner_addr = " + " + str(size_tb_y * idx_inner_tb_y)

    #
    return str_smem_column_inner_addr

#                    v
#   DEFAULT: SMEM[K][E]
#
def tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Row(opt_load_ext_int):
    #
    print ("[Code Generator][Load][Input][Addr][SMEM][Row]")
    str_smem_row_addr = ""

    #
    #   |TB_X| -(loads)-> K
    #   |TB_Y| -(loads)-> E ***
    #
    if opt_load_ext_int == -1:
        str_smem_row_addr = "threadIdx.y"
    #
    #   |TB_X| -(loads)-> E
    #   |TB_Y| -(loads)-> K ***
    #
    else:
        str_smem_row_addr = "threadIdx.x"
    
    #
    return str_smem_row_addr

#                    v
#   DEFAULT: SMEM[K][E] (depending on Inner-Load-Instructions)
#
def tc_gen_code_Kernel_Load_Inputs_Addr_SMEM_Row_Inner(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, 
                                                        size_tb_x,      size_tb_y,
                                                        size_ext_tb,
                                                        idx_inner_tb_x, idx_inner_tb_y):
    #
    str_smem_row_inner_addr = ""
    str_smem_row_stride     = ""

    #
    #   |TB_X| -(loads)-> K 
    #   |TB_Y| -(loads)-> E ***
    #
    if opt_load_ext_int == -1:
        #
        #   [Option] |TB_X| vs |E||
        #
        if opt_inner_load_input_tb_y == 2:
            str_smem_row_inner_addr = " + " + str(size_tb_y * idx_inner_tb_y)
        
    #
    #   |TB_X| -(loads)-> E ***
    #   |TB_Y| -(loads)-> K 
    #
    else:
        #
        #   [Option] |TB_Y| vs |E|
        #
        if opt_inner_load_input_tb_x == 2:
            str_smem_row_inner_addr = " + " + str(size_tb_x * idx_inner_tb_x)

    #
    #   [Alpha][Row][External][REG]
    #
    str_smem_row_inner_addr += " + ll * " + str(size_ext_tb)

    #
    return str_smem_row_inner_addr

#
#   [Load][Input][Address][Global Memeory][External]
#
def tc_gen_code_Kernel_Load_Inputs_Addr_Global_External(opt_load_ext_int, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, l_internal_idx, l_input_tensor, l_mapping_tb, l_mapping_reg, l_tile_sizes, l_info_matching_indices, opt_matching_index_fully, idx_kernel, each_inner_load_inst_x, each_inner_load_inst_y, size_tb_x, size_tb_y):
    #
    print ("[Code Generator][Load][Input][Addr][Global][External]")
    print ("[Code Generator][Load][Input][Addr][Global][External] l_input_tensor: ", l_input_tensor)
    print ("[Code Generator][Load][Input][Addr][Global][External] l_mapping_tb: ", l_mapping_tb)
    print ("[Code Generator][Load][Input][Addr][Global][External] l_mapping_reg: ", l_mapping_reg)
    str_input_ext_global_addr = ""

    #
    #   |TB_X| -(loads)-> K 
    #   |TB_Y| -(loads)-> E ***
    #
    if opt_load_ext_int == -1:
        l_idx_tb = l_mapping_tb[1]
    else:
        l_idx_tb = l_mapping_tb[0]

    #
    #   [Symbols] Block Index, 
    #
    rev_l_input_tensor  = list(reversed(l_input_tensor))
    idx_count           = 0
    idx_base_ext        = 0
    str_specific_idx    = ""
    for each_idx in rev_l_input_tensor:
        #
        #   Internal Index
        #
        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) != -1:
            #print ("[int.] ", each_idx, ", idx_count: ", idx_count)
            if idx_count != 0:
                str_input_ext_global_addr = "(" + str_input_ext_global_addr + ") * size_" + each_idx
                
        #
        #   External Index
        #
        else:
            #
            #   Mapped on REG
            #
            if tc_helper.tc_gen_helper_find_1d(l_mapping_reg, each_idx) != -1:
                #print (" >>> [ext.][REG] ", each_idx)
                if idx_base_ext == 0:
                    str_input_ext_global_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + ll"
                else:
                    str_input_ext_global_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + ll + (" + str_input_ext_global_addr + ") * size_" + each_idx
            else:
                #print (" >>> opts: tb_x: ", opt_inner_load_input_tb_x, ", tb_y: " , opt_inner_load_input_tb_y)
                #print (" >>> [ext.][TB] ", each_idx, " >>> ", tc_helper.tc_gen_helper_find(l_info_matching_indices, each_idx))
                str_input_specific_idx_multi = ""
                #
                #   |TB_X| -(loads)-> K
                #   |TB_Y| -(loads)-> E ***
                #
                if opt_load_ext_int == -1:
                    if opt_inner_load_input_tb_y == 2:
                        str_input_specific_idx_multi = " + " + str(int(size_tb_y * each_inner_load_inst_y))
                #
                #   |TB_X| -(loads)-> E ***
                #   |TB_Y| -(loads)-> K
                #   
                else:
                    if opt_inner_load_input_tb_x == 2:
                        str_input_specific_idx_multi = " + " + str(int(size_tb_x * each_inner_load_inst_x))

                #print (">>> str_input_specific_idx_multi: ", str_input_specific_idx_multi)
                if tc_helper.tc_gen_helper_find(l_info_matching_indices, each_idx) == "0":
                    str_specific_idx                = each_idx
                    str_input_specific_idx          = tc_helper.tc_gen_helper_find(l_info_matching_indices, each_idx)
                    str_input_specific_idx_multi    = ""
                else:
                    str_specific_idx        = each_idx
                    str_input_specific_idx  = "idx_" + tc_helper.tc_gen_helper_find(l_info_matching_indices, each_idx)

                #
                if idx_base_ext == 0:
                    str_input_ext_global_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_input_specific_idx + str_input_specific_idx_multi
                else:
                    str_input_ext_global_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_input_specific_idx + str_input_specific_idx_multi + " + (" + str_input_ext_global_addr + ") * size_" + each_idx
            #
            idx_base_ext += 1
        #
        idx_count += 1
    #
    return str_input_ext_global_addr, str_input_specific_idx + str_input_specific_idx_multi + " < rng_" + str_specific_idx

#
#   [Load][Input][Address][Global Memeory][Internal]
#
def tc_gen_code_Kernel_Load_Inputs_Addr_Global_Internal(opt_load_ext_int, opt_input_ab, opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, opt_internal,
                                                        input_tensor_name, 
                                                        l_internal_idx, l_input_tensor,
                                                        size_smem_k,
                                                        size_tb_x, size_tb_y,
                                                        each_inner_load_inst_x, each_inner_load_inst_y):
    #
    print ("[Code Generator][Load][Input][Addr][Global][Internal]")
    str_addr_global_internal        = ""
    str_addr_tb_axis                = ""
    str_addr_inner_for_statement    = ""

    #print ("(now) l_input_tensor: ", l_input_tensor)

    #
    #   |TB_X| -(loads)-> K ***
    #   |TB_Y| -(loads)-> E
    #
    if opt_load_ext_int == -1:
        str_addr_tb_axis = "threadIdx.x"
        #
        #   [Options] Multiple-Load-Instructions
        #
        if opt_inner_load_input_tb_x == 2:
            print ("[Code Generator][Load][Input][Addr][Global][Internal] opt_inner_load_input_tb_x: ", opt_inner_load_input_tb_x, ", size_tb_x: ", size_tb_x)
            str_addr_inner_for_statement = " + " + str(int(size_tb_x * each_inner_load_inst_x))
    #
    #   |TB_X| -(loads)-> E
    #   |TB_Y| -(loads)-> K ***
    #
    else:
        str_addr_tb_axis = "threadIdx.y"
        #
        #   [Options] Multiple-Load-Instructions
        #
        if opt_inner_load_input_tb_y == 2:
            print ("[Code Generator][Load][Input][Addr][Global][Internal] opt_inner_load_input_tb_y: ", opt_inner_load_input_tb_y, ", size_tb_y: ", size_tb_y)
            str_addr_inner_for_statement = " + " + str(int(size_tb_y * each_inner_load_inst_y))
            
    #
    #   |K| > 1: (basically) to use constant memory
    #
    if len(l_internal_idx) > 1:
        #
        #
        #
        if opt_internal > 1:
            #str_addr_global_internal = "dev_internal_offset_" + input_tensor_name + "[" + str_addr_tb_axis + " + l]"
            str_addr_global_internal = "dev_internal_offset_" + input_tensor_name + "[" + str_addr_tb_axis + " + l" + str_addr_inner_for_statement + "]"
        else:
            #str_addr_global_internal = "const_internal_" + input_tensor_name + "_offset[" + str_addr_tb_axis + " + l]"
            str_addr_global_internal = "const_internal_" + input_tensor_name + "_offset[" + str_addr_tb_axis + " + l" + str_addr_inner_for_statement + "]"
    else:
        #
        #   [Option] Strides
        #
        if l_input_tensor[0] == l_internal_idx[0]:
            #
            str_addr_global_internal = "(" + str_addr_tb_axis + " + l" + str_addr_inner_for_statement + ")"    
        else:
            #
            if opt_input_ab == 1:
                str_addr_global_internal = "(" + str_addr_tb_axis + " + l" + str_addr_inner_for_statement + ") * stride_int_t2"
            else:
                str_addr_global_internal = "(" + str_addr_tb_axis + " + l" + str_addr_inner_for_statement + ") * stride_int_v2"
            
    #
    #   ***
    #
    return str_addr_global_internal, str_addr_tb_axis + " + l" + str_addr_inner_for_statement

#
#   [Load][Input][Check][Inner-For-Statement]
#
def tc_gen_code_Kernel_Load_Inputs_Check_Inner_For_Statements(opt_load_ext_int, 
                                                                size_smem_k, size_ext_tb, size_ext_reg,
                                                                size_tb_x, size_tb_y):
    #
    opt_inner_load_input_tb_x   = -1
    opt_inner_load_input_tb_y   = -1
    num_inner_inst_tb_x         = 1
    num_inner_inst_tb_y         = 1

    #
    #   |TB_X| -(loads)-> K
    #   |TB_Y| -(loads)-> E
    #
    if opt_load_ext_int == -1:
        print ("[Code Generator][Load][Input][Abstract] |TB_X| -(loads)-> K && |TB_Y| -(loads)-> E")
        print ("[Code Generator][Load][Input][Abstract] |TB_X| = ", size_tb_x, ", |TB_Y| = ", size_tb_y)
        print ("[Code Generator][Load][Input][Abstract] |K| = ", size_smem_k, ", |E_TB| = ", size_ext_tb, ", |E_REG| = ", size_ext_reg)

        #
        #   |TB_X| vs |K|
        #       1. |TB_X| = |K|
        #           : GREAT
        #       2. |TB_X| > |K|
        #           : Boundary-Check such as threadIdx.x < |K|
        #       3. |TB_X| < |K|
        #           : Multiple-Load-Instructions such as |K| / |TB_X| times
        #
        if size_tb_x == size_smem_k:
            opt_inner_load_input_tb_x   = 0
        elif size_tb_x > size_smem_k:
            opt_inner_load_input_tb_x   = 1
        else:
            num_inner_inst_tb_x         = int(size_smem_k / size_tb_x)
            opt_inner_load_input_tb_x   = 2

        #
        #   |TB_Y| vs |E_TB|
        #       : Both the # of indices mapped on TB_Y and the # of indices mapped on E_TB are really important.
        #       1. |TB_Y| = |E_TB|
        #       2. |TB_Y| > |E_TB|
        #       3. |TB_Y| < |E_TB|
        #
        if size_tb_y == size_ext_tb:
            opt_inner_load_input_tb_y   = 0
        elif size_tb_y > size_ext_tb:
            opt_inner_load_input_tb_y   = 1
        else:
            num_inner_inst_tb_y         = int(size_ext_tb / size_tb_y)
            opt_inner_load_input_tb_y   = 2

        #
        print ("[Code Generator][Load][Input][Abstract] opt_inner_load_input_tb_x: ", opt_inner_load_input_tb_x, ", opt_inner_load_input_tb_y: ", opt_inner_load_input_tb_y)
        print ("[Code Generator][Load][Input][Abstract] Inner_TB_X: ", num_inner_inst_tb_x, ", Inner_TB_Y: ", num_inner_inst_tb_y)
    #
    #   |TB_X| -(loads)-> E
    #   |TB_Y| -(loads)-> K
    #
    else:
        print ("[Code Generator][Load][Input][Abstract] |TB_X| -(loads)-> E && |TB_Y| -(loads)-> K")
        print ("[Code Generator][Load][Input][Abstract] |TB_X| = ", size_tb_x, ", |TB_Y| = ", size_tb_y)
        print ("[Code Generator][Load][Input][Abstract] |K| = ", size_smem_k, ", |E_TB| = ", size_ext_tb, ", |E_REG| = ", size_ext_reg)

        #   
        #   |TB_X| vs |E_TB|
        #       : Both the # of indices mapped on TB_X and the # of indices mapped on E_TB are really important.
        #
        if size_tb_x == size_ext_tb:
            opt_inner_load_input_tb_x   = 0
        elif size_tb_x > size_ext_tb:
            opt_inner_load_input_tb_x   = 1
        else:
            num_inner_inst_tb_x         = int(size_ext_tb / size_tb_x)
            opt_inner_load_input_tb_x   = 2

        #
        #   |TB_Y| vs |K|
        #       1. |TB_Y| = |K|
        #           : GREAT
        #       2. |TB_Y| > |K|
        #           : Boundary-Check such as threadIdx.y < |K|
        #       3. |TB_Y| < |K|
        #           : Multiple-Load-Instructions such as |K| / |TB_Y| times
        #
        if size_tb_y == size_smem_k:
            opt_inner_load_input_tb_y   = 0
        elif size_tb_y > size_smem_k:
            opt_inner_load_input_tb_y   = 1
        else:
            num_inner_inst_tb_y         = int(size_smem_k / size_tb_y)
            opt_inner_load_input_tb_y   = 2
        
        #
        print ("[Code Generator][Load][Input][Abstract] opt_inner_load_input_tb_x: ", opt_inner_load_input_tb_x, ", opt_inner_load_input_tb_y: ", opt_inner_load_input_tb_y)
        print ("[Code Generator][Load][Input][Abstract] Inner_TB_X: ", num_inner_inst_tb_x, ", Inner_TB_Y: ", num_inner_inst_tb_y)

    #
    return opt_inner_load_input_tb_x, opt_inner_load_input_tb_y, num_inner_inst_tb_x, num_inner_inst_tb_y
