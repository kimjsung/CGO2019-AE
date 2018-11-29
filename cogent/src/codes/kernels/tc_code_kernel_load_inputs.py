import src.generators.tc_helper                                 as tc_helper

import src.codes.kernels.tc_code_kernel_helper                  as tc_code_kernel_helper
import src.codes.kernels.tc_code_kernel_load_inputs_details     as tc_code_kernel_load_inputs_details
import src.codes.kernels.tc_code_kernel_load_inputs_abstract    as tc_code_kernel_load_inputs_abstract

#
#   Fixed Version
#
def tc_gen_code_kernel_load_inputs_base(f,  opt_gen_ext,            opt_gen_int,
                                            opt_load_left,          opt_load_right,     opt_internal,
                                            tensor_contraction, 
                                            l_t3_slices,
                                            l_internal_idx, 
                                            l_t3_mapping_tb_2D,
                                            l_t3_mapping_reg, 
                                            size_smem_k,            size_tb_x,          size_tb_y,
                                            idx_kernel):
    #
    #
    #
    num_code_tabs = 2
    #
    #
    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "//---------------------------------------------------------------------------------------------------", 1)
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// This is for the new version", 1)
    #
    #   >>> Base Form <<<
    #   if (idx_a < rng_c2 && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)  // boundary: external-smem, internal
    #   if (threadIdx.x < 8)                                                        // boundary: |TB| and |K|
    #   {
    #       for (int ll = 0; ll < rng_c1; ll++)                                     // related to |REG| && boundary: external0-reg
    #       {
    #           sm_b[][] = dev_v2[ext_addr + int_addr];
    #       }
    #   }
    #   __synchthread();
    #
    #print ("tensor_contraction: ", tensor_contraction)

    #
    #   Which Axis is mapped on REG for Input-Left and Input-Right
    #
    opt_axis_reg_left   = 0
    opt_axis_reg_right  = 0
    if tensor_contraction[0][2] != "x":
        opt_axis_reg_left = 1

    if tensor_contraction[1][2] != "x":
        opt_axis_reg_right = 1

    #
    #   To Calculate Length of Indices' Tile-Size mapped on TB
    #
    size_len_external_tiles_left    = 1
    size_len_reg_tiles_left         = 1
    for each_idx in tensor_contraction[0][4]:
        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                size_len_external_tiles_left = size_len_external_tiles_left * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)
        else:
            size_len_reg_tiles_left = size_len_reg_tiles_left * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)

    #
    #   To Calculate Length of Indices' Tile-Size mapped on TB
    #
    size_len_external_tiles_right   = 1   
    size_len_reg_tiles_right        = 1
    for each_idx in tensor_contraction[1][4]:
        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                size_len_external_tiles_right = size_len_external_tiles_right * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)
        else:
            size_len_reg_tiles_right = size_len_reg_tiles_right * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)

    print ("[Code Generator][New][Load][Input] size_len_ext_left: ", size_len_external_tiles_left)
    print ("[Code Generator][New][Load][Input] size_len_ext_right: ", size_len_external_tiles_right)
    print ("[Code Generator][New][Load][Input] size_len_reg_left: ", size_len_reg_tiles_left)
    print ("[Code Generator][New][Load][Input] size_len_reg_right: ", size_len_reg_tiles_right)

    #
    #   [Load-Input][Left]
    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// This Part is for Loading Input-Left", 1)
    #
    tc_code_kernel_load_inputs_abstract.tc_gen_code_Kernel_Load_Inputs_Abstracts(f, num_code_tabs, 
                    #
                    tensor_contraction[0],
                    #   options
                    opt_load_left,
                    1, 1,
                    opt_gen_ext, opt_gen_int, 
                    opt_axis_reg_left,  # need to make automatically
                    opt_internal,
                    #   lists
                    l_t3_slices,
                    l_internal_idx,
                    l_t3_mapping_tb_2D,
                    l_t3_mapping_reg,
                    #   sizes
                    size_len_external_tiles_left, size_len_reg_tiles_left,
                    size_smem_k, size_tb_x, size_tb_y,
                    idx_kernel)

    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "", 1)

    #
    #   [Load-Input][Right]
    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "// This Part is for Loading Input-Right", 1)
    #
    tc_code_kernel_load_inputs_abstract.tc_gen_code_Kernel_Load_Inputs_Abstracts(f, num_code_tabs, 
                    #
                    tensor_contraction[1],
                    #   options
                    opt_load_right,
                    2, 2,
                    opt_gen_ext, opt_gen_int, 
                    opt_axis_reg_right,
                    opt_internal,
                    #   lists
                    l_t3_slices,
                    l_internal_idx,
                    l_t3_mapping_tb_2D,
                    l_t3_mapping_reg,
                    #   sizes
                    size_len_external_tiles_right, size_len_reg_tiles_right,
                    size_smem_k, size_tb_x, size_tb_y,
                    idx_kernel)

    #
    #   END: After Loading Both Inputs
    #
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "__syncthreads();", 1)
    
    #   
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "//---------------------------------------------------------------------------------------------------", 1)
    tc_helper.tc_gen_helper_code_a_line(f, num_code_tabs, "\n", 1)

#
#   Should be FIXED!!!!!!!
#
def tc_gen_code_Kernel_Load_Inputs(f,   size_tb_x,          size_tb_y,
                                        size_sm_a,          size_sm_b,          size_sm_p7,
                                        int_str_t2,         int_str_v2,
                                        l_blk_boundary_rng,
                                        tensor_contraction,
                                        l_input_strides,
                                        l_t3_slices,        l_internal_idx,
                                        l_t3_mapping_tb_2D, l_t3_mapping_reg,
                                        opt_gen_full,       opt_gen_p7,
                                        opt_load_t2,        opt_load_v2,        opt_pre_computed,
                                        idx_kernel):
    # For Shared Memory,
    #   need to support non-fvi for p7
    #   >>> it affects how to generalize loading inputs.
    #   [To-Do] What is the purpose of this?
    #   
    if len(l_blk_boundary_rng) > 0:
        upper_left, upper_right, l_left, l_right = tc_gen_code_Kernel_Load_Checking_Boundary(f, l_blk_boundary_rng, tensor_contraction)
    else:
        upper_left  = size_tb_x
        upper_right = size_tb_x

    #
    #   # of Internal Indices
    #
    num_internal_indices = len(l_internal_idx)

    #
    f.write("\t\t// Load Input Tensor to Shared Memory: " + str(size_tb_x) + ":" + str(tc_helper.tc_gen_helper_find(l_t3_slices, l_internal_idx[0])) +  "\n")
    f.write("\t\t// # of Internal Indices: " + str(num_internal_indices) + "\n")

    #
    #   Step 0. Boundaries for LEFT 
    #       - size_tb_x & size_tb_y are determined by tiles' size.
    l_idx_x                 = l_t3_mapping_tb_2D[0]
    l_idx_y                 = l_t3_mapping_tb_2D[1]
    l_left_indices          = tensor_contraction[0][4]
    l_left_target_indices   = list()
    l_left_indices_reg      = list()
    cond_boundary_left_ext  = -1
    cond_boundary_left_int  = -1
    str_cond_gen_external   = ""
    str_cond_gen_internal   = ""
    opt_gen_full_special_case_left = -1
    #
    #   - ThreadIdx.x -> Internal Indices -- |E_K|
    #   - ThreadIdx.y -> External Indices -- |E_LEFT|
    #
    if opt_load_t2 == -1:
        #
        #   OPTION #1. Partial Tiles for External Indices (Assumption: 4D Input Tensors)
        #
        opt_gen_full_special_case = -1
        if opt_gen_full == 1:
            cond_boundary_left_ext = 1
            for each_idx in l_left_indices:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:       # 
                    if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:     #
                        l_left_target_indices.append(each_idx)
                else:
                    l_left_indices_reg.append(each_idx)
            # 
            size_len_external_tiles = 1
            for each_target_index in l_left_target_indices:
                size_len_external_tiles = size_len_external_tiles * tc_helper.tc_gen_helper_find(l_t3_slices, each_target_index)

            #
            if size_tb_y > size_len_external_tiles:
                len_covered_reg = int(size_tb_y / size_len_external_tiles)
                size_reg_tile   = tc_helper.tc_gen_helper_find(l_t3_slices, l_left_indices_reg[0])

                if len_covered_reg > 1:
                    opt_gen_full_special_case = 1

            #
            #   Assumption 4D Input Tensor (3 External (2 -> TB, 1 -> REG), 1 Internal)
            #   New-Version
            #
            #
            #   [1] indices along y-axis can be used directly to check boundaries for indices in the left input.
            #
            list_alternative_mapping    = list()
            opt_boundary_left_input     = -1
            len_l_idx_y                 = len(l_idx_y)
            for each_idx_left in l_left_target_indices:
                for each_idx_y in l_idx_y:
                    if each_idx_left == each_idx_y:
                        len_l_idx_y = len_l_idx_y - 1
            
            #
            #   [1] indices along y-axis can be used directly to check boundaries for indices in the left input.
            #
            if len_l_idx_y == 0:
                opt_boundary_left_input = 1
            #
            #       [2] indices along y-axis can be used in-directly to check boundaries for indices in the left input.
            #   or  [3] indices along y-axis cannot bu used to check boundaries for indices in the left input.
            #   
            else:
                #
                #   [2] indices along y-axis can be used in-directly to check boundaries for indices in the left input.
                #   
                #
                if len(l_idx_y) == len(l_left_target_indices):
                    #
                    #   The Simplest Version (Directly Replacing Indices)
                    #
                    for each_idx_left in l_left_target_indices:
                        for each_idx_y in l_idx_y:
                            if each_idx_left == each_idx_y:
                                list_alternative_mapping.append([each_idx_y, each_idx_left])
                                break
                            else:
                                if tc_helper.tc_gen_helper_find(l_t3_slices, each_idx_left) == tc_helper.tc_gen_helper_find(l_t3_slices, each_idx_y):
                                    list_alternative_mapping.append([each_idx_y, each_idx_left])
                                    break
                
                    #
                    if len(list_alternative_mapping) == len(l_idx_y):
                        print ("[To-Do] list_alternative_mapping: ", list_alternative_mapping)
                        opt_boundary_left_input = 2
                    else:
                        opt_boundary_left_input = 3

                else:
                    opt_boundary_left_input = 3

                #
                #   [3] indices along y-axis cannot bu used to check boundaries for indices in the left input.
                #



            if opt_boundary_left_input == 1:
                #print (" >>> [1] indices along y-axis can be used directly to check boundaries for indices in the left input.")
                idx_count = 0
                for idx_tb in l_left_target_indices:
                    if idx_count == 0:
                        str_cond_gen_external = "idx_" + idx_tb + " < rng_" + idx_tb
                    else:
                        str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_tb + " < rng_" + idx_tb
                    idx_count = idx_count + 1
            elif opt_boundary_left_input == 2:
                #print (" >>> [2] indices along y-axis can be used in-directly to check boundaries for indices in the left input.")
                idx_count = 0
                for idx_mapping in list_alternative_mapping:
                    if idx_count == 0:
                        str_cond_gen_external = "idx_" + idx_mapping[0] + " < rng_" + idx_mapping[1]
                    else:
                        str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_mapping[0] + " < rng_" + idx_mapping[1]
                    idx_count = idx_count + 1

            else:
                print (" >>> [3] indices along y-axis cannot bu used to check boundaries for indices in the left input.")
                print (" >>> ERROR!!!! Not Support Yet")


            del list_alternative_mapping
            
        #
        #   OPTION #2. Partial Tiles for Internal Indices
        #
        if opt_gen_p7 == 1:
            cond_boundary_left_int = 1
            str_cond_gen_internal = "threadIdx.x < SIZE_INT_UNIT_" + str(idx_kernel) + " - internal_upperbound"

    #
    #   - ThreadIdx.x -> External Indices -- |E_LEFT|
    #   - ThreadIdx.y -> Internal Indices -- |E_K|
    #
    else:
        #
        #   OPTION #1. Partial Tiles for External Indices
        #
        if opt_gen_full == 1:
            #
            #   This case, TB_Y will load |T_K|
            #   However, when |TB_Y| < |T_K|, we need to load input (|T_K| / |TB_Y|) times.
            #
            cond_boundary_left_ext = 1
            for each_idx in l_left_indices:
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:       #   Not mapped on REG
                    if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:     #   External Indices
                        l_left_target_indices.append(each_idx)
                else:
                    l_left_indices_reg.append(each_idx)                                     #   Indices Mapped on REG

            #
            #   To Calculate Length of Indices' Tile-Size mapped on TB
            #
            size_len_external_tiles = 1
            for each_target_index in l_left_target_indices:
                size_len_external_tiles = size_len_external_tiles * tc_helper.tc_gen_helper_find(l_t3_slices, each_target_index)

            #
            #   
            #
            if size_tb_x > size_len_external_tiles:
                opt_gen_full_special_case_left = 1

            #
            #   Assumption 4D Input Tensor (3 External (2 -> TB, 1 -> REG), 1 Internal)
            #   To-Do: Fusion is a little bit complecated.
            #
            #print ("l_left_target_indices: ", l_left_target_indices)
            #print ("l_idx_x: ", l_idx_x)
            #print ("[To-Do] Boundary Case for External Indices ---- Fusion")
            #
            #   For a Tensor Contraction,
            #
            if len(l_left_target_indices) == len(l_idx_x):
                #
                #print ("============================")
                opt_fusion = -1
                for each_target in l_left_target_indices:
                    is_common = -1
                    for each_idx_x in l_idx_x:
                        if each_target == each_idx_x:
                            is_common = 1

                    if is_common == -1:
                        opt_fusion = 1
                        break

                #
                if opt_fusion == 1:
                    #print ("opt_fusion == 1")
                    idx_count = 0
                    for idx_tb in l_idx_x:
                        if idx_count == 0:
                            #str_cond_gen_external = "idx_" + idx_tb + " < rng_" + idx_tb#l_left_target_indices[idx_count]
                            str_cond_gen_external = "idx_" + idx_tb + " < rng_" + l_left_target_indices[idx_count]
                        else:
                            #str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_tb + " < rng_" + idx_tb#l_left_target_indices[idx_count]
                            str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_tb + " < rng_" + l_left_target_indices[idx_count]
                        idx_count = idx_count + 1
                else:
                    #print ("opt_fusion != 1")
                    idx_count = 0
                    for idx_tb in l_idx_x:
                        if idx_count == 0:
                            str_cond_gen_external = "idx_" + idx_tb + " < rng_" + idx_tb
                        else:
                            str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_tb + " < rng_" + idx_tb
                        idx_count = idx_count + 1

            else:
                idx_count = 0
                for idx_tb in l_idx_x:
                    if idx_count == 0:
                        #str_cond_gen_external = "idx_" + idx_tb + " < rng_" + idx_tb#l_left_target_indices[idx_count]
                        str_cond_gen_external = "idx_" + idx_tb + " < rng_" + l_left_target_indices[idx_count]
                    else:
                        #str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_tb + " < rng_" + idx_tb#l_left_target_indices[idx_count]
                        str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_tb + " < rng_" + l_left_target_indices[idx_count]
                    idx_count = idx_count + 1

            
        #
        #   OPTION #2. Partial Tiles for Internal Indices
        #
        if opt_gen_p7 == 1:
            cond_boundary_left_int = 1
            str_cond_gen_internal = "threadIdx.y < SIZE_INT_UNIT_" + str(idx_kernel) + " - internal_upperbound"

    #
    #   To Write Code for Boundary Cases
    #
    if (cond_boundary_left_ext == 1 and opt_gen_full_special_case_left == -1) or cond_boundary_left_int == 1:
        #
        #
        #
        tc_code_kernel_helper.code_kernel_load_input_left_boundary_case(f, 
                                                                    opt_gen_full_special_case_left, 
                                                                    cond_boundary_left_ext, cond_boundary_left_int,
                                                                    str_cond_gen_external, str_cond_gen_internal)

    #
    #
    #
    l_left_indices_target_temp  = list()
    l_left_indices_reg_temp     = list()
    opt_gen_full_special_case   = -1 
    len_covered_reg             =  1
    #
    #   To Figure out indices mapped on TB (l_left_indices_target_temp) and indices mapped on REG (l_left_indices_reg_temp)
    #
    for each_idx in l_left_indices:
        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                l_left_indices_target_temp.append(each_idx)
        else:
            l_left_indices_reg_temp.append(each_idx)

    #   To Calculate Length of Indices' Tile-Size mapped on TB
    len_external_tiles_left = 1
    for each_target_index in l_left_indices_target_temp:
        len_external_tiles_left = len_external_tiles_left * tc_helper.tc_gen_helper_find(l_t3_slices, each_target_index)
    
    #   Assumed only one index is mapped on REG
    size_reg_tile = tc_helper.tc_gen_helper_find(l_t3_slices, l_left_indices_reg_temp[0])

    #
    #   TB_X will load |E_LEFT| on TB_X without an index mapped on REG
    #
    if opt_load_t2 == -1:
        if size_tb_y > len_external_tiles_left:
            len_covered_reg = int(size_tb_y / len_external_tiles_left)
            opt_gen_full_special_case = 1
        else:
            len_covered_reg = 1
    #
    #   |TB_X| > |E_LEFT_TB|, Then, |TB_X| can cover |E_LEFT_TB|
    #    
    else:
        if size_tb_x > len_external_tiles_left:
            print ("[aft]size_tb_x > len_external_tiles_left :: ", size_tb_x, " >? ", len_external_tiles_left)
            # how many steps for register tile can be covered by TB_X
            len_covered_reg             = int(size_tb_x / len_external_tiles_left)
            opt_gen_full_special_case   = 1
        #
        #   |TB_X| == |E_LEFT_TB|
        #
        elif size_tb_x == len_external_tiles_left:   
            len_covered_reg = 1
    #
    #   |TB_X| < |E_LEFT_TB|
    #
    #else:
        #print ("HERE: ", size_tb_x, len_external_tiles_left)
        #len_covered_reg             = 1#size_tb_x / len_external_tiles_left # (To-Do) This case will be dealt in the loop.
        #opt_gen_full_special_case   = 2
    

    #
    #   Step 1: For-Statement: T2 (LEFT), |E_A'| * |E_A''|, where  A' is a set of indices mapped on Thread Block and
    #                                                             A'' is a set of indices mapped on Register Tile.
    tc_code_kernel_helper.code_kernel_load_input_left_for_statement(f, opt_gen_full, tensor_contraction[0][2],
    opt_gen_full_special_case, size_reg_tile, len_covered_reg, l_t3_mapping_reg)
    
    #
    str_str_t2  = ""
    idx_count   = 0
    for each_idx in tensor_contraction[0][4]:
        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
            if idx_count != 0:
                str_str_t2 = str_str_t2 + " * "
            str_str_t2 = str_str_t2 + "SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize()
            idx_count = idx_count + 1

    #   To Calculate Length of Indices' Tile-Size mapped on TB
    size_len_external_tiles_left    = 1
    size_len_reg_tiles_left         = 1
    for each_idx in tensor_contraction[0][4]:
        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                size_len_external_tiles_left = size_len_external_tiles_left * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)
        else:
            size_len_reg_tiles_left = size_len_reg_tiles_left * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)

    #
    #   [Sub-Routine] Load Tensor Inputs to sm_a[][]
    #
    if len(l_input_strides) > 0:
        tc_code_kernel_load_inputs_details.tc_gen_code_Kernel_Load_Inputs_Left(f, tensor_contraction, l_internal_idx, opt_load_t2, 
                                            size_tb_x, size_tb_y, size_sm_p7, size_len_external_tiles_left, str_str_t2, num_internal_indices, idx_kernel, l_input_strides[0], 
                                            opt_pre_computed, l_t3_mapping_tb_2D, l_t3_mapping_reg, 
                                            l_t3_slices)
    else:
        tc_code_kernel_load_inputs_details.tc_gen_code_Kernel_Load_Inputs_Left(f, tensor_contraction, l_internal_idx, opt_load_t2, 
                                            size_tb_x, size_tb_y, size_sm_p7, size_len_external_tiles_left, str_str_t2, num_internal_indices, idx_kernel, l_input_strides, 
                                            opt_pre_computed, l_t3_mapping_tb_2D, l_t3_mapping_reg,
                                            l_t3_slices)

    #   To Calculate Length of Indices' Tile-Size mapped on TB
    size_len_external_tiles_right   = 1   
    size_len_reg_tiles_right        = 1
    for each_idx in tensor_contraction[1][4]:
        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                size_len_external_tiles_right = size_len_external_tiles_right * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)
        else:
            size_len_reg_tiles_right = size_len_reg_tiles_right * tc_helper.tc_gen_helper_find(l_t3_slices, each_idx)

    #print ("size_len_reg_tiles_left, right:", size_len_reg_tiles_left, ",", size_len_reg_tiles_right)
    #
    #   When we cannot merge Two different loops for Both Inputs.
    #
    #if ((size_sm_a / size_tb_y) != (size_sm_b / size_tb_y)) or (opt_load_t2 != opt_load_v2) or (size_tb_x > upper_right) or (size_tb_x > upper_left) or opt_gen_full == 1 or (size_len_reg_tiles_left != size_len_reg_tiles_right):
    if ((size_sm_a / size_tb_y) != (size_sm_b / size_tb_y)) or (opt_load_t2 != opt_load_v2) or opt_gen_full == 1 or (size_len_reg_tiles_left != size_len_reg_tiles_right):
        f.write("\t\t}\n")
        f.write("\n")
    
        #
        #
        #
        l_right_indices         = tensor_contraction[1][4]
        l_right_indices_target  = list()
        l_right_indices_reg     = list()
        cond_boundary_right_ext = -1
        cond_boundary_right_int = -1
        cond_boundary_right_tbx = -1
        cond_boundary_right_tby = -1
        str_cond_gen_external   = ""
        str_cond_gen_internal   = ""
        str_cond_gen_tb_x       = ""
        str_cond_gen_tb_y       = ""

        f.write("\t\t// Load Input Tensor to Shared Memory\n")
        #
        #   [Load][Right] TB_X -> |K| && TB_Y -> |E_RIGHT|
        #
        if opt_load_v2 == -1:
            #
            #   OPTION #1.   External Index
            #
            if opt_gen_full == 1:
                cond_boundary_right_ext = 1
                for each_idx in l_right_indices:
                    if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
                        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                            l_right_indices_target.append(each_idx)
                    else:
                        l_right_indices_reg.append(each_idx)
                
                #print ("len(l_right_indices_target):", len(l_right_indices_target))
                #print ("len(l_idx_x);", len(l_idx_x))

                #
                #   To-Do
                #
                #   4D Input Tensor
                if len(l_right_indices_target) == len(l_idx_y):
                    idx_count = 0
                    for idx_tb in l_idx_y:
                        if idx_count == 0:
                            str_cond_gen_external = "idx_" + idx_tb + " < rng_" + l_right_indices_target[idx_count]
                        else:
                            str_cond_gen_external = str_cond_gen_external + " && idx_" + idx_tb + " < rng_" + l_right_indices_target[idx_count]
                        idx_count = idx_count + 1
                else:
                    print ("ERROR: (-1) Input Tensor Should be 4D...")

            #
            #   OPTION #2.  Internal Index
            #
            if opt_gen_p7 == 1:
                cond_boundary_right_int = 1
                str_cond_gen_internal = "threadIdx.x < SIZE_INT_UNIT_" + str(idx_kernel) + " - internal_upperbound"
            #
            #   OPTION #3.
            #
            if size_tb_x > size_sm_p7:
                cond_boundary_right_tbx     = 1
                str_cond_gen_tb_x           = "threadIdx.x < " + str(size_sm_p7)
        else:
            #
            #   OPTION #1.  External Index
            #
            #print ("[CODE][LOAD-INPUT][RIGHT] TB_X -> |E_RIGHT| && TB_Y -> |K|")
            opt_gen_full_special_case = -1
            if opt_gen_full == 1:
                cond_boundary_right_ext = 1
                for each_idx in l_right_indices:
                    if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
                        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                            l_right_indices_target.append(each_idx)
                    else:
                        l_right_indices_reg.append(each_idx)

                #   To Calculate Length of Indices' Tile-Size mapped on TB
                size_len_external_tiles = 1
                for each_target_index in l_right_indices_target:
                    size_len_external_tiles = size_len_external_tiles * tc_helper.tc_gen_helper_find(l_t3_slices, each_target_index)
                
                #   TB_X will load |E_LEFT| on TB_X without the index mapped on REG
                if size_tb_x > size_len_external_tiles:
                    # how many steps for register tile can be covered by TB_X
                    len_covered_reg = int(size_tb_x / size_len_external_tiles)
                    # Assumed only one index is mapped on REG
                    size_reg_tile = tc_helper.tc_gen_helper_find(l_t3_slices, l_right_indices_reg[0])

                    if len_covered_reg > 1:
                        opt_gen_full_special_case = 1


                '''
                if len(l_idx_y) > len(l_left_target_indices):
                idx_count = 0
                for idx_tb in l_left_target_indices:
                    if idx_count == 0:
                        str_cond_gen_external = "idx_" + l_idx_y[idx_count] + " < rng_" + idx_tb
                    else:
                        str_cond_gen_external = str_cond_gen_external + " && idx_" + l_idx_y[idx_count] + " < rng_" + idx_tb#l_left_target_indices[idx_count]
                    idx_count = idx_count + 1
                '''


                #
                
                #print ("len(l_right_indices_target):", len(l_right_indices_target))
                #print ("len(l_idx_x);", len(l_idx_x))

                if len(l_idx_x) > len(l_right_indices_target):
                    idx_count = 0
                    for idx_tb in l_right_indices_target:
                        if idx_count == 0:
                            str_cond_gen_external = "idx_" + l_idx_x[idx_count] + " < rng_" + idx_tb
                        else:   
                            temp = str_cond_gen_external     # bug??
                            str_cond_gen_external = temp + " && idx_" + l_idx_x[idx_count] + " < rng_" + idx_tb
                        idx_count = idx_count + 1
                else:
                    idx_count = 0
                    for idx_tb in l_idx_x:
                        if idx_count == 0:
                            str_cond_gen_external = "idx_" + idx_tb + " < rng_" + l_right_indices_target[idx_count]
                        else:   
                            temp = str_cond_gen_external     # bug??
                            str_cond_gen_external = temp + " && idx_" + idx_tb + " < rng_" + l_right_indices_target[idx_count]
                        idx_count = idx_count + 1
                '''
                if len(l_right_indices_target) == len(l_idx_x):
                    idx_count = 0
                    for idx_tb in l_idx_x:
                        if idx_count == 0:
                            str_cond_gen_external = "idx_" + idx_tb + " < rng_" + l_right_indices_target[idx_count]
                        else:   
                            temp = str_cond_gen_external     # bug??
                            str_cond_gen_external = temp + " && idx_" + idx_tb + " < rng_" + l_right_indices_target[idx_count]
                        idx_count = idx_count + 1
                else:
                    print ("ERROR: (!-1) Input Tensor Should be 4D...")
                '''
                                
            #
            #   OPTION #2.
            #
            if opt_gen_p7 == 1:
                cond_boundary_right_int = 1
                str_cond_gen_internal = "threadIdx.y < SIZE_INT_UNIT_" + str(idx_kernel) + " - internal_upperbound"

        #
        #   Boundary Cases (External, Internal and Thread Block)
        #   To Write Code for Boundary Cases
        #
        if cond_boundary_right_ext == 1 or cond_boundary_right_int == 1 or cond_boundary_right_tbx == 1:
            #
            #
            #
            tc_code_kernel_helper.code_kernel_load_input_right_boundary_case(f, 
                                            cond_boundary_right_ext, cond_boundary_right_tbx, cond_boundary_right_int, 
                                            str_cond_gen_external,  str_cond_gen_tb_x, str_cond_gen_internal)


        #
        #
        l_right_indices_target_temp = list()
        l_right_indices_reg_temp    = list()
        #
        for each_idx in l_right_indices:
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
                if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                    l_right_indices_target_temp.append(each_idx)
            else:
                l_right_indices_reg_temp.append(each_idx)

        #   To Calculate Length of Indices' Tile-Size mapped on TB
        size_len_external_tiles = 1
        for each_target_index in l_right_indices_target_temp:
            size_len_external_tiles = size_len_external_tiles * tc_helper.tc_gen_helper_find(l_t3_slices, each_target_index)
        
        # Assumed only one index is mapped on REG
        size_reg_tile = tc_helper.tc_gen_helper_find(l_t3_slices, l_right_indices_reg_temp[0])

        #print ("size_tb_x: ", size_tb_x)
        #print ("size_tb_y: ", size_tb_y)
        #print ("size_len_external_tiles: ", size_len_external_tiles)
        opt_gen_full_special_case = -1
        #
        #   TB_X will load |E_K|.
        #
        if opt_load_v2 == -1:
            #
            #
            if size_tb_y > size_len_external_tiles:
                # how many steps for register tile can be covered by TB_X
                len_covered_reg = int(size_tb_y / size_len_external_tiles)
                
                if len_covered_reg > 1:
                    opt_gen_full_special_case = 1
            else:
                len_covered_reg = 1
        #
        #   TB_X will load |E_RIGHT|.
        #
        else:
            #
            #
            if size_tb_x > size_len_external_tiles:
                # how many steps for register tile can be covered by TB_X
                len_covered_reg = int(size_tb_x / size_len_external_tiles)
                
                if len_covered_reg > 1:
                    opt_gen_full_special_case = 1
            else:
                len_covered_reg = 1

        #
        #   Step 2: This "For-Statement" is related to "Regiter-Tile."
        #           The size of Thread Block depends on Indices' Tile-Size mapped on Thread Block.
        #           However, when the lengh of a dimension along thread block which load inputs can cover some "Register-Tile,"
        #           then, we need to change the ranges.
        #
        tc_code_kernel_helper.code_kernel_load_input_right_for_statement(f, 
                                                                            opt_gen_full, tensor_contraction[1][2],
                                                                            opt_gen_full_special_case, 
                                                                            size_len_reg_tiles_right, size_reg_tile, len_covered_reg, l_t3_mapping_reg)

        #
        #   free
        #
        del l_right_indices_target
        del l_right_indices_reg
        del l_right_indices_target_temp
        del l_right_indices_reg_temp
    #
    #
    #
    str_str_v2  = ""
    idx_count   = 0
    for each_idx in tensor_contraction[1][4]:
        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
            if idx_count != 0:
                str_str_v2 = str_str_v2 + " * "
            str_str_v2 = str_str_v2 + "SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize()
            idx_count = idx_count + 1

    #
    #   [Sub-Routine] Load Tensor Inputs to sm_b[][]
    #
    if len(l_input_strides) > 0:
        tc_code_kernel_load_inputs_details.tc_gen_code_Kernel_Load_Inputs_Right(f, tensor_contraction, l_internal_idx, opt_load_v2, 
                                            size_tb_x, size_tb_y, size_sm_p7, size_len_external_tiles_right, str_str_v2, num_internal_indices, idx_kernel, l_input_strides[2], 
                                            opt_pre_computed, l_t3_mapping_tb_2D, l_t3_mapping_reg,
                                            opt_gen_full, opt_gen_p7,
                                            l_t3_slices)
    else:
        tc_code_kernel_load_inputs_details.tc_gen_code_Kernel_Load_Inputs_Right(f, tensor_contraction, l_internal_idx, opt_load_v2, 
                                            size_tb_x, size_tb_y, size_sm_p7, size_len_external_tiles_right, str_str_v2, num_internal_indices, idx_kernel, l_input_strides, 
                                            opt_pre_computed, l_t3_mapping_tb_2D, l_t3_mapping_reg, 
                                            opt_gen_full, opt_gen_p7,
                                            l_t3_slices)

    #
    #   END: To Load Inputs
    #
    f.write("\t\t}\n")
    f.write("\t\t__syncthreads();\n")
    f.write("\n")

