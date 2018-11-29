#!/usr/bin/python
import sys
import os

#
import src.algs.bases.tc_gen_inputs             as tc_gen_inputs

#
import src.generators.tc_helper                 as tc_helper
import src.generators.tc_gen_etc                as tc_gen_etc
import src.generators.proc_input_groups         as proc_input_groups

#
import src.algs.bases.tc_gen_permutations       as tc_gen_permutations

#
import src.codes.tc_code_include                as tc_code_include
import src.codes.tc_code_define                 as tc_code_define
import src.codes.tc_code_globalvar              as tc_code_globalvar
import src.codes.tc_code_etc                    as tc_code_etc

#
import src.codes.kernels.tc_code_kernel         as tc_code_kernel
import src.codes.kernels.tc_interface           as tc_interface
import src.codes.kernels.tc_code_kernel_fusion  as tc_code_kernel_fusion

#
import src.codes.others.tc_pre_BasicBlock       as tc_pre_BasicBlock
import src.codes.others.tc_pre_SD2_Functions    as tc_pre_SD2_Functions
import src.codes.others.tc_pre_IndirectArray    as tc_pre_IndirectArray
import src.codes.others.tc_pre_CUDA_Malloc      as tc_pre_CUDA_Malloc
import src.codes.others.tc_pre_findSlice        as tc_pre_findSlice
import src.codes.others.tc_post_Correctness     as tc_post_Correctness      # need to check if both += and -= work well or not.
import src.codes.others.tc_post_HostDevice_Free as tc_post_HostDevice_Free

#
l_host_dynamic      = list()
l_device_dynamic    = list()
l_cuda_malloc       = list()
l_blk_boundary_rng  = list()

#
l_tmp_mapping_register_transpose = list()

#
#   [Main: Pre-Processing]
#
def tc_gen_input(tmp_count, tmp_config, filename, opt_print, opt_data_type):
    #
    #   Input:  filename
    #   Output: l_str_input_tensor_contractions
    #
    l_str_input_tensor_contractions = tc_gen_etc.tc_gen_helper_GettingInputs(filename)

    #
    if opt_print == 1:
        print ("l_str_input_tensor_contractions: ", l_str_input_tensor_contractions)

    #
    #   Step 1: Create Outer-Groups based on Output's index from Input Tensor Contractions
    #       Input:  l_str_input_tensor_contractions, opt_print (1: ON)
    #       Output: l_outer_groups--- [0] output index, [1] tensor contractions, [2] all index
    #
    l_outer_groups = tc_gen_inputs.tc_gen_Outer_Group(l_str_input_tensor_contractions, 1)    #

    #
    if opt_print == 1:
        print ("l_outer_groups: ", l_outer_groups)

    #
    #   Step 2: Create Inner-Groups based on "Tile-Sizes" and "Mappings" Within an Outer-Group
    #       Input:  l_outer_groups, opt_print (1: ON)
    #       Output: l_inner_groups
    #   
    #   Option for Register Transpose: -1 (OFF), 1 (ON)
    l_inner_groups, l_interface_info = proc_input_groups.tc_gen_Inner_Group(l_outer_groups, tmp_count, tmp_config, -1, opt_data_type)

    #
    if opt_print == 1:
        print ("l_inner_groups: ",      l_inner_groups)
        print ("l_interface_info: ",    l_interface_info)

    #
    #   (ISSUE) Assume That An-Outer-Group      --> Multi-Inner-Groups  (Currrent)
    #                       Multi-Outer-Groups  --> Multi-Inner-Groups  (Future)
    #

    #
    #   Step 3: Tune Data for "tc_gen_code"
    #       Input:  l_inner_groups, tmp_count, opt_print (1: ON)
    #       Output: l_temp_inner_output
    #
    opt_register_transpose  = -1
    l_temp_inner_output     = tc_gen_inputs.tc_gen_Processing_Inner_Group(l_inner_groups, tmp_count, opt_register_transpose, -1)

    #
    #   Return Output
    #
    return l_temp_inner_output, l_interface_info

#
#   Processing for Register Transpose
#
#
#   Function for Checking if Fusion with Register Transpose is possible or not.
#
#   - Assumptions for Fusion
#   (1) LHS among tensor contractions should be identical.                                                                                      # Outer-Groups
#   (2) For a given tile size and a given mapping, both indices used in a mapping of Register Tiling cannot come from the same input tensor.    # Inner-Groups
#   (3) For a given tile size, the size of Shared Memory for each tensor contraction should be same among tensor contractions.                  # Register Transpose
#   : Based on the given tile size, the size of Shared Memory among multiple tensor contractions might be different.
#   : 
#
def tc_gen_Q_Fusion_Register_Transpose(l_temp_inner_output):
    print ("[Code Generator][Fusion_Q_Register_Transpose] # of Inner-Groups (Processed Data): ", len(l_temp_inner_output))

    #
    l_temp_register_transpose_inner_output = list()

    #
    #   Before Processing Inner-Groups for Register Transpose,
    #   We need to check if Inner-Groups can be combined by using Register Transpose.
    #       (Temporary)
    #       1. Mappings for TB_X and REG_X should be identical.
    #       2. There should exist Indices mapped on TB_Y whose Tile-size 
    #
    l_ok_combined_inner_groups = list()  # It holds indices of inner-groups which can be combined by using Register Transpose.

    #
    idx_inner_group = 1
    for each_inner_group in l_temp_inner_output:
        #   Creating a List for Y-Axis of BASE
        l_base_y_axis = list()
        for each_idx in each_inner_group[1][1]:
            l_base_y_axis.append(each_idx)
        l_base_y_axis.append(each_inner_group[2][1])

        #   Target Inner-Groups
        idx_target_group = 1
        for each_target_group in l_temp_inner_output:
            #
            if idx_target_group > idx_inner_group:
                #   Creating a List for Y-Axis of TARGET
                l_target_y_axis = list()
                for each_idx in each_target_group[1][1]:
                    l_target_y_axis.append(each_idx)
                l_target_y_axis.append(each_target_group[2][1])

                #   Check: Base's Y-axis == Target's Y-axis
                if len(list(set(l_base_y_axis) & set(l_target_y_axis))) == len(l_base_y_axis):
                    #
                    if  tc_helper.tc_gen_helper_find(each_target_group[8], each_target_group[2][1]) == tc_helper.tc_gen_helper_find(each_inner_group[8], each_inner_group[2][1]) and \
                        tc_helper.tc_gen_helper_find(each_target_group[8], each_inner_group[2][1]) == tc_helper.tc_gen_helper_find(each_inner_group[8], each_target_group[2][1]):
                        print ("[Code Generator][Fusion_Register_Transpose] Base #", idx_inner_group, "can be combined with Targe #", idx_target_group)     
                        l_ok_combined_inner_groups.append([idx_inner_group, idx_target_group])          

            idx_target_group = idx_target_group + 1

        idx_inner_group = idx_inner_group + 1

    #
    return l_ok_combined_inner_groups

#
#   Function for Constraints (To-Do: Different File)
#
def tc_gen_Constraints(f, size_tb_x, size_tb_y, size_sm_a, size_sm_b, size_sm_p7):
    # Const. (5)
    if (size_tb_x * size_tb_y) < 64:
        # exit
        print ("[Code Generator][tc_gen_Constraints] ERROR: Const. (5): The number of Threads in a Thread Block should be greater than or equal to 64: " + str((size_tb_x * size_tb_y)) + ": " + str(size_tb_x) + "," + str(size_tb_y))
        f.close()
        os.remove(f.name)
        sys.exit()
    else:
        if (size_tb_x * size_tb_y) > 1024:
            # exit
            print ("[Code Generator][tc_gen_Constraints] ERROR: By Const. (5): The number of Threads in a Thread Block should be less than or equal to 1024: " + str((size_tb_x * size_tb_y)))
            f.close()
            os.remove(f.name)
            sys.exit()
    '''
    if (size_tb_x * size_tb_y) != 256:
        print ("ERROR: Current Exp. for all cases of the number of threads in a TB is 256")
        f.close()
        os.remove(f.name)
        sys.exit()
    '''
    #
    if (size_sm_a * size_sm_p7) + (size_sm_b * size_sm_p7) > (6 * 1024):
        print ("[Code Generator][tc_gen_Constraints]ERROR: Const. (9): The Size of Shared Memory in a Thread Block should be less than or equal to 6K: " + str((size_sm_a * size_sm_p7) + (size_sm_b * size_sm_p7)))
        f.close()
        os.remove(f.name)
        sys.exit()

    print ("[Code Generator][tc_gen_Constraints] PASSED: Const. (5) and (9)")

#
#   [Register Transpose]
#
def tc_gen_Check_RegisterTranspose(l_inner_groups):
    #
    #   list --- l_inner_groups:
    #           [0] 4 External Index mapped on Thread Blocks (1D)
    #           [1] 4 External Index mapped on Thread Blocks (2D)
    #           [2] 2 Exteranl Index mapped on Register Tile
    #           [3] Problem Size (DO NOT USE: Will be Excluded)
    #           [4] External Indices
    #           [5] Internal Indices
    #           [6] Tensor Contractions will be Fused.
    #           [7] Info. of [6] (CAN BE MERGED WITH [6])
    #           [8] Tile Sizes
    #

    #
    #   Let N be the number of inner-groups in "l_inner_groups."
    #
    #
    src_count = 0

    if len(l_inner_groups) == 1:
        return -1

    for src_inner_group in l_inner_groups:
        src_reg_x = src_inner_group[2][0]
        src_reg_y = src_inner_group[2][1]

        #
        #
        #
        src_size_tb_x   = 1
        src_size_tb_y   = 1
        src_size_reg_x  = 1
        src_size_reg_y  = 1

        for each in src_inner_group[1][0]:
            src_size_tb_x = src_size_tb_x * tc_helper.tc_gen_helper_find(src_inner_group[8], each)
        
        for each in src_inner_group[1][1]:
            src_size_tb_y = src_size_tb_y * tc_helper.tc_gen_helper_find(src_inner_group[8], each)
        
        src_size_reg_x = tc_helper.tc_gen_helper_find(src_inner_group[8], src_inner_group[2][0])
        src_size_reg_y = tc_helper.tc_gen_helper_find(src_inner_group[8], src_inner_group[2][1])
    
        dest_count = 0
        for dest_inner_group in l_inner_groups:
            dest_reg_x      = dest_inner_group[2][0]
            dest_reg_y      = dest_inner_group[2][1]

            dest_size_tb_x  = 1
            dest_size_tb_y  = 1
            dest_size_reg_x = 1
            dest_size_reg_y = 1
            
            for each in dest_inner_group[1][0]:
                dest_size_tb_x = dest_size_tb_x * tc_helper.tc_gen_helper_find(dest_inner_group[8], each)
            
            for each in dest_inner_group[1][1]:
                dest_size_tb_y = dest_size_tb_y * tc_helper.tc_gen_helper_find(dest_inner_group[8], each)

            dest_size_reg_x = tc_helper.tc_gen_helper_find(dest_inner_group[8], dest_inner_group[2][0])
            dest_size_reg_y = tc_helper.tc_gen_helper_find(dest_inner_group[8], dest_inner_group[2][1])

            if src_count < dest_count:
                #
                #   [1] Tile-Sizes Should be Identical
                #  
                if src_inner_group[8] != dest_inner_group[8]:
                    print ("[Code Generator][Register Transpose] UnCompatible: Tile-Sizes")
                    return -1

                #
                #   [2] TB-SIZE
                #
                if src_size_tb_x != dest_size_tb_x or src_size_tb_y != dest_size_tb_y:
                    print ("[Code Generator][Register Transpose] UnCompatible: TB-Sizes")
                    return -1

                #
                #   [3] REG-SIZE
                #
                if src_size_reg_x != dest_size_reg_x or src_size_reg_y != dest_size_reg_y:
                    print ("[Code Generator][Register Transpose UncCompatible: REG-Sizes")
                    return -1

                #
                #   Compatible If [1], [2] and [3] are enough for Register-Transpose.
                #
                print ("[Code Generator][Register Transpose] Compatible")

            dest_count = dest_count + 1

        src_count = src_count + 1

    return 1
