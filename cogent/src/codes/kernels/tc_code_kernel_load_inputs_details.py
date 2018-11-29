import copy
import src.generators.tc_helper                                 as tc_helper
import src.codes.kernels.tc_code_kernel_load_inputs_abstract    as tc_code_kernel_load_inputs_abstract

from inspect import currentframe, getframeinfo


#
#   [To-Do] Need to Make a Complete Algorithm about How to Load Inputs
#
def tc_gen_code_Kernel_Load_Inputs_Left(f, tensor_contraction, l_internal_idx, opt_load_t2, size_tb_x, size_tb_y, size_sm_p7, size_tb_ext, str_str_t2, num_internal_indices, idx_kernel, str_stride_int, opt_pre_computed, l_t3_mapping_tb_2D, l_t3_mapping_reg, l_t3_slices):
    #
    #   Modulo (related to For-Statement)
    #
    opt_modulo = 1  #  1: TRUE  // modulo operation is possible
                    # -1: FALSE // modulo operation is impossible

    #
    #   To-Do: What is "opt_special??" This is identical to "opt_load_t2"
    #
    opt_special = -1
    if num_internal_indices == 1:
        idx_count = 0
        for each_idx in tensor_contraction[0][4]:
            if each_idx == l_internal_idx[0]:
                if idx_count == 0:
                    opt_special = 1
            idx_count = idx_count + 1

    #
    #   w/o pre-computed
    #   This is not yet generalized.
    #
    str_input_addr_left = ""
    if opt_pre_computed == -1:
        f.write("\t\t\t// without pre-computed arrays\n")

        #
        l_tb_idx = list()
        for each_axis in l_t3_mapping_tb_2D:
            for each_idx in each_axis:
                l_tb_idx.append(each_idx)

        #
        l_input_idx_left        = tensor_contraction[0][4]
        rev_l_input_idx_left    = list(reversed(tensor_contraction[0][4]))

        #
        l_ext_tb_smem = list()
        for idx_ext_tb in tensor_contraction[0][4]:
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, idx_ext_tb) == -1:
                if tc_helper.tc_gen_helper_find_1d(l_internal_idx, idx_ext_tb) == -1:
                    l_ext_tb_smem.append(idx_ext_tb)

        #
        opt_print       = 1        

        #
        #   [1-1] TB_X -> K && "TB_Y -> E_A"
        #
        if opt_load_t2 == -1:
            #
            if opt_print == 1:
                print ("[Code Generator][Kernel][Load Inputs-Left] opt_load_t2: ", opt_load_t2, ": TB_X -> E_A && TB_Y -> K")
            #
            #   there are three ways to load inputs:    [1] Directly
            #                                           [2] Swapped
            #                                           [3] Manually
            #
            method_load_t2      = 1
            list_swappable_pair = list()    # original -> alternative

            #
            #   For "TB_Y -> E_A"
            #   # of Indices mapped on TB_Y == # of Indices mapped on SMEM
            #
            if len(l_t3_mapping_tb_2D[1]) == len(l_ext_tb_smem):
                #
                #   Check if [1] or not
                #
                for each_idx in range(0, len(l_ext_tb_smem)):
                    print (">>> l_ext_tb_smem[each_idx]: ", l_ext_tb_smem[each_idx], ", l_t3_mapping_tb_2D[0][each_idx]: ", l_t3_mapping_tb_2D[1][each_idx])
                    if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) != tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[1][each_idx]):
                        method_load_t2 = 2

                #
                #   Check if [2] or not
                #
                if method_load_t2 != 1:
                    tmp_l_ext_tb_smem = copy.deepcopy(l_ext_tb_smem)
                    for each_tb_idx in l_t3_mapping_tb_2D[1]:
                        #
                        #
                        #
                        idx_count = 0
                        for each_input_idx in tmp_l_ext_tb_smem:
                            if tc_helper.tc_gen_helper_find(l_t3_slices, each_tb_idx) == tc_helper.tc_gen_helper_find(l_t3_slices, each_input_idx):
                                list_swappable_pair.append([each_input_idx, each_tb_idx])
                                tmp_l_ext_tb_smem.pop(idx_count)
                            #
                            idx_count = idx_count + 1
                    #
                    if len(tmp_l_ext_tb_smem) != 0:
                        method_load_t2 = 3
                    #else:
                    #    print ("[LEFT][2] list_swappable_pair: ", list_swappable_pair)
                    
                    #
                    del tmp_l_ext_tb_smem
                else:
                    #
                    #   [1]
                    #
                    for each_idx in range(0, len(l_ext_tb_smem)):
                        if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) == tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[1][each_idx]):
                            list_swappable_pair.append([l_ext_tb_smem[each_idx], l_t3_mapping_tb_2D[1][each_idx]])
                    print ("[LEFT][IF][1] list_swappable_pair: ", list_swappable_pair)
            #
            #   (3) // 
            #
            else:
                #print ("len(l_t3_mapping_tb_2D[1]) != len(l_ext_tb_smem): Need Extra Variables")
                method_load_t2 = 3

            #
            print ("[Code Generator][Kernel][Load Inputs-LEFT] When we load input, we will use [", method_load_t2, "] method")

            #
            #   Both (1) and (2)
            #
            if method_load_t2 < 3:
                print ("[Code Generator][Kernel][Load Inputs-Left] Both (1) and (2) Cases")
            #
            #   (3)
            #
            else:
                print ("[Code Generator][Kernel][Load Inputs-Left] (3) Case")
                print ("[Code Generator][Kernel][Load Inputs-Left] |TB_Y| = ", size_tb_y, ", |SMEM| = ", size_tb_ext)
                #
                #   1: [1]
                #   2: [2]
                #
                opt_how_to_manually = 1

                #
                #   [SMEM_X] How to Load Input Tensors on SMEM_X
                #
                int_num_idx_smem_y  = 0
                int_num_idx_tb_y    = len(l_t3_mapping_tb_2D[1])
                for each_idx in rev_l_input_idx_left:
                    if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
                            print ("This Index is mapped on SMEM_Y Directly: ", each_idx, ", |idx| = ", tc_helper.tc_gen_helper_find(l_t3_slices, each_idx))
                            int_num_idx_smem_y += 1
                #
                #   [TB_Y] 
                #
                for each_idx in l_t3_mapping_tb_2D[1]:
                    print ("[TB_Y] ", each_idx, ", |idx| = ", tc_helper.tc_gen_helper_find(l_t3_slices, each_idx))

                #
                #
                #
                print ("# of External Indices mapped on SMEM_Y: ", int_num_idx_smem_y)
                print ("# of External Indices mapped on TB_Y: ", int_num_idx_tb_y)

                #
                #   [Manually]
                #
                if int_num_idx_smem_y == 1 and int_num_idx_tb_y == 1:
                    print ("[1] # of External Indices mapped on SMEM_Y == # of External Indices mapped on TB_Y == 1")
                
                else:
                    print ("[2] Otherwise, we need temporal indices to indicate indices mapped on SMEM_Y")
                    f.write("\t\t\t// tmp tmp tmp\n")

            #
            #
            #
            idx_count = 0
            for each_idx in rev_l_input_idx_left:
                print ("rev_l_input_idx_left: ", rev_l_input_idx_left, ", each_idx: ", each_idx)
                #
                #   [Current Ver.] only one index can mapped on REG
                #
                str_specific_idx = ""
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) != -1:
                    #
                    #   index (ll) -> for-statement
                    #
                    str_specific_idx = "ll"
                else:
                    if method_load_t2 == 1:
                        str_specific_idx = "idx_" + str(tc_helper.tc_gen_helper_find(list_swappable_pair, each_idx))
                    elif method_load_t2 == 2:
                        str_specific_idx = "idx_" + str(tc_helper.tc_gen_helper_find(list_swappable_pair, each_idx))
                    else:
                        print ("Need to Fix IT!")
                        str_specific_idx = "idx_#"

                #
                #   Internal Index
                #
                if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) != -1:
                    if idx_count == 0:
                        str_input_addr_left = ""
                    else:
                        str_input_addr_left = "(" + str_input_addr_left + ") * size_" + each_idx
                #
                #   External Index
                #
                else:
                    if idx_count == 0:
                        str_input_addr_left = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx
                    else:
                        str_input_addr_left = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx + " + (" + str_input_addr_left + ") * size_" + each_idx
                    #
                    #   Need to Skip for the Internal Index on SVI
                    #
                    idx_count = idx_count + 1
            #
            del list_swappable_pair
            #print ("result: ", str_input_addr_left)
            
        #
        #   [1-2] "TB_X -> EA" && TB_Y -> K
        #
        else:
            #
            if opt_print == 1:
                print ("[Code Generator][Kernel][Load Inputs-Left] opt_load_t2: ", opt_load_t2, ": TB_X -> K && TB_Y -> E_A")
            #
            #   there are three ways to load inputs:    [1] Directly
            #                                           [2] Swapped
            #                                           [3] Manually
            #
            method_load_t2      = 1
            list_swappable_pair = list()
            if len(l_t3_mapping_tb_2D[0]) == len(l_ext_tb_smem):
                #
                #   Check If [1] or not
                #
                #print ("Need To Check If [1] or not")
                for each_idx in range(0, len(l_ext_tb_smem)):
                    print (">>> l_ext_tb_smem[each_idx]: ", l_ext_tb_smem[each_idx], ", l_t3_mapping_tb_2D[0][each_idx]: ", l_t3_mapping_tb_2D[0][each_idx])
                    if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) != tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][each_idx]):
                        method_load_t2 = 2
                #
                #   Check If [2] or not
                #
                if method_load_t2 != 1:
                    #print ("Need to Check If [2] or not")
                    tmp_l_ext_tb_smem = copy.deepcopy(l_ext_tb_smem)
                    for each_tb_idx in l_t3_mapping_tb_2D[0]:
                        #
                        #
                        #
                        idx_count = 0
                        for each_input_idx in tmp_l_ext_tb_smem:
                            if tc_helper.tc_gen_helper_find(l_t3_slices, each_tb_idx) == tc_helper.tc_gen_helper_find(l_t3_slices, each_input_idx):
                                list_swappable_pair.append([each_input_idx, each_tb_idx])
                                tmp_l_ext_tb_smem.pop(idx_count)
                            #
                            idx_count = idx_count + 1
                    #
                    if len(tmp_l_ext_tb_smem) != 0:
                        method_load_t2 = 3
                    #else:
                        #print ("[LEFT][2] list_swappable_pair: ", list_swappable_pair)
                    
                    #
                    del tmp_l_ext_tb_smem
                else:
                    for each_idx in range(0, len(l_ext_tb_smem)):
                        if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) == tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][each_idx]):
                            list_swappable_pair.append([l_ext_tb_smem[each_idx], l_t3_mapping_tb_2D[0][each_idx]])
                    print ("[LEFT][ELSE][1] list_swappable_pair: ", list_swappable_pair)
            else:
                #
                #
                #
                #print ("len(l_t3_mapping_tb_2D[0]) != len(l_ext_tb_smem): Need Extra Variables")
                method_load_t2 = 3

            #
            print ("[Code Generator][Kernel][Load Inputs-LEFT] When we load input, we will use [", method_load_t2, "] method")
            #
            #
            #
            idx_count = 0
            for each_idx in rev_l_input_idx_left:
                print ("rev_l_input_idx_left: ", rev_l_input_idx_left, ", each_idx: ", each_idx)
                #
                #   [Current Ver.] only one index can be mapped on REG
                #
                str_specific_idx = ""
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) != -1:
                    #
                    #   index (ll) -> for-statement
                    #
                    str_specific_idx = "ll"
                else:
                    if method_load_t2 == 1:
                        str_specific_idx = "idx_" + each_idx
                    elif method_load_t2 == 2:
                        str_specific_idx = "idx_" + str(tc_helper.tc_gen_helper_find(list_swappable_pair, each_idx))
                    else:
                        print ("Need to Fix IT!!!!")
                        str_specific_idx = "idx_#" 
                
                #
                #   Internal Index
                #
                if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) != -1:
                    if idx_count == 0:
                        str_input_addr_left = ""
                    else:
                        str_input_addr_left = "(" + str_input_addr_left + ") * size_" + each_idx
                #
                #   External Index
                #
                else:
                    if idx_count == 0:
                        str_input_addr_left = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx
                    else:
                        str_input_addr_left = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx + " + (" + str_input_addr_left + ") * size_" + each_idx
                    #
                    #   Need to Skip for the internal index on SVI.
                    #
                    idx_count = idx_count + 1
            #
            del list_swappable_pair
        #
        #
        #
        del l_tb_idx
        del l_ext_tb_smem
    #
    #   Load Tensor Inputs to sm_a[][]
    #   [1] The FVI in an Input is an Internal Index
    #
    if opt_load_t2 == -1:
        #
        #   [1-1] |TB_X| < |T_k|: Need Multiple-Load-Instructions
        #
        if size_tb_x < size_sm_p7:
            for inner_step in range(0, int(size_sm_p7 / size_tb_x)):
                f.write("\t\t\tsm_a[threadIdx.x + " + str(int(inner_step * size_tb_x)) + "][threadIdx.y + ll * SIZE_TB_" + str(idx_kernel) + "_Y] = ")
                f.write("dev_" + tensor_contraction[0][3] + "[")

                #
                #
                #
                if opt_pre_computed == -1:
                    f.write(str_input_addr_left)
                else:
                    f.write("dev_" + tensor_contraction[0][3] + "_addr[threadIdx.y + ")
                    f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_Y")
                    f.write(" + blockIdx.x * (" + str_str_t2 + ")]")

                #
                #   |K| > 1
                #
                if num_internal_indices > 1:
                    f.write(" + const_internal_" + tensor_contraction[0][3] + "_offset[(threadIdx.x + " + str(int(inner_step * size_tb_x)) + " + l)]];\n")
                #
                #   |K| <= 1
                #
                else:
                    if opt_special == 1:
                        f.write(" + (threadIdx.x + " + str(int(inner_step * size_tb_x)) + " + l)]; // 1\n")
                    else:
                        #f.write(" + (threadIdx.x + " + str(int(inner_step * size_tb_x)) + " + l) * " + tensor_contraction[0][1] + "];\n")
                        f.write(" + (threadIdx.x + " + str(int(inner_step * size_tb_x)) + " + l) * " + str_stride_int + "]; // 2\n")
        #
        #   [1-2] |TB_X| >= |T_K|:
        #
        else:
            f.write("\t\t\t// |TB_X| >= |T_K|\n")
            f.write("\t\t\tsm_a[threadIdx.x][threadIdx.y + ll * SIZE_TB_" + str(idx_kernel) + "_Y] = ") # LHS
            f.write("dev_" + tensor_contraction[0][3] + "[")                                            # RHS

            #
            if opt_pre_computed == -1:
                f.write(str_input_addr_left)
            else:
                f.write("dev_" + tensor_contraction[0][3] + "_addr[threadIdx.y + ")
                f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_Y")
                f.write(" + blockIdx.x * (" + str_str_t2 + ")]")

            #
            #   |K| > 1
            #
            if num_internal_indices > 1:
                f.write(" + const_internal_" + tensor_contraction[0][3] + "_offset[threadIdx.x + l]];\n")
            #
            #   |K| <= 1
            #
            else:
                if opt_special == 1:
                    f.write(" + (threadIdx.x + l)]; // 5\n")
                else:
                    #f.write(" + (threadIdx.x + l) * " + tensor_contraction[0][1] + "];\n")
                    f.write(" + (threadIdx.x + l) * " + str_stride_int + "]; // 4 \n")
    #
    #   [2] The FVI in an Input is an External Index.
    #
    else:
        #
        #
        #
        if size_tb_y < size_sm_p7:
            for inner_step in range(0, int(size_sm_p7 / size_tb_y)):
                f.write("\t\t\tsm_a[threadIdx.y + " + str(int(inner_step * size_tb_y)) + "][threadIdx.x + ll * SIZE_TB_" + str(idx_kernel) + "_X] = ")
                f.write("dev_" + tensor_contraction[0][3] + "[")

                #
                #
                #
                if opt_pre_computed == -1:
                    f.write(str_input_addr_left)
                else:
                    f.write("dev_" + tensor_contraction[0][3] + "_addr[threadIdx.x + ")
                    f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_X")
                    f.write(" + blockIdx.x * (" + str_str_t2 + ")]")

                #
                #
                #
                if num_internal_indices > 1:
                    f.write(" + const_internal_" + tensor_contraction[0][3] + "_offset[threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l]];\n")
                else:
                    if opt_special == 1:
                        f.write(" + (threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l)]; // 12\n")
                    else:
                        f.write(" + (threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l) * " + str_stride_int + "]; // 11\n")
        else:
            f.write("\t\t\tsm_a[threadIdx.y][threadIdx.x + ll * SIZE_TB_" + str(idx_kernel) + "_X] = ")
            f.write("dev_" + tensor_contraction[0][3] + "[")

            #
            if opt_pre_computed == -1:
                f.write(str_input_addr_left)
            else:
                f.write("dev_" + tensor_contraction[0][3] + "_addr[threadIdx.x + ")
                f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_X")
                f.write(" + blockIdx.x * (" + str_str_t2 + ")]")

            if num_internal_indices > 1:
                f.write(" + const_internal_" + tensor_contraction[0][3] + "_offset[threadIdx.y + l]];\n")
            else:
                if opt_special == 1:
                    f.write(" + (threadIdx.y + l)]; // 9\n")
                else:
                    f.write(" + (threadIdx.y + l) * " + str_stride_int + "]; // 8\n")

#
#
#
def tc_gen_code_Kernel_Load_Inputs_Right(f, tensor_contraction, l_internal_idx, opt_load_v2, size_tb_x, size_tb_y, size_sm_p7, size_tb_ext, str_str_v2, num_internal_indices, idx_kernel, str_stride_int, opt_pre_computed, l_t3_mapping_tb_2D, l_t3_mapping_reg, opt_full_partial_ext, opt_full_partial_int, l_t3_slices):
    #
    #   [DEBUG]
    #
    #frameinfo = getframeinfo(currentframe())
    #print (frameinfo.filename, frameinfo.lineno)
    print ("===============================================================================================")
    print ("[Code Generator][Kernel][Load Input-Right] Start")

    #
    #   Modulo (related to For-Statement)
    #
    opt_modulo          =  1    #  1: TRUE  // modulo operation is possible
                                # -1: FALSE // modulo operation is impossible
    
    #
    #
    #
    opt_special = -1
    if num_internal_indices == 1:
        idx_count = 0
        for each_idx in tensor_contraction[1][4]:
            if each_idx == l_internal_idx[0]:
                if idx_count == 0:
                    opt_special = 1
            idx_count = idx_count + 1

    #
    #   [Option] w/o pre-computed arrays
    #
    str_input_addr_right = ""
    if opt_pre_computed == -1:
        f.write("\t\t\t// without pre-computed arrays (Right)\n")

        #
        l_tb_idx = list()
        for each_axis in l_t3_mapping_tb_2D:
            for each_idx in each_axis:
                l_tb_idx.append(each_idx)
            
        #
        l_input_idx_right       = tensor_contraction[1][4]
        rev_l_input_idx_right   = list(reversed(tensor_contraction[1][4]))
    
        #
        l_ext_tb_smem = list()
        for idx_ext_tb in l_input_idx_right:
            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, idx_ext_tb) == -1:
                if tc_helper.tc_gen_helper_find_1d(l_internal_idx, idx_ext_tb) == -1:
                    l_ext_tb_smem.append(idx_ext_tb)

        #print ("l_ext_tb_smem: ", l_ext_tb_smem)
        #
        #
        opt_print = 1

        #
        #   [1-1] TB_X -> K && "TB_Y" -> E_B
        #
        if opt_load_v2 == -1:
            #
            if opt_print == 1:
                print ("[Code Generator][Kernel][Load Input-Right] opt_load_v2: ", opt_load_v2, ": TB_X -> E_B && TB_Y -> K")

            #
            #   (1) Directly
            #   (2) Swapped
            #   (3) Manually
            #
            method_load_v2      = 1
            list_swappable_pair = list()

            #
            #   For "TB_Y -> E_B,"
            #   # of Indices mapped on TB_Y == # of Indices mapped on SMEM
            #
            print ("l_t3_mapping_tb_2D[1]: ", l_t3_mapping_tb_2D[1])
            print ("l_ext_tb_smem: ", l_ext_tb_smem)
            if len(l_t3_mapping_tb_2D[1]) == len(l_ext_tb_smem):
                #
                #   Check if (1) is or not
                #
                for each_idx in range(0, len(l_ext_tb_smem)):
                    if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) != tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[1][each_idx]):
                        method_load_v2 = 2
                #
                #   Check if (2) is or not
                #
                if method_load_v2 != 1:
                    tmp_l_ext_tb_smem = copy.deepcopy(l_ext_tb_smem)
                    for each_tb_idx in l_t3_mapping_tb_2D[1]:
                        #
                        #
                        #
                        idx_count = 0
                        for each_input_idx  in tmp_l_ext_tb_smem:
                            if tc_helper.tc_gen_helper_find(l_t3_slices, each_tb_idx) == tc_helper.tc_gen_helper_find(l_t3_slices, each_input_idx):
                                list_swappable_pair.append([each_input_idx, each_tb_idx])
                                tmp_l_ext_tb_smem.pop(idx_count)
                            #
                            idx_count = idx_count + 1
                    #
                    if len(tmp_l_ext_tb_smem) != 0:
                        method_load_v2 = 3
                    #else:
                        #print ("[RIGHT][2] list_swappable_pair: ", list_swappable_pair)
                
                    #
                    del tmp_l_ext_tb_smem
                else:
                    #
                    #   (1)
                    #
                    for each_idx in range(0, len(l_ext_tb_smem)):
                        if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) == tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[1][each_idx]):
                            list_swappable_pair.append([l_ext_tb_smem[each_idx], l_t3_mapping_tb_2D[1][each_idx]])
                    #print ("[RIGHT][1] list_swappable_pair: ", list_swappable_pair)
            else:
                print ("len(l_t3_mapping_tb_2D[1]) != len(l_ext_tb_smem): Need Extra Variables", len(l_t3_mapping_tb_2D[1]), ", ", len(l_ext_tb_smem))
                method_load_v2 = 3

            #
            print ("[Code Generator][Kernel][Load Inputs-RIGHT] When we load input, we will use [", method_load_v2, "] method (if)")

            #
            #
            #
            idx_count = 0
            for each_idx in rev_l_input_idx_right:
                #
                #
                #
                str_specific_idx = ""
                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) != -1:
                    #
                    #
                    #
                    str_specific_idx = "ll"
                else:
                    if method_load_v2 == 1:
                        str_specific_idx = "idx_" + str(tc_helper.tc_gen_helper_find(list_swappable_pair, each_idx))
                    elif method_load_v2 == 2:
                        str_specific_idx = "idx_" + str(tc_helper.tc_gen_helper_find(list_swappable_pair, each_idx))
                    else:
                        print ("method_load_v2 == 3: not yet supported")
                        str_specific_idx = "idx_#"
                    
                #
                #   Internal Idex
                #
                if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) != -1:
                    if idx_count == 0:
                        str_input_addr_right = ""
                    else:
                        str_input_addr_right = "(" + str_input_addr_right + ") * size_" + each_idx
                #
                #   External Index
                #
                else:
                    if idx_count == 0:
                        str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx
                    else:
                        str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx + " + (" + str_input_addr_right + ") * size_" + each_idx
                    #
                    #
                    # 
                    idx_count = idx_count + 1

            #
            del list_swappable_pair
        #
        #   [1-2] "TB_X" -> E_B && TB_Y -> K
        #    
        else:
            #
            if opt_print == 1:
                print ("[Code Generator][Kernel][Load Input-Right] opt_load_v2: ", opt_load_v2, ": TB_X -> K && TB_Y -> E_B")
            #
            #   (1) Directly
            #   (2) Swapped
            #   (3) Manually (Partially Swapped or Manually)
            #
            method_load_v2      = 1
            list_swappable_pair = list()
            '''
            if opt_print == 1:
                #
                idx_count = 0
                for each_idx in l_t3_mapping_tb_2D[0]:
                    print ("TB_X[", idx_count ,"]: ", each_idx)
                    idx_count += 1
                #
                idx_count = 0
                for each_idx in l_ext_tb_smem:
                    print ("SMEM_X[", idx_count, "]: ", each_idx)

                print ("len(l_t3_mapping_tb_2D[0]) == len(l_ext_tb_smem): ", len(l_t3_mapping_tb_2D[0]), ", ", len(l_ext_tb_smem))
            '''
            #
            #   len(l_t3_mapping_tb_2D[0]): # of Indices mapped on x-axis (for the FVI)
            #   len(l_ext_tb_smem):         # of External Indices except for indices mapped on Register Tiles.
            #
            if len(l_t3_mapping_tb_2D[0]) == len(l_ext_tb_smem):
                #
                #   Check if (1) is or not
                #
                for each_idx in range(0, len(l_ext_tb_smem)):
                    if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) != tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][each_idx]):
                        #
                        #   (1) is not possible.
                        #
                        method_load_v2 = 2
                #
                #   Check if (2) is or not
                #
                if method_load_v2 != 1:
                    #
                    #   Need to Check if there are swappable indices or not.
                    #
                    tmp_l_ext_tb_smem = copy.deepcopy(l_ext_tb_smem)
                    for each_tb_idx in l_t3_mapping_tb_2D[0]:
                        #
                        #
                        #
                        idx_count = 0
                        for each_input_idx  in tmp_l_ext_tb_smem:
                            if tc_helper.tc_gen_helper_find(l_t3_slices, each_tb_idx) == tc_helper.tc_gen_helper_find(l_t3_slices, each_input_idx):
                                list_swappable_pair.append([each_input_idx, each_tb_idx])
                                tmp_l_ext_tb_smem.pop(idx_count)
                            #
                            idx_count = idx_count + 1
                    #
                    #   (3) Manually
                    #
                    if len(tmp_l_ext_tb_smem) != 0:
                        method_load_v2 = 3
                    
                    #
                    del tmp_l_ext_tb_smem
                #
                #   To Handle (1)
                #
                else:
                    #
                    #   (1) Directly
                    #   >>> output: list_swappable_pair
                    #
                    for each_idx in range(0, len(l_ext_tb_smem)):
                        if tc_helper.tc_gen_helper_find(l_t3_slices, l_ext_tb_smem[each_idx]) == tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][each_idx]):
                            list_swappable_pair.append([l_ext_tb_smem[each_idx], l_t3_mapping_tb_2D[0][each_idx]])
                    #
                    #print ("[RIGHT][1] list_swappable_pair: ", list_swappable_pair)
            #
            #   Both (1) and (2) are Impossible
            #
            else:
                #
                #   (3) Manually
                #
                method_load_v2 = 3

            #
            #   After figuring out how to handle boundary cases according to tile-sizes and mappings,
            #   it needs to find a way to calculate the input tensor's addresses.    
            #
            if opt_print == 1:
                print ("[Code Generator][Kernel][Load Inputs-RIGHT] When we load input, we will use [", method_load_v2, "] method (else)")
                print (">>> l_t3_slices: ", l_t3_slices)
                print (">>> TB_X: ", l_t3_mapping_tb_2D[0])
                print (">>> TB_Y: ", l_t3_mapping_tb_2D[1])
                print (">>> REG_X: ", l_t3_mapping_reg[0])
                print (">>> REG_Y: ", l_t3_mapping_reg[1])
                print (">>> size_tb_ext: ", size_tb_ext, ", size_tb_x: ", size_tb_x, ", size_tb_y: ", size_tb_y, "size_sm_p7: ", size_sm_p7)
                print (">>> list_swappable_pair: ", list_swappable_pair)


            #
            #   (1) and (2)
            #
            if method_load_v2 < 3:
                idx_count = 0
                for each_idx in rev_l_input_idx_right:
                    print ("rev_l_input_idx_right: ", rev_l_input_idx_right, ", each_idx: ", each_idx)
                    #
                    #   an index mapped on REG
                    #   the others should be mapped on TB
                    #
                    str_specific_idx = ""
                    print ("tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx): ", tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx))
                    if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) != -1:
                            #
                            #
                            #
                            str_specific_idx = "ll"
                        else:
                            #
                            #   (1) Directly
                            #
                            if method_load_v2 == 1:
                                str_specific_idx = "idx_" + str(tc_helper.tc_gen_helper_find(list_swappable_pair, each_idx))
                            #
                            #   (2) Swappable
                            #
                            elif method_load_v2 == 2:
                                str_specific_idx = "idx_" + str(tc_helper.tc_gen_helper_find(list_swappable_pair, each_idx))
                            #
                            #   (3) Manually
                            #
                            else:
                                print ("==================================================================")
                                print ("method_load_v2 == 3: not yet supported")
                                print ("|TB_X| = ", size_tb_x, ", |SMEM_X| = ", size_tb_ext)
                                print ("==================================================================")
                                str_specific_idx = "idx_#"
                    #
                    #   
                    #
                    else:
                        #
                        str_specific_idx = ""
                        
                    #
                    #   Internal Idex
                    #   [To-Do] Multiple-Internal Indices
                    #
                    if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) != -1:
                        if idx_count == 0:
                            str_input_addr_right = ""
                        else:
                            str_input_addr_right = "(" + str_input_addr_right + ") * size_" + each_idx
                    #
                    #
                    #
                    else:
                        print (">>>> ", each_idx, ", str_specific_idx: ", str_specific_idx)
                        if idx_count == 0:
                            str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx
                        else:
                            str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx + " + (" + str_input_addr_right + ") * size_" + each_idx
                        #
                        #
                        # 
                        idx_count = idx_count + 1
                #
                del list_swappable_pair
            #
            #   (3) 
            #
            else:
                print ("[Else]==================================================================")
                print (" method_load_v2 == 3: not yet supported")
                print (" |TB_X| = ", size_tb_x, ", |SMEM_X| = ", size_tb_ext)
                print (" rev_l_input_idx_right: ", rev_l_input_idx_right)
                print ("========================================================================")
                #
                #   1: [1]
                #   2: [2]
                #
                opt_how_to_manually = 1

                #
                #   [SMEM_X] How to Load Input Tensors on SMEM_X
                #
                int_num_idx_smem_x  = 0
                int_num_idx_tb_x    = len(l_t3_mapping_tb_2D[0])
                for each_idx in rev_l_input_idx_right:
                    if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                        if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:
                            print ("This Index is mapped on SMEM_X Directly: ", each_idx, ", |idx| = ", tc_helper.tc_gen_helper_find(l_t3_slices, each_idx))
                            int_num_idx_smem_x += 1

                #
                #   [TB_X]  Indices mapped on TB_X
                #
                for each_idx in l_t3_mapping_tb_2D[0]:
                    print ("[TB_X] ", each_idx, ", |idx| = ", tc_helper.tc_gen_helper_find(l_t3_slices, each_idx))

                print ("# of External Indices mapped on SMEM_X: ", int_num_idx_smem_x)
                print ("# of External Indices mapped on TB_X: ", int_num_idx_tb_x)

                #
                #   [Manually]
                #   [1] If # of External Indices mapped on SMEM_X == # of Indices mapped on TB_X == 1,
                #   [2] Otherwise,
                #       [2-1] If (# of External Indices mapped on SMEM_X == # of Indices mapped on TB_X) > 1,
                #               : But, non-swappable
                #               : New Temporal Indices 
                #       [2-2] Otherwise, # of External Indices mapped on SMEM_X != # of Indices mapped on TB_X,
                #           [2-2-1] # of Indices mapped on TB_X
                #               : Can be Swappable with something special
                #           [2-2-2] Others
                #               : New Temporal Indices
                #
                
                #
                #   [Question][1] If # of External Indices mapped on SMEM_X == # of Indices mapped on TB_X == 1
                #
                if int_num_idx_smem_x == 1 and int_num_idx_tb_x == 1:
                    print ("[1] If # of External Indices mapped on SMEM_X == # of Indices mapped on TB_X == 1,")
                    #
                    #   (1) |TB_X| = |SMEM_X|   // Should be Swappable (2)
                    #   (2) |TB_X| > |SMEM_X|   // Should handle differently between full-tiles and partial-tiles
                    #   (3) |TB_X| < |SMEM_X|   // Should handle differently between full-tiles and partial-tiles
                    #                           // In For-Statement, there will be multiple-load instructions.
                    #
                    print (" >>> |TB_X| = ", size_tb_x, ", |SMEM_X| = ", size_tb_ext)
                    #
                    #   (2) |TB_X| > |SMEM_X|              
                    #   : Should handle differently between full-tiles and partial-tiles
                    #   : The number of iterations in For-Statement can be reduced if |TB_X| % |SMEM_X| == 0.
                    #       (2-1) Option #1 [Full-Tile]
                    #       (2-2) Option #2 [Partial-Tile]
                    #
                    if size_tb_x > size_tb_ext:
                        print ("|TB_X| > |SMEM_X|")
                        #
                        #   |TB_X| % |SMEM_X| == 0
                        #
                        if size_tb_x % size_tb_ext == 0:
                            print ("|TB_X| % |SMEM_X| == 0")

                        #
                        #   |TB_X| % |SMEM_X| != 0
                        #
                        else:
                            print ("|TB_X| % |SMEM_X| != 0")
                            print (" >>> Not Yet Supported!")
                            opt_modulo = -1

                    #
                    #   (3) |TB_X| < |SMEM_X|  
                    #   : Should handle differently between full-tiles and partial-tiles
                    #   : The number of Load-Instructions can be increasedif |SMEM_X| % |TB_X| == 0.
                    #       (3-1)
                    #
                    else:
                        print ("|TB_X| < |SMEM_X|")
                        #
                        #   |SMEM_X| % |TB_X| == 0
                        #
                        if size_tb_ext % size_tb_x == 0:
                            print ("|SMEM_X| % |TB_X| == 0")
                        #
                        #   |SMEM_X| & |TB_X| != 0
                        #
                        else:
                            print ("|SMEM_X| % |TB_X| != 0")
                            print (" >>> Not Yet Supported!")
                            opt_modulo = -1
                #
                #   [Question][2] Otherwise, we need temporal indices to indicate indices mapped on SMEM_X
                #     
                else:
                    print ("[2] Otherwise, we need temporal indices to indicate indices mapped on SMEM_X")
                    print (" >>> Not Yet Supported!")
                    opt_modulo          = -1
                    opt_how_to_manually = 2

                #
                #   [Solution][1] If |# of External Indices mapped on SMEM_X| == |# of Indices mapped on TB_X| == 1
                #
                if opt_how_to_manually == 1:
                    print ("[Solution][1] If # Ext. Idx. mapped on SMEM_X == # of Ext. Idx. mapped on TB_X == 1")
                    #
                    #
                    #
                    idx_count = 0
                    for each_idx in rev_l_input_idx_right:
                        str_specific_idx = ""
                        #
                        #   For External Indices
                        #
                        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                            #
                            #   Indices mapped on REG
                            #
                            if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) != -1:
                                #
                                #   (Outer-Loop)
                                #
                                if opt_modulo == 1 and opt_full_partial_ext == -1:
                                    str_specific_idx = "(ll * " + str(int(size_tb_x / size_tb_ext)) + ") + (idx_" + l_t3_mapping_tb_2D[0][0] + " / SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + ")"
                                else:
                                    str_specific_idx = "ll"
                                
                            #
                            #   Indices mapped on TB
                            #
                            else:
                                #
                                #
                                #
                                if opt_modulo == 1 and opt_full_partial_ext == -1:
                                    str_specific_idx = "(idx_" + l_t3_mapping_tb_2D[0][0] + " % SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + ")"
                                else:
                                    str_specific_idx = "idx_" + l_t3_mapping_tb_2D[0][0]
                        #
                        #   For Internal Indices,
                        #
                        else:
                            #
                            str_specific_idx = ""
                        
                        #
                        #   Internal Index
                        #   [To-Do] Multiple-Internal Indices
                        #
                        if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) != -1:
                            if idx_count == 0:
                                str_input_addr_right = ""
                            else:
                                str_input_addr_right = "(" + str_input_addr_right + ") * size_" + each_idx
                        #
                        #   External Index
                        #
                        else:
                            print (">>>> ", each_idx, ", str_specific_idx: ", str_specific_idx)
                            if idx_count == 0:
                                str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx
                            else:
                                str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx + " + (" + str_input_addr_right + ") * size_" + each_idx
                            #
                            #
                            # 
                            idx_count = idx_count + 1
                #
                #   [Solution][2] Otherwise, we need temporal indices to indicate indices mapped on SMEM_X
                #
                else:
                    print ("[Solution][2] Temporal Indices to Indicate Indices mapped on SMEM_X (Not Support Yet!)")

                    #
                    #   [2-1] Partially Swapped
                    #
                    if int_num_idx_tb_x >= int_num_idx_smem_x:
                        print ("[Code Generator] Tried to Check if Partially-Swapped is Possible or not.")
                        opt_partially_swapped   = -1    #   -1: false, 0: true (=), 1: true (>)
                        idx_count               = 0
                        for each_idx in rev_l_input_idx_right:
                            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:         # external indices
                                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) == -1:   # mapped on smem_x
                                    print ("[smem] each_idx: ", each_idx, ", ", tc_helper.tc_gen_helper_find(l_t3_slices, each_idx))
                                    print ("[tb_x] ", l_t3_mapping_tb_2D[0][idx_count], ", ", tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][idx_count]))
                                    #
                                    #   First Index   
                                    #
                                    if idx_count == 0:
                                        if tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][idx_count]) >= tc_helper.tc_gen_helper_find(l_t3_slices, each_idx):
                                            #
                                            #   0: true (=)
                                            #
                                            if tc_helper.tc_gen_helper_find(l_t3_slices, each_idx) == tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][idx_count]):
                                                print (" (if) 0: true (=)")
                                                opt_partially_swapped = 0
                                            #
                                            #   1: true (>)
                                            #
                                            else:
                                                print (" (if) 1: true (>)")
                                                opt_partially_swapped = 1
                                            #
                                            list_swappable_pair.append([each_idx, l_t3_mapping_tb_2D[0][idx_count]])
                                    #
                                    #   The Others (opt_partially_swapped == 0)
                                    #
                                    else:
                                        #
                                        #   previous opt_partially_swapped should be "0"
                                        #
                                        if opt_partially_swapped != 0:
                                            opt_partially_swapped = -1
                                        else:
                                            if tc_helper.tc_gen_helper_find(l_t3_slices, each_idx) >= tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][idx_count]):
                                                if tc_helper.tc_gen_helper_find(l_t3_slices, each_idx) == tc_helper.tc_gen_helper_find(l_t3_slices, l_t3_mapping_tb_2D[0][idx_count]):
                                                    opt_partially_swapped = 0
                                                else:
                                                    opt_partially_swapped = 1
                                                #
                                                list_swappable_pair.append([each_idx, l_t3_mapping_tb_2D[0][idx_count]])
                                            else:
                                                opt_partially_swapped = -1

                                    #
                                    idx_count += 1
                        print (">>> opt_partially_swapped: ", opt_partially_swapped)
                        print (">>> list_swappable_pair: ", list_swappable_pair)

                        #
                        #   To Calculate Address
                        #
                        idx_count = 0
                        for each_idx in rev_l_input_idx_right:
                            str_specific_idx = ""
                            #
                            #   >>>> Index Part
                            #
                            #   External Indices
                            #
                            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) == -1:
                                #
                                #   Indices mapped on REG
                                #
                                if tc_helper.tc_gen_helper_find_1d(l_t3_mapping_reg, each_idx) != -1:
                                    #
                                    #   (Outer-Loop)
                                    #
                                    if opt_modulo == 1 and opt_full_partial_ext == -1:
                                        str_specific_idx = "(ll * " + str(int(size_tb_x / size_tb_ext)) + ") + (idx_" + l_t3_mapping_tb_2D[0][0] + " / SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + ")"
                                    else:
                                        str_specific_idx = "ll"
                                #
                                #   Indices mapped on TB
                                #
                                else:
                                    #
                                    #
                                    #
                                    if opt_modulo == 1 and opt_full_partial_ext == -1:
                                        str_specific_idx = "(idx_" + l_t3_mapping_tb_2D[0][0] + " % SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + ")"
                                    else:
                                        str_specific_idx = "idx_" + l_t3_mapping_tb_2D[0][0]
                            #
                            #   Internal Indices
                            #   
                            else:
                                #
                                str_specific_idx = ""

                            #   
                            #   >>>> Address-Part
                            #
                            #   Internal Index
                            #
                            if tc_helper.tc_gen_helper_find_1d(l_internal_idx, each_idx) != -1:
                                print ("[int] ", each_idx)
                                if idx_count == 0:
                                    str_input_addr_right = ""
                                else:
                                    str_input_addr_right = "(" + str_input_addr_right + ") * size_" + each_idx
                            #
                            #   External Index
                            #
                            else:
                                print ("[ext] ", each_idx)
                                if idx_count == 0:
                                    str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx
                                else:
                                    str_input_addr_right = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + " + str_specific_idx + " + (" + str_input_addr_right + ") * size_" + each_idx
                                #
                                idx_count += 1
                    #
                    #   [2-2] Manually
                    #
                    else:
                        print ("[Code Generator] Manually (Not Yet)")
        #
        #
        #
        del l_ext_tb_smem
        del l_tb_idx
    
    #
    #   [Result]
    #
    print ("[Result] str_input_addr_right: ", str_input_addr_right)

    #
    #   >>> 
    #   
    #
    #
    print ("----------------------------------------------------------------------------------------------------------------------------------")

    #
    #   Load Tensor Inputs to sm_b[][]
    #   [1] The FVI in the Input-Right is an Interal Index
    if opt_load_v2 == -1:
        print ("[Option][Load-Input] v2 == -1: TB_X -(loads)-> K (internal) && TB_Y -(loads)-> E (external)")
        print ("[Option][Load-Input] |TB_X| = ", size_tb_x, ", |T_K| = ", size_sm_p7)
        #
        #   |TB_X| < |T_K|
        #
        if size_tb_x < size_sm_p7:
            for inner_step in range(0, int(size_sm_p7 / size_tb_x)):
                f.write("\t\t\tsm_b[threadIdx.x + " + str(int(inner_step * size_tb_x)) + "][threadIdx.y + ll * SIZE_TB_" + str(idx_kernel) + "_Y] = ")
                f.write("dev_" + tensor_contraction[1][3] + "[")

                #
                #
                #
                if opt_pre_computed == -1:
                    f.write(str_input_addr_right)
                else:
                    f.write("dev_" + tensor_contraction[1][3] + "_addr[threadIdx.y + ")
                    f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_Y")
                    f.write(" + blockIdx.x * (" + str_str_v2 + ")]")

                if num_internal_indices > 1:
                    f.write(" + const_internal_" + tensor_contraction[1][3] + "_offset[threadIdx.x + " + str(int(inner_step * size_tb_x)) + " + l]];\n")
                else:
                    if opt_special == 1:
                        f.write(" + (threadIdx.x + " + str(int(inner_step * size_tb_x)) + " + l)]; // 1\n")
                    else:
                        f.write(" + (threadIdx.x + " + str(int(inner_step * size_tb_x)) + " + l) * " + str_stride_int + "]; // 2\n")
        #
        #   |TB_X| >= |T_K|
        #
        else:
            #
            #   |TB_Y| < |E_TB|, 
            #
            if size_tb_y < size_tb_ext:
                for inner_step in range(0, int(size_tb_ext / size_tb_y)):
                    f.write("\t\t\tsm_b[threadIdx.x][threadIdx.y + " + str((int(inner_step * size_tb_y))) + " + ll * SIZE_TB_" + str(idx_kernel) + "_Y * " + str(int(size_tb_ext / size_tb_y)) + "] = ")
                    f.write("dev_" + tensor_contraction[1][3] + "[")

                    #
                    #
                    #
                    if opt_pre_computed == -1:
                        f.write(str_input_addr_right)
                    else:
                        f.write("dev_" + tensor_contraction[1][3] + "_addr[threadIdx.y + " + str((int(inner_step * size_tb_y))) + " + ")
                        f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_Y * " + str(int(size_tb_ext / size_tb_y)))
                        f.write(" + blockIdx.x * (" + str_str_v2 + ")]")

                    if num_internal_indices > 1:
                        f.write(" + const_internal_" + tensor_contraction[1][3] + "_offset[threadIdx.x + l]];\n")
                    else:
                        if opt_special == 1:
                            f.write(" + (threadIdx.x + l)]; // FVI = Int.\n")
                        else:
                            #f.write(" + (threadIdx.x + l) * " + tensor_contraction[1][1] + "];\n")
                            f.write(" + (threadIdx.x + l) * " + str_stride_int + "]; // FVI = Ext.\n")    

            else:
                f.write("\t\t\tsm_b[threadIdx.x][threadIdx.y + ll * SIZE_TB_" + str(idx_kernel) + "_Y] = ")
                f.write("dev_" + tensor_contraction[1][3] + "[")

                #
                #
                #
                if opt_pre_computed == -1:
                    f.write(str_input_addr_right)
                else:
                    f.write("dev_" + tensor_contraction[1][3] + "_addr[threadIdx.y + ")
                    f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_Y")
                    f.write(" + blockIdx.x * (" + str_str_v2 + ")]")

                if num_internal_indices > 1:
                    f.write(" + const_internal_" + tensor_contraction[1][3] + "_offset[threadIdx.x + l]];\n")
                else:
                    if opt_special == 1:
                        f.write(" + (threadIdx.x + l)]; // FVI = Int.\n")
                    else:
                        #f.write(" + (threadIdx.x + l) * " + tensor_contraction[1][1] + "];\n")
                        f.write(" + (threadIdx.x + l) * " + str_stride_int + "]; // FVI = Ext.\n")
    #
    #   [2] The FVI in the Input-Right is an External Index
    #   [Write CUDA Kernel]
    #
    else:
        print ("[Option][Load-Input] v2 == 1: TB_X -(loads)-> E (external) && TB_Y -(loads)-> K (internal)")
        print ("[Option][Load-Input] |TB_Y| = ", size_tb_y, " ?? |K| = ", size_sm_p7)
        #
        #   |TB_Y| < |K|
        #   : Need to Load |K| / |TB_Y| times in the loop.
        #
        if size_tb_y < size_sm_p7:
            print ("[Option][Load-Input] |TB_Y| < |K| >> [Solution] Need to Load |K|/|TB_Y| times in the loop")
            print ("[Option][Load-Input] opt_full_partial_ext: ", opt_full_partial_ext) # ???
            print ("[Option][Load-Input] Modulo-Option: ", opt_modulo)  #   ??
            #
            #   [Modulo]
            #       : Full-Tile for External Indices
            #   [1] "opt_modulo == 1":  
            #
            if opt_modulo == 1:# and opt_full_partial_ext == -1:
                print ("[Solution] Modulo is possible and Full-Tile")
                #
                #
                #
                if opt_full_partial_ext == 1 and method_load_v2 > 2:
                    f.write("\t\t\tif (threadIdx.x < " + str(size_tb_ext) + ")\n")
                    f.write("\t\t\t{\n")
                #
                #   Inner-Steps for Internal Indices
                #
                for inner_step in range(0, int(size_sm_p7 / size_tb_y)):
                    #
                    #   |TB_X| == |SMEM_X|
                    #
                    if size_tb_ext == size_tb_x:
                        f.write("\t\t\tsm_b[threadIdx.y + " + str(int(inner_step * size_tb_y)) + "][threadIdx.x + ll * SIZE_TB_" + str(idx_kernel) + "_X] = ")
                    #
                    #   |TB_X| != |SMEM_X|
                    #
                    else:
                        #
                        #   [Ext.] Full-Tile
                        #
                        if opt_full_partial_ext == -1:
                            f.write("\t\t\tsm_b[threadIdx.y + " + str(int(inner_step * size_tb_y)) + "][threadIdx.x + ll * SIZE_TB_" + str(idx_kernel) + "_X] = ")
                        #
                        #   [Ext.] Partial-Tile
                        #
                        else:
                            f.write("\t\t\tsm_b[threadIdx.y + " + str(int(inner_step * size_tb_y)) + "][threadIdx.x + ll * " + str(size_tb_ext) + "] = ")
                    f.write("dev_" + tensor_contraction[1][3] + "[")

                    #
                    #   [External Index]
                    #
                    if opt_pre_computed == -1:
                        f.write(str_input_addr_right)
                    else:
                        f.write("dev_" + tensor_contraction[1][3] + "_addr[threadIdx.x + ")
                        f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_X")
                        f.write(" + blockIdx.x * (" + str_str_v2 + ")]")
                    #
                    #   [Internal Index]
                    #
                    if num_internal_indices > 1:
                        f.write(" + const_internal_" + tensor_contraction[1][3] + "_offset[threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l]]; // 5\n")
                    else:
                        if opt_special == 1:
                            f.write(" + (threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l)]; // 3\n")
                        else:
                            f.write(" + (threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l) * " + str_stride_int + "]; // 4\n")
                #
                #
                #
                if opt_full_partial_ext == 1 and method_load_v2 > 2:
                    f.write("\t\t\t}\n")
            #
            #   [2] "opt_modulo == -1": 
            #
            else:
                print ("[Solution] Modulo is impossible or Partial-Tile")
                print ("[Solution] method_load_v2: ", method_load_v2)
                print ("[Solution] opt_full_partial_ext: ", opt_full_partial_ext)
                #
                #   Partially Swappable
                #
                opt_num_tabs = 3
                if opt_partially_swapped == 1:
                    print ("[Option][Load-Input] partially swapped == 1")
                    print ("[Option][Load-Input] opt_full_partial_ext: ", opt_full_partial_ext, ", method_load_v2: ", method_load_v2)
                    if opt_full_partial_ext == 1 and method_load_v2 > 2:
                        #
                        #
                        #
                        tc_helper.tc_gen_helper_code_a_line(f, opt_num_tabs, "if (threadIdx.x < " + str(size_tb_ext) + ")", 1)
                        tc_helper.tc_gen_helper_code_a_line(f, opt_num_tabs, "{", 1)
                        opt_num_tabs += 1

                    #
                    #   Inner-Steps for Internal Indices: |K| / |TB_Y|
                    #
                    for inner_step in range(0, int(size_sm_p7 / size_tb_y)):
                        #
                        #   |TB_X| == |SMEM_X|
                        #
                        if size_tb_ext == size_tb_x:
                            f.write("\t\t\t// |TB_X| == |SMEM_X|\n")
                            f.write("\t\t\tsm_b[threadIdx.y + " + str(int(inner_step * size_tb_y)) + "][threadIdx.x + ll * SIZE_TB_" + str(idx_kernel) + "_X] = ")
                        else:
                            #
                            #   [Ext.] Full-Tile
                            #
                            if opt_full_partial_ext == -1:
                                tc_helper.tc_gen_helper_code_a_line(f, opt_num_tabs, "sm_b[threadIdx.y + " + str(int(inner_step * size_tb_y)) + "][threadIdx.x + ll * SIZE_TB_" + str(idx_kernel) + "_X] = ", -1)
                            #
                            #   [Ext.] Partial-Tile
                            #
                            else:
                                tc_helper.tc_gen_helper_code_a_line(f, opt_num_tabs, "sm_b[threadIdx.y + " + str(int(inner_step * size_tb_y)) + "][threadIdx.x + ll * " + str(size_tb_ext) + "] = ", -1)
                        
                        #
                        #   
                        #
                        tc_helper.tc_gen_helper_code_a_line(f, 0, "dev_" + tensor_contraction[1][3] + "[", -1)
                        #
                        #   [External Index]
                        #   [Option] With Pre-Computations
                        #
                        if opt_pre_computed == -1:
                            tc_helper.tc_gen_helper_code_a_line(f, 0, str_input_addr_right, -1)
                        #
                        #   [Option] Without Pre-Computations
                        #
                        else:
                            tc_helper.tc_gen_helper_code_a_line(f, 0, "dev_" + tensor_contraction[1][3] + "_addr[threadIdx.x + ", -1)
                            tc_helper.tc_gen_helper_code_a_line(f, 0, "ll * " + "SIZE_TB_" + str(idx_kernel) + "_X", -1)
                            tc_helper.tc_gen_helper_code_a_line(f, 0, " + blockIdx.x * (" + str_str_v2 + ")]", -1)
                        #
                        #   [Internal Index]
                        #   [Option] Multiple Internal Indices
                        #
                        if num_internal_indices > 1:
                            f.write(" + const_internal_" + tensor_contraction[1][3] + "_offset[threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l]]; // 7\n")
                        #
                        #   [Option] Single Internal Index
                        #
                        else:
                            #
                            #   [Option] The Internal Index is the FVI.
                            #
                            if opt_special == 1:
                                tc_helper.tc_gen_helper_code_a_line(f, 0, " + (threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l)]; // 5", 1)
                            #
                            #   [Option] The Internal Index is not the FVI, requiring its computed strides.
                            #
                            else:                            
                                tc_helper.tc_gen_helper_code_a_line(f, 0, " + (threadIdx.y + " + str(int(inner_step * size_tb_y)) + " + l) * " + str_stride_int + "]; // 6", 1)
                    #
                    if opt_full_partial_ext == 1 and method_load_v2 > 2:
                        f.write("\t\t\t}\n")
                #
                #   Totally Manually
                #
                else:
                    print ("[opt] partially swapped != 1")
                    f.write("\t\t\t// temp\n")
        #
        #   |TB_Y| >= |K|
        #
        else:
            print ("[Option] |TB_Y| >= |K|")
            #
            #   ???
            #
            if size_tb_x < size_tb_ext:
                for inner_step in range(0, int(size_tb_ext / size_tb_x)):
                    f.write("\t\t\tsm_b[threadIdx.y][threadIdx.x + (ll * " + str(int(size_tb_ext / size_tb_x)) + " + " + str(inner_step) + ") * SIZE_TB_" + str(idx_kernel) + "_X] = ")
                    f.write("dev_" + tensor_contraction[1][3] + "[")
                    
                    #
                    #
                    #
                    if opt_pre_computed == -1:
                        f.write(str_input_addr_right)
                    else:
                        f.write("dev_" + tensor_contraction[1][3] + "_addr[threadIdx.x + ")
                        f.write("(ll * " + str(int(size_tb_ext / size_tb_x)) + " + " + str(inner_step) + ") * " + "SIZE_TB_" + str(idx_kernel) + "_X")
                        f.write(" + blockIdx.x * (" + str_str_v2 + ")]")

                    #
                    #
                    #
                    if num_internal_indices > 1:
                        f.write(" + const_internal_" + tensor_contraction[1][3] + "_offset[threadIdx.y + l]];\n")
                    else:
                        if opt_special == 1:
                            f.write(" + (threadIdx.y + l)];\n")
                        else:
                            f.write(" + (threadIdx.y + l) * " + tensor_contraction[1][1] + "];\n")
            else:
                f.write("\t\t\tsm_b[threadIdx.y][threadIdx.x + ll * SIZE_TB_" + str(idx_kernel) + "_X] = ")
                f.write("dev_" + tensor_contraction[1][3] + "[")

                #
                #
                #
                if opt_pre_computed == -1:
                    f.write(str_input_addr_right)
                else:
                    f.write("dev_" + tensor_contraction[1][3] + "_addr[threadIdx.x + ")
                    f.write("ll * " + "SIZE_TB_" + str(idx_kernel) + "_X")
                    f.write(" + blockIdx.x * (" + str_str_v2 + ")]")
                #
                #
                #
                if num_internal_indices > 1:
                    f.write(" + const_internal_" + tensor_contraction[1][3] + "_offset[threadIdx.y + l]];\n")
                else:
                    if opt_special == 1:
                        f.write(" + (threadIdx.y + l)];//666\n")
                    else:
                        #f.write(" + (threadIdx.y + l) * " + tensor_contraction[1][1] + "];\n")
                        f.write(" + (threadIdx.y + l) * " + str_stride_int + "]; // 555\n")
    #
    print ("[Code Generator][Kernel][Load Input-Right] End")
    print ("===============================================================================================")




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
#           : 
#       2.2. |TB_X| > |K|
#           :
#       2.3. |TB_X| < |K|
#           :
#       2.4. |TB_Y| = |E|
#           :
#       2.5. |TB_Y| > |E|
#           :
#       2.6. |TB_Y| < |E|
#           :
#
#   3. For 1.2. case, TB_X -(loads)-> E && TB_Y -(loads)-> K, where E \ REG
#
