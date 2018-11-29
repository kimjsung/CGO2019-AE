import src.generators.tc_helper                 as tc_helper
import src.codes.tc_code_etc                    as tc_code_etc
import src.codes.kernels.helper_interface       as helper_interface
import src.codes.others.tc_pre_CUDA_Malloc      as tc_pre_CUDA_Malloc 
import src.codes.others.tc_post_HostDevice_Free as tc_post_HostDevice_Free

#
#   A Caller for Kernel(s)
#
def tc_gen_code_kernel_caller(f,    interface_name,     kernel_name,        l_interface_info,       
                                    l_external_index,   l_internal_index,
                                    l_mapping_TB_K,
                                    l_var_thread_block, l_var_outputs,      l_var_outputs_helpers,  l_var_input_left,   l_var_input_right, l_var_input_internal,
                                    l_combined_var_input_left, l_combined_var_input_right, l_combined_var_outputs_helpers, l_combined_var_thread_block, l_combined_register_mappings,
                                    l_internal_addrs,
                                    l_combined_internal_addrs,
                                    l_cuda_malloc,      l_device_dynamic,   l_host_dynamic,
                                    l_t3_parameters,    l_t2_parameters,    l_v2_parameters,
                                    l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters,
                                    opt_pre_computed, opt_data_type):
    #
    #   [0] Header
    #     
    tc_gen_code_interface_Header(f, interface_name, l_interface_info, opt_data_type)
    f.write("{\n")

    #
    #   [1] Variables Used in the Caller
    #
    tc_gen_code_interface_Variables(f, l_combined_var_thread_block, l_var_outputs, l_combined_var_outputs_helpers, l_combined_var_input_left, l_combined_var_input_right, l_var_input_internal, opt_pre_computed)
   
    #
    if opt_pre_computed != -1:
        #
        #   [2] Tile-Arroaches
        #
        tc_gen_code_interface_TileApproach(f, l_combined_var_outputs_helpers, l_interface_info)
        
        #
        #   [3] Pre-Computed and In-Direction Arrays 
        #
        tc_gen_code_interface_PreComputedArrays(f, l_combined_var_outputs_helpers, l_combined_var_input_left, l_combined_var_input_right, l_interface_info)
    else:
        #
        #   # of Thread Blocks
        #
        str_num_thread_blocks   = ""
        idx_count               = 0
        for each_idx in l_external_index:
            if idx_count == 0:
                str_num_thread_blocks = "CEIL(size_" + each_idx + ", SIZE_SLICE_1_" + each_idx.capitalize() + ")"
            else:
                str_num_thread_blocks = str_num_thread_blocks + " * CEIL(size_" + each_idx + ", SIZE_SLICE_1_" + each_idx.capitalize() + ")"
            idx_count = idx_count + 1

        f.write("\t" + l_combined_var_thread_block[0][0][1] + " = " + str_num_thread_blocks + ";\n")
    
    #
    #   [4] cudaMalloc & cudaMemcpy
    #
    tc_pre_CUDA_Malloc.tc_gen_code_driver_CUDA_Malloc(f, l_cuda_malloc, l_device_dynamic, opt_pre_computed)

    #
    #   [5] Related to Kernels
    #
    tc_gen_code_interface_RelatedKernels(f, l_combined_var_input_left, l_combined_var_thread_block, l_external_index, l_internal_index, l_combined_register_mappings, l_internal_addrs, l_combined_internal_addrs, l_mapping_TB_K, opt_data_type)
    

    #
    #   Need One More Condition to Check if Register Transpose is Possible or not.
    #
    #
    #   [6] Decision Tree for Kernel Types
    #
    #tc_gen_code_interface_Force_Partial_Kernel(f, kernel_name, l_combined_var_thread_block, l_external_index, l_internal_index,
    #                                    l_combined_register_mappings, l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs, opt_pre_computed)

    tc_gen_code_interface_DecisionTree(f, kernel_name, l_combined_var_thread_block, l_external_index, l_internal_index,
                                        l_combined_register_mappings, l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs, opt_pre_computed)
   
    #
    #   [7] Copy the Result from Device to Host
    #
    tc_gen_code_interface_MemcpyOutput(f, l_external_index, opt_data_type)

    #
    #   [8] CudaFree
    #  
    tc_gen_code_interface_CUDAFree(f, l_cuda_malloc, opt_pre_computed) 

    #
    #   [9] HostFree
    #
    print (">>>>>>> l_host_dynamic: ", l_host_dynamic)
    f.write("\t// Shoule be Fixed\n")
    tc_gen_code_interface_HostFree(f, l_host_dynamic)

    #
    #   End of the Caller
    #
    f.write("}\n")

#
#   [0] Header
#
def tc_gen_code_interface_Header(f, interface_name, l_interface_info, opt_data_type):
    #
    f.write("\n")
    f.write("// written by tc_interface.tc_gen_code_interface_Header()\n")
    f.write("extern \"C\"\n")
    f.write("void " + interface_name + "(")

    #
    #   l_interface_info: [0] All Index, [1] Output, [2] Inputs, [3] Conditions, [4] Options
    #   If the code generator supports multiple-output-groups, this should be corrected.
    #
    list_info_split_indices = l_interface_info[0][5]

    #   [0] All Index
    idx_count = 0
    for each_index in l_interface_info[0][0]:
        if idx_count == 0:        
            f.write("int size_" + each_index)
        else:
            f.write(", int size_" + each_index)
        idx_count = idx_count + 1
    
    #   [1] Output
    if opt_data_type == "DOUBLE":
        f.write(", double* " + l_interface_info[0][1])
    else:
        f.write(", float* " + l_interface_info[0][1])

    #   [2] Inputs
    for each_pair_inputs in l_interface_info[0][2]:
        for each_input in each_pair_inputs:
            if opt_data_type == "DOUBLE":
                f.write(", double* host_" + each_input)
            else:
                f.write(", float* host_" + each_input)
    
    #   [3] Conditions
    for each_condition in l_interface_info[0][3]:
        f.write(", int " + each_condition)

    #   [4] Option(s): (Currently) Only One Option for Register Transpose
    f.write(", int " + l_interface_info[0][4])

    f.write(")\n")

#
#   [1] Variables
#
def tc_gen_code_interface_Variables(f, l_combined_var_thread_block, l_var_outputs, l_combined_var_outputs_helpers, l_combined_var_input_left, l_combined_var_input_right, l_var_input_internal, opt_pre_computed):
     #   1. The Number of Thread Blocks per Kernel
    for each_inner_group in l_combined_var_thread_block:
        for each_var in each_inner_group:
            f.write("\t" + each_var[0] + " " + each_var[1] + ";")
        f.write("\n")
    f.write("\n")

    #   2. Outputs
    for each_var in l_var_outputs:
        f.write("\t" + each_var[0] + " " + each_var[1] + ";")
    f.write("\n")

    #   3. Outputs-Helpers
    if opt_pre_computed != -1:
        for each_inner_group in l_combined_var_outputs_helpers:
            for each_var in each_inner_group:
                f.write("\t" + each_var[0] + " " + each_var[1] + ";")
            f.write("\n")
        f.write("\n")

    #   4. Input-Left
    for each_inner_group in l_combined_var_input_left:
        for each_var in each_inner_group:
            if opt_pre_computed == -1:
                if "addr" in each_var[1]:
                    continue
                if "offset" in each_var[1]:
                    continue
                #
                f.write("\t" + each_var[0] + " " + each_var[1] + ";")
            else:
                f.write("\t" + each_var[0] + " " + each_var[1] + ";")
        f.write("\n")

    #   5. Input-Right
    for each_inner_group in l_combined_var_input_right:
        for each_var in each_inner_group:
            if opt_pre_computed == -1:
                if "addr" in each_var[1]:
                    continue
                if "offset" in each_var[1]:
                    continue
                #
                f.write("\t" + each_var[0] + " " + each_var[1] + ";")
            else:
                f.write("\t" + each_var[0] + " " + each_var[1] + ";")
        f.write("\n")
    f.write("\n")

    #   6. Input-Internal
    for each_var in l_var_input_internal:
        f.write("\t" + each_var[0] + " " + each_var[1] + ";\n")
    f.write("\n")

#
#   [2] (Pre-Computed Part) Tile-Approach
#
def tc_gen_code_interface_TileApproach(f, l_combined_var_outputs_helpers, l_interface_info):
    f.write("\t// Tile-Approaches\n")
    idx_count = 1
    for each_inner_group in l_combined_var_outputs_helpers:
        f.write("\tpre_TileApproach_" + str(idx_count) + "(")

        #   Parameters [1]
        var_count = 0
        for each_var in l_combined_var_outputs_helpers[idx_count - 1]:
            if "host" in each_var[1]:
                if "block" in each_var[1]:
                    if var_count == 0:
                        f.write(each_var[1])
                    else:
                        f.write(", " + each_var[1])
                    var_count = var_count + 1
        
        #   Parameters [2]
        f.write(", &num_thread_blocks_kernel_" + str(idx_count))

        #   Parameters [3]
        for each_index in l_interface_info[0][0]:
            f.write(", size_" + each_index)
        
        f.write(");\n")
        idx_count = idx_count + 1
    f.write("\n")

#
#   [3] (Pre-Computed Part) Arrays
#
def tc_gen_code_interface_PreComputedArrays(f, l_combined_var_outputs_helpers, l_combined_var_input_left, l_combined_var_input_right, l_interface_info):
    f.write("\t// Pre-Computed and In-Direction Arrays\n")
    idx_count = 1
    for each_inner_group in l_combined_var_outputs_helpers:
        f.write("\tpre_PreComputedArray_" + str(idx_count) + "(")

        #   Parameters [1]
        var_count = 0
        for each_var in l_combined_var_outputs_helpers[idx_count - 1]:
            if "host" in each_var[1]:
                if not "range" in each_var[1]:
                    if var_count == 0:
                        f.write(each_var[1])
                    else:
                        f.write(", " + each_var[1])
                    var_count = var_count + 1

        #   Parameters [2]
        f.write(", num_thread_blocks_kernel_" + str(idx_count))

        #   Parameters [3]
        for each_var in l_combined_var_input_left[idx_count - 1]:
            if "host" in each_var[1]:
                f.write(", " + each_var[1])
        
        #   Parameters [4]
        for each_var in l_combined_var_input_right[idx_count - 1]:
            if "host" in each_var[1]:
                f.write(", " + each_var[1])

        #   Parameters [5]
        for each_index in l_interface_info[0][0]:
            f.write(", size_" + each_index)

        f.write(");\n")
        idx_count = idx_count + 1
    f.write("\n")

#
#   [5] Related Kernels
#
def tc_gen_code_interface_RelatedKernels(f, l_combined_var_input_left, l_combined_var_thread_block, l_external_index, l_internal_index, l_combined_register_mappings, l_internal_addrs, l_combined_internal_addrs, l_mapping_TB_K, opt_data_type):
    f.write("\t// Related to Kernels\n")
    f.write("\t// There are " + str(len(l_combined_var_input_left)) + " Basic Kernels\n")

    #
    str_operations = ""
    idx_count = 0
    for each_idx in l_external_index:
        if idx_count == 0:
            str_operations = "size_" + each_idx
        else:
            str_operations = str_operations + " * size_" + each_idx
        idx_count += 1
    
    # assumption: # of external index > 0
    for each_idx in l_internal_index:
        str_operations = "(long long int)(" + str_operations + ") * size_" + each_idx

    #
    for idx_kernel in range(1, len(l_combined_var_thread_block) + 1):
        if opt_data_type == "DOUBLE":
            f.write("\tlong long int tmp_operations = 2 * " + str_operations + ";\n")
        else:
            f.write("\tlong long int tmp_operations = " + str_operations + ";\n")
        #
        f.write("\tprintf (\"========================================= fusedKernels =============================================\\n\");\n")
        f.write("\tprintf (\"\t\tGrid Size  : %6d (1D)\\n\", num_thread_blocks_kernel_" + str(idx_kernel) + ");\n")
        f.write("\tprintf (\"\t\tBlock-size : %2d, %2d (2D)\\n\", SIZE_TB_" + str(idx_kernel) + "_X, SIZE_TB_" + str(idx_kernel) + "_Y);\n")
        f.write("\tprintf (\"\t\tReg.-size  : %2d, %2d (2D)\\n\", SIZE_REG_" + str(idx_kernel) + "_X, SIZE_REG_" + str(idx_kernel) + "_Y);\n")
        f.write("\tprintf (\"\t\tA thread deals with (%d x %d) elements (basically)\\n\", SIZE_TB_" + str(idx_kernel) + "_X * SIZE_REG_" + str(idx_kernel) + "_X, SIZE_TB_" + str(idx_kernel) + "_Y * SIZE_REG_" + str(idx_kernel) + "_Y);\n")
        f.write("\tprintf (\"\t\t# of Operations: %lld\\n\", tmp_operations);\n")
        f.write("\tprintf (\"====================================================================================================\\n\");\n")

    #
    for idx_kernel in range(1, len(l_combined_var_thread_block) + 1):
        #   Grid-Size
        f.write("\tdim3 gridsize_" + str(idx_kernel) + "(num_thread_blocks_kernel_" + str(idx_kernel) + ");\n")

        #   Block-Size
        f.write("\tdim3 blocksize_" + str(idx_kernel) + "(SIZE_TB_" + str(idx_kernel) + "_X, SIZE_TB_" + str(idx_kernel) + "_Y);\n")
        f.write("\n")

    #
    #   Strides for Output's Address
    #
    idx_count       = 0
    str_prev_idx    = ""
    for each_ext_idx in l_external_index:
        if idx_count == 0:
            f.write("\tint stride_output_" + each_ext_idx + " = 1;\n")
        else:
            f.write("\tint stride_output_" + each_ext_idx + " = stride_output_" + str_prev_idx + " * size_" + str_prev_idx + ";\n")
        str_prev_idx    = each_ext_idx
        idx_count       = idx_count + 1
    f.write("\n")

    #
    #   Strides for Register Tiles from The Strides
    #
    idx_count = 1
    for each_mapping in l_combined_register_mappings:
        f.write("\tint stride_reg_x_" + str(idx_count) + " = stride_output_" + each_mapping[0] + ";\n")
        f.write("\tint stride_reg_y_" + str(idx_count) + " = stride_output_" + each_mapping[1] + ";\n")
        idx_count = idx_count + 1
    f.write("\n")

    #
    #
    #
    str_prod_internal   = ""
    idx_count           = 0
    for each_idx in l_internal_index:
        if idx_count == 0:
            str_prod_internal = "size_" + each_idx
        else:
            str_prod_internal = str_prod_internal + " * size_" + each_idx
        idx_count = idx_count + 1
    f.write("\tint size_internal = " + str_prod_internal + ";\n")
    f.write("\n")

    #
    #   |K| > 1, To-Do: Multiple Tensor Contractions
    #
    if len(l_internal_index) > 1:
        l_rev_internal_index = list(reversed(l_internal_index))
        f.write("\t// (manually) " + str(l_mapping_TB_K) + "\n")
        f.write("\thost_internal_left_offset \t= (int*)malloc(sizeof(int) * size_internal);\n")
        f.write("\thost_internal_right_offset \t= (int*)malloc(sizeof(int) * size_internal);\n")


        #
        for each_int_idx in l_rev_internal_index:
            f.write("\tfor (int idx_" + each_int_idx + " = 0; idx_" + each_int_idx + " < size_" + each_int_idx + "; idx_" + each_int_idx + "++)\n")

        #
        str_idx_common  = ""
        str_idx_prev    = ""
        idx_first       = 0
        for each_int_idx in l_mapping_TB_K:
            if idx_first == 0:
                str_idx_common  = "idx_" + each_int_idx
                str_idx_prev    = each_int_idx
                idx_first       = 1
            else:
                str_idx_common  = str_idx_common + " + (idx_" + each_int_idx + ") * size_" + str_idx_prev
                str_idx_prev    = each_int_idx
        #
        f.write("\t{\n")

        # "l_internal_addrs" has currently one list.
        #   address_offset
        str_addr_left    = l_internal_addrs[0][0]
        str_addr_right   = l_internal_addrs[0][1]

        #
        f.write("\t\thost_internal_left_offset[" + str_idx_common + "] \t= " + str_addr_left + ";\n")
        f.write("\t\thost_internal_right_offset[" + str_idx_common + "] \t= " + str_addr_right + ";\n")
        
        #
        f.write("\t}\n")
        f.write("\n")

        #
        f.write("\tcudaMemcpyToSymbol(const_internal_t2_offset, host_internal_left_offset, sizeof(int) * size_internal);\n")
        f.write("\tcudaMemcpyToSymbol(const_internal_v2_offset, host_internal_right_offset, sizeof(int) * size_internal);\n")
        f.write("\n")

        #
        f.write("\tint* dev_internal_offset_t2;\n")
        f.write("\tint* dev_internal_offset_v2;\n")

        #
        f.write("\t// cudaMalloc()\n")
        tc_pre_CUDA_Malloc.tc_gen_code_helper_cudaMalloc(f, "dev_internal_offset_t2", "int", "size_internal")
        tc_pre_CUDA_Malloc.tc_gen_code_helper_cudaMalloc(f, "dev_internal_offset_v2", "int", "size_internal")
        f.write("\n")

        #
        f.write("\t// cudaMemcpy()\n")
        # tc_gen_code_helper_cudaMemcpy(f, host_device[1], host_device[2], host_device[0], host_device[3], 1)
        tc_pre_CUDA_Malloc.tc_gen_code_helper_cudaMemcpy(f, "dev_internal_offset_t2", "host_internal_left_offset", "int","size_internal", 1)
        tc_pre_CUDA_Malloc.tc_gen_code_helper_cudaMemcpy(f, "dev_internal_offset_v2", "host_internal_right_offset", "int","size_internal", 1)
        
    #
    #   |K| == 1,
    #
    elif len(l_internal_index) == 1:
        for each_input_group in l_combined_internal_addrs:
            for each_input in each_input_group:
                #
                #
                #
                f.write("\tint " + each_input[0] + " = " + each_input[1] + ";\n")
                f.write("\tint " + each_input[2] + " = " + each_input[3] + ";\n")
    #
    f.write("\n")

#
#
#
def tc_gen_code_interface_Force_Partial_Kernel(f, kernel_name, 
                                                l_combined_var_thread_block, l_external_index, l_internal_index, 
                                                l_combined_register_mappings, l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs, 
                                                opt_pre_computed):
    f.write("\t// New Caller\n")
    '''
    for each_inner_group in range(1, len(l_combined_var_thread_block) + 1):
        steps_tab = 1
        helper_interface.call_kernel(f, steps_tab, kernel_name + "_4_" + str(each_inner_group), each_inner_group, 
        l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
        l_external_index, l_internal_index,
        0, opt_pre_computed)
    f.write("\n")
    '''
    for each_inner_group in range(1, len(l_combined_var_thread_block) + 1):
        if len(l_internal_index) > 1:
            f.write("\tif (size_internal > MAX_CONST_LEN)\n")
            f.write("\t{\n")

            steps_tab = 2
            #helper_interface.call_kernel(f, steps_tab, kernel_name + "_1_tex_" + str(each_inner_group), each_inner_group, 
            helper_interface.call_kernel(f, steps_tab, kernel_name + "_4_tex_" + str(each_inner_group), each_inner_group, 
            l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
            l_external_index, l_internal_index,
            1, opt_pre_computed)

            f.write("\t}\n")
            f.write("\telse\n")

            f.write("\t{\n")
            
            
            #helper_interface.call_kernel(f, steps_tab, kernel_name + "_1_" + str(each_inner_group), each_inner_group, 
            helper_interface.call_kernel(f, steps_tab, kernel_name + "_4_" + str(each_inner_group), each_inner_group, 
            l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
            l_external_index, l_internal_index,
            0, opt_pre_computed)

            f.write("\t}\n")
        else:
            steps_tab = 1
            
            #helper_interface.call_kernel(f, steps_tab, kernel_name + "_1_" + str(each_inner_group), each_inner_group, 
            helper_interface.call_kernel(f, steps_tab, kernel_name + "_4_" + str(each_inner_group), each_inner_group, 
            l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
            l_external_index, l_internal_index,
            0, opt_pre_computed)
    f.write("\n")

#
#   [6] Decision Tree
#
def tc_gen_code_interface_DecisionTree(f, kernel_name, l_combined_var_thread_block, l_external_index, l_internal_index, 
                                        l_combined_register_mappings, l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs, 
                                        opt_pre_computed):
    
    f.write("\t// Decision Tree for Kernel Types\n")
    if len(l_combined_var_thread_block) > 1:
        #
        #   
        #
        f.write("\tif (opt_register_transpose == 0)\n")
        f.write("\t{\n")
        f.write("\t\t// [1] Register Transpose: OFF\n")

        #   For Each Inner-Group, 
        for each_inner_group in range(1, len(l_combined_var_thread_block) + 1):
            #
            f.write("\t\tif (")

            idx_count = 0
            for ext_idx in l_external_index:
                if idx_count == 0:
                    f.write(    "size_" + ext_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + ext_idx.capitalize() + " == 0") 
                else:
                    f.write(" && size_" + ext_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + ext_idx.capitalize() + " == 0")
                idx_count = idx_count + 1

            f.write(")\n")
            f.write("\t\t{\n")
            f.write("\t\t\t// [2] Extenral Index: Full\n")
            
            #
            f.write("\t\t\tif (")

            idx_count = 0
            for int_idx in l_internal_index:
                if idx_count == 0:
                    f.write(    "size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                else:
                    f.write(" && size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                idx_count = idx_count + 1

            f.write(")\n")
            f.write("\t\t\t{\n")
            f.write("\t\t\t\t// [3] Internal Index: Full\n")
            #
            #
            #
            f.write("\t\t\t\t// >>> External: Full && Internal: Full\n")
            f.write("\t\t\t\tprintf (\"External: Full, Internal: Full\\n\");\n")
            #
            #
            #
            f.write("\t\t\t\tif (size_internal > 8192)\n")
            f.write("\t\t\t\t\tprintf (\"size_internal: %d\n\", size_internal)\"\n;")




            #for each_inner_group in range(1, len(l_combined_register_mappings) + 1):
            f.write("\t\t\t\t// kernel_" + str(each_inner_group) + "\n")
            f.write("\t\t\t\t" + kernel_name + "_" + str(each_inner_group) + "<<<")
            f.write("gridsize_" + str(each_inner_group) + ", blocksize_" + str(each_inner_group) + ">>>")
            f.write("(")

            #   related to "output"
            par_count = 0
            for each_par in l_combined_t3_parameters[each_inner_group - 1]:
                print ("t3: ", each_par)
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1

            #   related to "left"
            for each_par in l_combined_t2_parameters[each_inner_group - 1]:
                print ("t2: ", each_par)
                f.write(", " + each_par[0])

            #   related to "right"
            for each_par in l_combined_v2_parameters[each_inner_group - 1]:
                print ("v2: ", each_par)
                f.write(", " + each_par[0])
            
            #
            f.write(", stride_reg_x_" + str(each_inner_group))
            f.write(", stride_reg_y_" + str(each_inner_group))

            #
            f.write(", size_internal")

            f.write(");\n")
            f.write("\n")
                #idx_count = idx_count + 1
            #
            #
            #
            f.write("\t\t\t}\n")
            f.write("\t\t\telse\n")
            f.write("\t\t\t{\n")
            f.write("\t\t\t\t// [4] Internal Index: Partial\n")
            #
            #
            #
            f.write("\t\t\t\t// >>> External: Full && Internal: Partial\n")
            f.write("\t\t\t\tprintf (\"External: Full, Internal: Partial\\n\");\n")
            #
            #
            #
            #idx_count = 1
            #for each_inner_group in range(1, len(l_combined_register_mappings) + 1):
            f.write("\t\t\t\t// kernel_" + str(each_inner_group) + "\n")
            f.write("\t\t\t\t" + kernel_name + "_1_" + str(each_inner_group) + "<<<")
            f.write("gridsize_" + str(each_inner_group) + ", blocksize_" + str(each_inner_group) + ">>>")
            f.write("(")

            #   related to "output"
            par_count = 0
            for each_par in l_combined_t3_parameters[each_inner_group - 1]:
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1

            #   related to "left"
            for each_par in l_combined_t2_parameters[each_inner_group - 1]:
                f.write(", " + each_par[0])

            #   related to "right"
            for each_par in l_combined_v2_parameters[each_inner_group - 1]:
                f.write(", " + each_par[0])

            #
            #   "opt_pre_computed" (temporally)
            #
            if opt_pre_computed == -1:
                for each_idx in l_interface_info[0][0]:
                    print ("each_idx: ", each_idx)
                f.write(", size_h3")
                f.write(", size_h2")
                f.write(", size_h1")
                f.write(", size_p6")
                f.write(", size_p5")
                f.write(", size_p4")
                f.write(", size_h7")

                f.write(", CEIL(size_h3, SIZE_SLICE_1_H3)")
                f.write(", CEIL(size_h2, SIZE_SLICE_1_H2)")
                f.write(", CEIL(size_h1, SIZE_SLICE_1_H1)")
                f.write(", CEIL(size_p6, SIZE_SLICE_1_P6)")
                f.write(", CEIL(size_p5, SIZE_SLICE_1_P5)")
                f.write(", CEIL(size_p4, SIZE_SLICE_1_P4)")
            
            #
            f.write(", stride_reg_x_" + str(each_inner_group))
            f.write(", stride_reg_y_" + str(each_inner_group))

            #
            f.write(", size_internal")

            f.write(");\n")
            f.write("\n")
                #idx_count = idx_count + 1
            #
            #
            #
            f.write("\t\t\t}\n")
            f.write("\t\t}\n")
            f.write("\t\telse\n")
            f.write("\t\t{\n")
            f.write("\t\t\t// [2] Extenral Index: Partial\n")

            #
            f.write("\t\t\tif (")

            idx_count = 0
            for int_idx in l_internal_index:
                if idx_count == 0:
                    f.write(    "size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                else:
                    f.write(" && size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                idx_count = idx_count + 1

            f.write(")\n")
            f.write("\t\t\t{\n")
            f.write("\t\t\t\t// [3] Internal Index: Full\n")
            #
            #
            #
            f.write("\t\t\t\t// >>> External: Partial && Internal: Full\n")
            f.write("\t\t\t\tprintf (\"External: Partial, Internal: Full\\n\");\n")
            #
            #
            #
            #idx_count = 1
            #for each_inner_group in range(1, len(l_combined_register_mappings) + 1):
            f.write("\t\t\t\t// kernel_" + str(each_inner_group) + "\n")
            f.write("\t\t\t\t" + kernel_name + "_2_" + str(each_inner_group) + "<<<")
            f.write("gridsize_" + str(each_inner_group) + ", blocksize_" + str(each_inner_group) + ">>>")
            f.write("(")

            #   related to "output"
            par_count = 0
            for each_par in l_combined_t3_parameters[each_inner_group - 1]:
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1

            #   related to "left"
            for each_par in l_combined_t2_parameters[each_inner_group - 1]:
                f.write(", " + each_par[0])

            #   related to "right"
            for each_par in l_combined_v2_parameters[each_inner_group - 1]:
                f.write(", " + each_par[0])

            #
            #   "opt_pre_computed"
            #
            if opt_pre_computed == -1:
                f.write(", size_h3")
                f.write(", size_h2")
                f.write(", size_h1")
                f.write(", size_p6")
                f.write(", size_p5")
                f.write(", size_p4")
                f.write(", size_h7")

                f.write(", CEIL(size_h3, SIZE_SLICE_1_H3)")
                f.write(", CEIL(size_h2, SIZE_SLICE_1_H2)")
                f.write(", CEIL(size_h1, SIZE_SLICE_1_H1)")
                f.write(", CEIL(size_p6, SIZE_SLICE_1_P6)")
                f.write(", CEIL(size_p5, SIZE_SLICE_1_P5)")
                f.write(", CEIL(size_p4, SIZE_SLICE_1_P4)")
            
            #
            f.write(", stride_reg_x_" + str(each_inner_group))
            f.write(", stride_reg_y_" + str(each_inner_group))

            #
            f.write(", size_internal")

            f.write(");\n")
            f.write("\n")
                #idx_count = idx_count + 1
            #
            #
            #
            f.write("\t\t\t}\n")
            f.write("\t\t\telse\n")
            f.write("\t\t\t{\n")
            f.write("\t\t\t\t// [4] Internal Index: Partial\n")
            #
            #
            #
            f.write("\t\t\t\t// >>> External: Partial && Internal: Partial\n")
            f.write("\t\t\t\tprintf (\"External: Partial, Internal: Partial\\n\");\n")
            #
            #
            #
            #idx_count = 1
            #for each_inner_group in range(1, len(l_combined_register_mappings) + 1):
            f.write("\t\t\t\t// kernel_" + str(each_inner_group) + "\n")
            f.write("\t\t\t\t" + kernel_name + "_3_" + str(each_inner_group) + "<<<")
            f.write("gridsize_" + str(each_inner_group) + ", blocksize_" + str(each_inner_group) + ">>>")
            f.write("(")

            #   related to "output"
            par_count = 0
            for each_par in l_combined_t3_parameters[each_inner_group - 1]:
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1

            #   related to "left"
            for each_par in l_combined_t2_parameters[each_inner_group - 1]:
                f.write(", " + each_par[0])

            #   related to "right"
            for each_par in l_combined_v2_parameters[each_inner_group - 1]:
                f.write(", " + each_par[0])

            #
            #   "opt_pre_computed"
            #
            if opt_pre_computed == -1:
                f.write(", size_h3")
                f.write(", size_h2")
                f.write(", size_h1")
                f.write(", size_p6")
                f.write(", size_p5")
                f.write(", size_p4")
                f.write(", size_h7")

                f.write(", CEIL(size_h3, SIZE_SLICE_1_H3)")
                f.write(", CEIL(size_h2, SIZE_SLICE_1_H2)")
                f.write(", CEIL(size_h1, SIZE_SLICE_1_H1)")
                f.write(", CEIL(size_p6, SIZE_SLICE_1_P6)")
                f.write(", CEIL(size_p5, SIZE_SLICE_1_P5)")
                f.write(", CEIL(size_p4, SIZE_SLICE_1_P4)")
            
            #
            f.write(", stride_reg_x_" + str(each_inner_group))
            f.write(", stride_reg_y_" + str(each_inner_group))

            #
            f.write(", size_internal")

            f.write(");\n")
            f.write("\n")
                #idx_count = idx_count + 1
            #
            #
            #
            f.write("\t\t\t}\n")
            f.write("\t\t}\n")

        f.write("\t}\n")
        f.write("\telse\n")
        f.write("\t{\n")
        f.write("\t\t// [1] Register Transpose: ON\n")

        #
        f.write("\t\tif (")

        idx_count = 0
        for ext_idx in l_external_index:
            if idx_count == 0:
                f.write(    "size_" + ext_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + ext_idx.capitalize() + " == 0") 
            else:
                f.write(" && size_" + ext_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + ext_idx.capitalize() + " == 0")
            idx_count = idx_count + 1

        f.write(")\n")
        f.write("\t\t{\n")
        f.write("\t\t\t// [2] Extenral Index: Full\n")

        # 
        f.write("\t\t\tif (")

        idx_count = 0
        for int_idx in l_internal_index:
            if idx_count == 0:
                f.write(    "size_" + int_idx + " % SIZE_SLICE_" + str(1) + "_" + int_idx.capitalize() + " == 0")
            else:
                f.write(" && size_" + int_idx + " % SIZE_SLICE_" + str(1) + "_" + int_idx.capitalize() + " == 0")
            idx_count = idx_count + 1

        f.write(")\n")
        f.write("\t\t\t{\n")

        f.write("\t\t\t\t// [3] Internal Index: Full\n")
        #
        #
        #
        f.write("\t\t\t\t// >>> External: Full && Internal: Full\n")        # <------------    -1 & -1
        f.write("\t\t\t\t" + kernel_name + "_1_rt" + "<<<")
        f.write("gridsize_" + str(1) + ", blocksize_" + str(1) + ">>>")
        f.write("(")

        #
        inner_count = 0
        for each_inner_group in l_combined_t3_parameters:
            print ("t3: ", each_par)
            par_count = 0
            for each_par in each_inner_group:
                if inner_count != 0 and par_count == 0:
                    par_count = par_count + 1
                    continue
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1
            inner_count = inner_count + 1

        #
        for each_inner_group in l_combined_t2_parameters:
            print ("t2: ", each_par)
            for each_par in each_inner_group:
                f.write(", " + each_par[0])
    
        #
        for each_inner_group in l_combined_v2_parameters:
            print ("v2: ", each_par)
            for each_par in each_inner_group:
                f.write(", " + each_par[0])
        
        #
        #   "opt_pre_computed"
        #
        if opt_pre_computed == -1:
            f.write(", size_h3")
            f.write(", size_h2")
            f.write(", size_h1")
            f.write(", size_p6")
            f.write(", size_p5")
            f.write(", size_p4")
            f.write(", size_h7")

            f.write(", CEIL(size_h3, SIZE_SLICE_1_H3)")
            f.write(", CEIL(size_h2, SIZE_SLICE_1_H2)")
            f.write(", CEIL(size_h1, SIZE_SLICE_1_H1)")
            f.write(", CEIL(size_p6, SIZE_SLICE_1_P6)")
            f.write(", CEIL(size_p5, SIZE_SLICE_1_P5)")
            f.write(", CEIL(size_p4, SIZE_SLICE_1_P4)")


        #
        f.write(", stride_reg_x_" + str(1))
        f.write(", stride_reg_y_" + str(1))

        # 
        f.write(", size_internal")

        f.write(");\n")
        f.write("\n")
        #
        #
        #
        f.write("\t\t\t}\n")
        f.write("\t\t\telse\n")
        f.write("\t\t\t{\n")
        f.write("\t\t\t\t// [3] Internal Index: Partial\n")
        #
        #
        #
        f.write("\t\t\t\t// >>> External: Full && Internal: Partial\n")     # <-------------
        f.write("\t\t\t\t" + kernel_name + "_2_rt" + "<<<")
        f.write("gridsize_" + str(1) + ", blocksize_" + str(1) + ">>>")
        f.write("(")

        #
        inner_count = 0
        for each_inner_group in l_combined_t3_parameters:
            par_count = 0
            for each_par in each_inner_group:
                if inner_count != 0 and par_count == 0:
                    par_count = par_count + 1
                    continue
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1
            inner_count = inner_count + 1

        #
        for each_inner_group in l_combined_t2_parameters:
            for each_par in each_inner_group:
                f.write(", " + each_par[0])
    
        #
        for each_inner_group in l_combined_v2_parameters:
            for each_par in each_inner_group:
                f.write(", " + each_par[0])

        #
        #   "opt_pre_computed"
        #
        if opt_pre_computed == -1:
            f.write(", size_h3")
            f.write(", size_h2")
            f.write(", size_h1")
            f.write(", size_p6")
            f.write(", size_p5")
            f.write(", size_p4")
            f.write(", size_h7")

            f.write(", CEIL(size_h3, SIZE_SLICE_1_H3)")
            f.write(", CEIL(size_h2, SIZE_SLICE_1_H2)")
            f.write(", CEIL(size_h1, SIZE_SLICE_1_H1)")
            f.write(", CEIL(size_p6, SIZE_SLICE_1_P6)")
            f.write(", CEIL(size_p5, SIZE_SLICE_1_P5)")
            f.write(", CEIL(size_p4, SIZE_SLICE_1_P4)")

        #
        f.write(", stride_reg_x_" + str(1))
        f.write(", stride_reg_y_" + str(1))

        # 
        f.write(", size_internal")

        f.write(");\n")
        f.write("\n")
        #
        #
        #
        f.write("\t\t\t}\n")
        f.write("\t\t}\n")
        f.write("\t\telse\n")
        f.write("\t\t{\n")
        f.write("\t\t\t// [2] Extenral Index: Partial\n")
        # 
        f.write("\t\t\tif (")

        idx_count = 0
        for int_idx in l_internal_index:
            if idx_count == 0:
                f.write(    "size_" + int_idx + " % SIZE_SLICE_" + str(1) + "_" + int_idx.capitalize() + " == 0")
            else:
                f.write(" && size_" + int_idx + " % SIZE_SLICE_" + str(1) + "_" + int_idx.capitalize() + " == 0")
            idx_count = idx_count + 1

        f.write(")\n")
        f.write("\t\t\t{\n")
        f.write("\t\t\t\t// [3] Internal Index: Full\n")
        #
        #
        #
        f.write("\t\t\t\t// >>> External: Partial && Internal: Full\n")     # <-------------
        f.write("\t\t\t\t" + kernel_name + "_3_rt" + "<<<")
        f.write("gridsize_" + str(1) + ", blocksize_" + str(1) + ">>>")
        f.write("(")

        #
        inner_count = 0
        for each_inner_group in l_combined_t3_parameters:
            par_count = 0
            for each_par in each_inner_group:
                if inner_count != 0 and par_count == 0:
                    par_count = par_count + 1
                    continue
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1
            inner_count = inner_count + 1

        #
        for each_inner_group in l_combined_t2_parameters:
            for each_par in each_inner_group:
                f.write(", " + each_par[0])
    
        #
        for each_inner_group in l_combined_v2_parameters:
            for each_par in each_inner_group:
                f.write(", " + each_par[0])

        #
        #   "opt_pre_computed"
        #
        if opt_pre_computed == -1:
            f.write(", size_h3")
            f.write(", size_h2")
            f.write(", size_h1")
            f.write(", size_p6")
            f.write(", size_p5")
            f.write(", size_p4")
            f.write(", size_h7")

            f.write(", CEIL(size_h3, SIZE_SLICE_1_H3)")
            f.write(", CEIL(size_h2, SIZE_SLICE_1_H2)")
            f.write(", CEIL(size_h1, SIZE_SLICE_1_H1)")
            f.write(", CEIL(size_p6, SIZE_SLICE_1_P6)")
            f.write(", CEIL(size_p5, SIZE_SLICE_1_P5)")
            f.write(", CEIL(size_p4, SIZE_SLICE_1_P4)")

        #
        f.write(", stride_reg_x_" + str(1))
        f.write(", stride_reg_y_" + str(1))

        # 
        f.write(", size_internal")

        f.write(");\n")
        f.write("\n")
        #
        #
        #
        f.write("\t\t\t}\n")
        f.write("\t\t\telse\n")
        f.write("\t\t\t{\n")
        f.write("\t\t\t\t// [3] Internal Index: Partial\n")
        #
        #
        #
        f.write("\t\t\t\t// >>> External: Partial && Internal: Partial\n")  # <-------------
        f.write("\t\t\t\t" + kernel_name + "_4_rt" + "<<<")
        f.write("gridsize_" + str(1) + ", blocksize_" + str(1) + ">>>")
        f.write("(")

        #
        inner_count = 0
        for each_inner_group in l_combined_t3_parameters:
            par_count = 0
            for each_par in each_inner_group:
                if inner_count != 0 and par_count == 0:
                    par_count = par_count + 1
                    continue
                if par_count == 0:
                    f.write(each_par[0])
                else:
                    f.write(", " + each_par[0])
                par_count = par_count + 1
            inner_count = inner_count + 1

        #
        for each_inner_group in l_combined_t2_parameters:
            for each_par in each_inner_group:
                f.write(", " + each_par[0])
    
        #
        for each_inner_group in l_combined_v2_parameters:
            for each_par in each_inner_group:
                f.write(", " + each_par[0])

        #
        #   "opt_pre_computed"
        #
        if opt_pre_computed == -1:
            f.write(", size_h3")
            f.write(", size_h2")
            f.write(", size_h1")
            f.write(", size_p6")
            f.write(", size_p5")
            f.write(", size_p4")
            f.write(", size_h7")

            f.write(", CEIL(size_h3, SIZE_SLICE_1_H3)")
            f.write(", CEIL(size_h2, SIZE_SLICE_1_H2)")
            f.write(", CEIL(size_h1, SIZE_SLICE_1_H1)")
            f.write(", CEIL(size_p6, SIZE_SLICE_1_P6)")
            f.write(", CEIL(size_p5, SIZE_SLICE_1_P5)")
            f.write(", CEIL(size_p4, SIZE_SLICE_1_P4)")

        #
        f.write(", stride_reg_x_" + str(1))
        f.write(", stride_reg_y_" + str(1))

        # 
        f.write(", size_internal")

        f.write(");\n")
        f.write("\n")
        #
        #
        #
        f.write("\t\t\t}\n")
        f.write("\t\t}\n")
        f.write("\t}\n")
    else:
        #
        #
        #
        f.write("\t// No Chance to Utilize the Register Transpose\n")
        for each_inner_group in range(1, len(l_combined_var_thread_block) + 1):
            #
            f.write("\tif (")

            idx_count = 0
            for ext_idx in l_external_index:
                if idx_count == 0:
                    f.write(    "size_" + ext_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + ext_idx.capitalize() + " == 0") 
                else:
                    f.write(" && size_" + ext_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + ext_idx.capitalize() + " == 0")
                idx_count = idx_count + 1

            f.write(")\n")
            f.write("\t{\n")
            f.write("\t\t// [2] Extenral Index: Full\n")
            
            #
            f.write("\t\tif (")

            idx_count = 0
            for int_idx in l_internal_index:
                if idx_count == 0:
                    f.write(    "size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                else:
                    f.write(" && size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                idx_count = idx_count + 1

            f.write(")\n")
            f.write("\t\t{\n")
            f.write("\t\t\t// [3] Internal Index: Full\n")
            #
            #
            #
            f.write("\t\t\t// >>> External: Full && Internal: Full\n")
            f.write("\t\t\tprintf (\"External: Full, Internal: Full\\n\");\n")
            #
            #
            #
            if len(l_internal_index) > 1:
                f.write("\t\t\tif (size_internal > MAX_CONST_LEN)\n")
                f.write("\t\t\t{\n")
                
                #
                steps_tab = 4
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_1_tex_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                1, opt_pre_computed)

                f.write("\t\t\t}\n")
                f.write("\t\t\telse\n")
                f.write("\t\t\t{\n")

                #
                steps_tab = 4
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_1_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)

                f.write("\t\t\t}\n")
            else:
                #
                steps_tab = 3
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_1_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)
            #
            #
            #
            f.write("\t\t}\n")
            f.write("\t\telse\n")
            f.write("\t\t{\n")
            f.write("\t\t\t// [4] Internal Index: Partial\n")
            #
            #
            #
            f.write("\t\t\t// >>> External: Full && Internal: Partial\n")
            f.write("\t\t\tprintf (\"External: Full, Internal: Partial\\n\");\n")
            #
            #
            #
            if len(l_internal_index) > 1:
                f.write("\t\t\tif (size_internal > MAX_CONST_LEN)\n")
                f.write("\t\t\t{\n")
                
                steps_tab = 4
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_2_tex_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                1, opt_pre_computed)

                f.write("\t\t\t}\n")
                f.write("\t\t\telse\n")
                f.write("\t\t\t{\n")

                helper_interface.call_kernel(f, steps_tab, kernel_name + "_2_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)

                f.write("\t\t\t}\n")                
            else:
                steps_tab = 3
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_2_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)
            #
            #
            #
            f.write("\t\t}\n")
            f.write("\t}\n")
            f.write("\telse\n")
            f.write("\t{\n")
            f.write("\t\t// [2] Extenral Index: Partial\n")

            #
            f.write("\t\tif (")

            idx_count = 0
            for int_idx in l_internal_index:
                if idx_count == 0:
                    f.write(    "size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                else:
                    f.write(" && size_" + int_idx + " % SIZE_SLICE_" + str(each_inner_group) + "_" + int_idx.capitalize() + " == 0")
                idx_count = idx_count + 1

            f.write(")\n")
            f.write("\t\t{\n")
            f.write("\t\t\t// [3] Internal Index: Full\n")
            #
            #
            #
            f.write("\t\t\t// >>> External: Partial && Internal: Full\n")
            f.write("\t\t\tprintf (\"External: Partial, Internal: Full\\n\");\n")
            #
            #
            #
            if len(l_internal_index) > 1:
                f.write("\t\t\tif (size_internal > MAX_CONST_LEN)\n")
                f.write("\t\t\t{\n")

                steps_tab = 4
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_3_tex_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                1, opt_pre_computed)
    
                f.write("\t\t\t}\n")
                f.write("\t\t\telse\n")
                f.write("\t\t\t{\n")

                helper_interface.call_kernel(f, steps_tab, kernel_name + "_3_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)

                f.write("\t\t\t}\n")  
            else:
                steps_tab = 3
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_3_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)
            #
            #
            #
            f.write("\t\t}\n")
            f.write("\t\telse\n")
            f.write("\t\t{\n")
            f.write("\t\t\t// [4] Internal Index: Partial\n")
            #
            #
            #
            f.write("\t\t\t// >>> External: Partial && Internal: Partial\n")
            f.write("\t\t\tprintf (\"External: Partial, Internal: Partial\\n\");\n")
            #
            #
            #
            if len(l_internal_index) > 1:
                f.write("\t\t\tif (size_internal > MAX_CONST_LEN)\n")
                f.write("\t\t\t{\n")

                steps_tab = 4
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_4_tex_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                1, opt_pre_computed)

                f.write("\t\t\t}\n")
                f.write("\t\t\telse\n")

                f.write("\t\t\t{\n")
                
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_4_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)

                f.write("\t\t\t}\n")
            else:
                steps_tab = 3
                helper_interface.call_kernel(f, steps_tab, kernel_name + "_4_" + str(each_inner_group), each_inner_group, 
                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters, l_combined_internal_addrs,
                l_external_index, l_internal_index,
                0, opt_pre_computed)

            #
            #
            #
            f.write("\t\t}\n")
            f.write("\t}\n")
    #
    f.write("\n")

#
#   [7] Copy the Result from Device to Host
#
def tc_gen_code_interface_MemcpyOutput(f, l_external_index, opt_data_type):
    f.write("\t// Copy the Result from Device to Host\n")

    #
    if opt_data_type == "DOUBLE":
        f.write("\tcudaMemcpy(t3, dev_t3, sizeof(double) * (")
    else:
        f.write("\tcudaMemcpy(t3, dev_t3, sizeof(float) * (")

    idx_count       = 0
    str_size_output = ""
    for each_idx in l_external_index:
        if idx_count == 0:
            str_size_output = "size_" + each_idx
        else:
            str_size_output = str_size_output + " * size_" + each_idx
        idx_count = idx_count + 1

    f.write(str_size_output)
    f.write("), cudaMemcpyDeviceToHost);\n")
    f.write("\n")

#
#   [8] CudaFree
#
def tc_gen_code_interface_CUDAFree(f, l_cuda_malloc, opt_pre_computed):  
    f.write("\t// cudaFree()\n")
    for each_dev_var in l_cuda_malloc:
        if opt_pre_computed == -1:
            if "range"  in each_dev_var[0]:
                continue
            if "base"   in each_dev_var[0]:
                continue
            if "offset" in each_dev_var[0]:
                continue
            if "addr"   in each_dev_var[0]:
                continue
            #
            tc_post_HostDevice_Free.tc_gen_code_helper_cudaFree_noline(f, each_dev_var[0])
        else:
            tc_post_HostDevice_Free.tc_gen_code_helper_cudaFree_noline(f, each_dev_var[0])
    f.write("\n")
    f.write("\n")

#
#   [9] HostFree
#
def tc_gen_code_interface_HostFree(f, l_host_dynamic):
    f.write("\t// HostFree\n")
    for each_host_var in l_host_dynamic:
        tc_post_HostDevice_Free.tc_gen_code_helper_hostFree_noline(f, each_host_var)
    f.write("\n")

#
#
#
def tc_gen_code_main_new(f):
    #
    #
    #
    f.write("\n")
    f.write("// (Temporally) For Test\n")
    f.write("int main(int argc, char** argv)\n")
    f.write("{\n")

    #
    f.write("\t// [1] Inputs Initialized by Random Values and an Empty Output\n")
    f.write("\t// [2] To Call the Interface\n")
    f.write("\t// [3] Correctness Check\n")

    #
    f.write("}\n")

#
#   An Interface to Call the Above Caller. 
#
def tc_gen_code_interface(f, interface_name, l_interface_info, l_tile_sizes, l_split_representative_problem_size, opt_data_type):
    #
    #
    f.write("\n")
    f.write("// This is written by tc_interface.tc_gen_code_interface()\n")
    f.write("// This Interface Should be Called to Run the Kernels\n")
    f.write("extern \"C\"\n")
    f.write("void " + interface_name + "_(")

    #
    #   l_interface_info: [0] All Index, [1] Output, [2] Inputs, [3] Conditions, [4] Options
    #   If the code generator supports multiple-output-groups, this should be corrected.
    #
    #   [0] All Index
    #
    l_split_info = l_interface_info[0][5]
    if len(l_interface_info[0][5]) > 0:
        #
        idx_count = 0
        for each_index in l_interface_info[0][0]:
            #
            #   Checking if this index is used for Split or not
            #
            is_skip = -1
            for each_split_info in l_split_info:
                #
                if each_index == each_split_info[1]:
                    each_index = each_split_info[0]
                elif each_index == each_split_info[2]:
                    is_skip = 1
            #
            #
            #
            if is_skip != 1:
                if idx_count == 0:
                    f.write("int size_" + each_index)
                else:
                    f.write(", int size_" + each_index)
                idx_count = idx_count + 1
            else:
                is_skip == -1   # reset
    else:
        idx_count = 0
        for each_index in l_interface_info[0][0]:
            if idx_count == 0:
                f.write("int size_" + each_index)
            else:
                f.write(", int size_" + each_index)
            idx_count = idx_count + 1

    #   [1] Output
    if opt_data_type == "DOUBLE":
        f.write(", double* " + l_interface_info[0][1])
    else:
        f.write(", float* " + l_interface_info[0][1])

    #   [2] Inputs
    for each_pair_inputs in l_interface_info[0][2]:
        for each_input in each_pair_inputs:
            #
            if opt_data_type == "DOUBLE":
                f.write(", double* " + each_input)
            else:
                f.write(", float* " + each_input)
    
    #   [3] Conditions
    for each_condition in l_interface_info[0][3]:
        f.write(", int " + each_condition)

    #   [4] Option(s): (Currently) Only One Option for Register Transpose
    f.write(", int " + l_interface_info[0][4])

    f.write(")\n")
    f.write("{\n")

    #   [>] Pre-Processing
    f.write("\t// Pre-Processing for Split\n")
    f.write("\t// Based on Tile-Sizes and Problem-Size\n")
    f.write("\t// Currently, one index can be split into two indices\n")

    # l_split_info
    for each_split in l_split_info:
        print (">> each_split: ", each_split)
        print (">> l_tile_sizes: ", l_tile_sizes)
        print (">> l_split_representative_problem_size: ", l_split_representative_problem_size)
        #
        #   Each Split-Infomation
        #
        tc_code_etc.tc_gen_code_write_line(f, 1, "int size_" + each_split[1] + ";")
        tc_code_etc.tc_gen_code_write_line(f, 1, "int size_" + each_split[2] + ";")

        #str_first_tile_size = str(tc_helper.tc_gen_helper_find(l_tile_sizes, each_split[1]))
        #print ("~~~~~~: ", tc_helper.tc_gen_helper_find(l_split_representative_problem_size, each_split[1]))
        str_first_tile_size = str(tc_helper.tc_gen_helper_find(l_split_representative_problem_size, each_split[1]))

        #
        #   Pre-Processing
        #
        tc_code_etc.tc_gen_code_write_line(f, 0, "")
        tc_code_etc.tc_gen_code_write_line(f, 1, "if (size_" + each_split[0] + " % " + str_first_tile_size + " == 0)")
        tc_code_etc.tc_gen_code_write_line(f, 1, "{")

        #
        tc_code_etc.tc_gen_code_write_line(f, 2, "//")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[1] + " = " + str_first_tile_size + ";")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[2] + " = size_" + each_split[0] + " / " + str_first_tile_size + ";")

        #
        tc_code_etc.tc_gen_code_write_line(f, 1, "}")
        tc_code_etc.tc_gen_code_write_line(f, 1, "else")
        tc_code_etc.tc_gen_code_write_line(f, 1, "{")

        #
        tc_code_etc.tc_gen_code_write_line(f, 2, "//")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[1] + " = size_" + each_split[0] + ";")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[2] + " = 1;")
        
        #
        tc_code_etc.tc_gen_code_write_line(f, 1, "}")
        
    f.write("\n")

    f.write("\t// Call An Application\n")
    f.write("\t" + interface_name + "(")
    
    #
    #   [0] All Index (Split-Version)
    #
    idx_count = 0
    for each_index in l_interface_info[0][0]:
        if idx_count == 0:
            f.write("size_" + each_index)
        else:
            f.write(", size_" + each_index)
        idx_count = idx_count + 1

    
    #   [1] Output
    f.write(", " + l_interface_info[0][1])

    #   [2] Inputs
    for each_pair_inputs in l_interface_info[0][2]:
        for each_input in each_pair_inputs:
            f.write(", " + each_input)
    
    #   [3] Conditions
    for each_condition in l_interface_info[0][3]:
        f.write(", " + each_condition)

    #   [4] Option(s):
    f.write(", " + l_interface_info[0][4])
    f.write(");\n")


    f.write("}\n")

#
#
#
def tc_interface_SMEM_Size(l_tensor_contractions, l_external_index, l_internal_index, l_tile_sizes, l_register_tiles):
    #
    #
    #
    for each_tc in l_tensor_contractions:
        #
        #   Outputs
        #
        size_smem_left  = 1
        size_smem_right = 1
        str_left        = 1
        str_right       = 1

        #
        #   each_tc[0][1]: Left Tensor
        #
        for each_idx in each_tc[0][1]:
            #
            #   2D SMEM (The Product of External Indices)
            #
            if tc_helper.tc_gen_helper_find_1d(l_internal_index, each_idx) == -1:
                size_smem_left = size_smem_left * tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)
            
            #
            #   Only external indices (except for one mapped on register tiles)
            #
            if tc_helper.tc_gen_helper_find_1d(l_external_index, each_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_register_tiles, each_idx) == -1:
                    str_left = str_left * tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)

        #
        #   each_tc[1][1]: Right Tensor
        #
        for each_idx in each_tc[1][1]:
            #
            #   2D SMEM (The Product of External Indices)
            #
            if tc_helper.tc_gen_helper_find_1d(l_internal_index, each_idx) == -1:
                size_smem_right = size_smem_right * tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)

            #
            #   Only external indices (except for one mapped on register tiles)
            #
            if tc_helper.tc_gen_helper_find_1d(l_external_index, each_idx) != -1:
                if tc_helper.tc_gen_helper_find_1d(l_register_tiles, each_idx) == -1:
                    str_right = str_right * tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)

    #
    return size_smem_left, size_smem_right, str_left, str_right

#
#
#
def tc_interface_TB_Size(l_mapping_tb_x, l_mapping_tb_y, l_tile_sizes):
    #
    size_TB_X = 1
    size_TB_Y = 1

    #
    for each_idx in l_mapping_tb_x:
        size_TB_X = size_TB_X * tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)
    
    #
    for each_idx in l_mapping_tb_y:
        size_TB_Y = size_TB_Y * tc_helper.tc_gen_helper_find(l_tile_sizes, each_idx)
    
    #
    return size_TB_X, size_TB_Y
