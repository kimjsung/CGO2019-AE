import src.generators.tc_helper     as tc_helper

#
#   Kernels for Register Transpose
#
def tc_gen_code_Kernel_Register_Transpose(f,    kernel_name, l_inner_groups, l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                                opt_gen_p7, opt_gen_full):
    #
    #   [1] Header (To-Do: Strides for Non-FVI of Internal Index)
    #
    tc_gen_code_Kernel_Head_RT(f, kernel_name + "_rt", l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var)

    #   Open
    f.write("{\n")

    f.write("\t// Kernel for Register Transpose\n")
    f.write("\t// " + str(len(l_inner_groups)) + " of Inner Groups will be merged by using Register Transpose\n")
    f.write("\n")

    #
    #   Each Inner-Group   
    #   
    inner_count         = 1
    temp_tb_2D          = list()
    for each_inner_group in l_inner_groups:
        #
        #   For Each Inner-Group,
        #
        size_smem_left, size_smem_right, str_left, str_right    = tc_interface.tc_interface_SMEM_Size(each_inner_group[6], each_inner_group[4], each_inner_group[5], each_inner_group[8], each_inner_group[2])
        size_TB_X, size_TB_Y                                    = tc_interface.tc_interface_TB_Size(each_inner_group[1][0], each_inner_group[1][1], each_inner_group[8])
        size_smem_internal                                      = tc_helper.tc_gen_helper_CheckingIntUnit(each_inner_group[4], each_inner_group[8], each_inner_group[5])
        size_REG_X                                              = tc_helper.tc_gen_helper_find(each_inner_group[8], each_inner_group[2][1])
        size_REG_Y                                              = tc_helper.tc_gen_helper_find(each_inner_group[8], each_inner_group[2][0])
        opt_load_t2, opt_load_v2                                = tc_helper.tc_gen_helper_CheckingInternalFVI(each_inner_group[6], each_inner_group[5])
        l_blk_boundary_rng  = list()
    
        #print (inner_count, ">>> TB:", size_TB_X, size_TB_Y, ", REG: ", size_REG_X, size_REG_Y)
        #
        #
        #
        if opt_gen_full != -1:
            tc_helper.tc_gen_helper_CheckingBoundary(l_blk_boundary_rng, each_inner_group[3], each_inner_group[8], each_inner_group[2], each_inner_group[1], each_inner_group[6][0][0][1], each_inner_group[6][0][1][1])

        if inner_count == 1:
            #
            #   [2] Initialization 
            #       Q) This first information can be used for the other inner-groups?
            #   
            f.write("\t// Initialization\n")
            tc_gen_code_Kernel_Initial(f,
            size_smem_internal, size_smem_left, size_smem_right,
            each_inner_group[1], each_inner_group[4],
            size_REG_X, size_REG_Y,
            opt_gen_p7, opt_gen_full, inner_count)    # "1" kernel number...

            temp_tb_2D = each_inner_group[1]

        #
        #   [3] An Inner-Group
        #
        f.write("\t// Within Inner-Group\n")
        #
        #   For Each Tensor Contraction,
        #
        idx_countractions = 1
        for tensor_contraction in each_inner_group[7]:
            f.write("\t// Tensor Contraction\n")

            #
            if opt_gen_p7 == 1 and (idx_countractions > 1 or inner_count > 1):
                f.write("\tinternal_upperbound = 0;\n")

            #   [START] Tensor Contraction
            f.write("\t#pragma unroll 1\n")
            f.write("\tfor (int l = 0; l < size_internal; l += SIZE_INT_UNIT_" + str(inner_count) + ")\n")
            f.write("\t{\n")

            #
            #   For Generalizing Internal Index,
            #
            if opt_gen_p7 == 1:
                f.write("\t\t// For Generalizing Contraction Index\n")
                f.write("\t\tinternal_offset = (l + SIZE_INT_UNIT_" + str(inner_count) + ") - size_internal;\n")
                f.write("\t\tif (internal_offset > 0) internal_upperbound = internal_offset;\n")
                f.write("\n")

            #
            #   [Main] Loads Inputs >> To-Do: Need to Double-Check!
            #
            f.write("\t\t// Load Inputs\n")
            tc_gen_code_Kernel_Load_Inputs(f, size_TB_X, size_TB_Y, size_smem_left, size_smem_right, size_smem_internal,  #
                l_blk_boundary_rng,
                tensor_contraction,                         #
                each_inner_group[8], each_inner_group[5],
                temp_tb_2D, each_inner_group[2],            # temp_tb_2D
                #each_inner_group[1], each_inner_group[2],
                opt_gen_full, opt_gen_p7, 
                opt_load_t2, opt_load_v2, inner_count)

            #
            #   [Start] Computes
            #
            f.write("\t\t// Computes: Cross-Product\n")
            f.write("\t\tfor (int ll = 0; ll < SIZE_INT_UNIT_" + str(inner_count))
            
            #   For "Internal Index",
            if opt_gen_p7 == 1:
                f.write(" - internal_upperbound")

            f.write("; ll++)\n")
            f.write("\t\t{\n")

            #
            #   [Main] Computes
            #
            f.write("\t\t\t// Computes\n")
            tc_gen_code_Kernel_Compute(f, size_REG_X, size_REG_Y, str_left, str_right, tensor_contraction)

            #   [End] Computes
            f.write("\t\t}\n")
            f.write("\t\t__syncthreads();\n")

            #   [END] Tensor Contraction
            f.write("\t}\n")
            f.write("\n")

            #
            idx_countractions = idx_countractions + 1

        #
        #   Part: Register Transpose
        #
        if inner_count < len(l_inner_groups):
            #
            f.write("\t// Register-Transpose: " + str(inner_count - 1) + " with " + str(inner_count) +  "\n")

            #
            #
            #
            tc_gen_code_Kernel_Process_Register_Transpose(f, l_inner_groups[inner_count - 1], l_inner_groups[inner_count], size_smem_internal, size_smem_left, size_smem_right, size_TB_X * size_TB_Y * size_REG_X * size_REG_Y)


            f.write("\n")

        #
        #   [5] Register Tiles -> Global Memory
        #
        if inner_count == len(l_inner_groups):
            f.write("\t// Store the Results to Global Memory\n")
            f.write("\t// This should be based on the last inner-group\n")
            tc_gen_code_Kernel_Store_Results(f, opt_gen_full, 
                                                #each_inner_group[1], each_inner_group[2], 
                                                temp_tb_2D, each_inner_group[2], 
                                                size_REG_X, size_REG_Y, 1, 1)
                                                #size_REG_X, size_REG_Y, inner_count)

        inner_count = inner_count + 1
        #
        #   END OF FOR: INNER-GROUP
        #

    #   Close
    f.write("}\n")
#
#   [Register Transpose][Process] >>>> Creating Codes for Register Transpose According to the Information Created by "tc_gen_code_Kernel_Algorithm_Register_Transpose()"
#
def tc_gen_code_Kernel_Process_Register_Transpose(f, top_inner_group, bottom_inner_group, size_smem_internal, size_smem_left, size_smem_right, size_output):
    #
    #
    #
    print ("[Code Generator][Code Kernel][Process][Register-Transpose] In here, Register Transpose is Processed")
    print ("[Code Generator][Code Kernel][Process][Register-Transpose] TOP TB:", top_inner_group[1])
    print ("[Code Generator][Code Kernel][Process][Register-Transpose] TOP REG:", top_inner_group[2])
    print ("[Code Generator][Code Kernel][Process][Register-Transpose] TOP TILES:", top_inner_group[8])
    print ("[Code Generator][Code Kernel][Process][Register-Transpose] BOTTOM TB:", bottom_inner_group[1])
    print ("[Code Generator][Code Kernel][Process][Register-Transpose] BOTTOM REG:", bottom_inner_group[2])
    print ("[Code Generator][Code Kernel][Process][Register-Transpose] BOTTOM TILES:", bottom_inner_group[8])

    #
    #   The Number of Loops Depends on Isolated Groups among Threads in a Thread Block.
    #
    f.write("\t// SMEM-LEFT: " + str(size_smem_left * size_smem_internal) + "\n")
    f.write("\t// SMEM-RIGHT: " + str(size_smem_right * size_smem_internal) + "\n")
    f.write("\t// OUTPUT: " + str(size_output) + "\n")
    f.write("\t// Register Transpose Requires at Least " + str(size_output / (size_smem_internal * (size_smem_left + size_smem_right))) + " Steps\n")

    #
    #   [1] Register -> SMEM (sm_a and sm_b) (__synchthreads();)
    #
    f.write("\tif (1)\n")
    f.write("\t{\n")
    f.write("\t\t// STORE the Intermediate Results to SMEM from Registers\n")
    f.write("\t}\n")
    f.write("\t__syncthreads();\n")
    f.write("\n")

    #
    #   [2] SMEM -> Register (sm_a and sm_b) (_synchthreads();)
    #
    f.write("\tif (1)\n")
    f.write("\t{\n")
    f.write("\t\t// LOAD the Intermediate Results to Registers from SMEM\n")
    f.write("\t}\n")
    f.write("\t__syncthreads();\n")
    
