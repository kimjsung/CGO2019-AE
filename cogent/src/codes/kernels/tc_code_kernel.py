import copy

import src.generators.tc_helper                                 as tc_helper
import src.codes.tc_code_etc                                    as tc_code_etc

import src.codes.kernels.tc_interface                           as tc_interface
import src.codes.kernels.tc_code_kernel_fusion                  as tc_code_kernel_fusion
import src.codes.kernels.tc_code_kernel_helper                  as tc_code_kernel_helper
import src.codes.kernels.tc_code_kernel_compute                 as tc_code_kernel_compute
import src.codes.kernels.tc_code_kernel_load_inputs             as tc_code_kernel_load_inputs
import src.codes.kernels.tc_code_kernel_load_inputs_details     as tc_code_kernel_load_inputs_details

import src.codes.kernels.tc_code_kernel_store_output            as tc_code_kernel_store_output
import src.codes.kernels.tc_code_kernel_head                    as tc_code_kernel_head

#
# For the fused kernel,
# Which information is needed to build the fused kernel?
#
def tc_gen_code_Kernel( f,              name,               l_t3_d_decl_var,    l_t2_d_decl_var,    l_v2_d_decl_var,
                        l_input_strides,
                        l_inputs_addr,  l_t3_mapping_tb_2D, l_t3_mapping_reg,   l_t3_idx,           l_internal_idx,     l_t3_slices,
                        size_sm_a,      size_sm_b,          size_sm_p7,
                        size_reg_y,     size_reg_x,         size_tb_y,          size_tb_x,          int_str_t2,         int_str_v2,
                        l_blk_boundary_rng,
                        opt_gen_p7,     opt_gen_full,       opt_load_t2,        opt_load_v2,        opt_pre_computed,    opt_internal, opt_data_type,
                        opt_shared_padding,
                        idx_kernel):
    #
    #   Step 0: Writing Kernel Head
    #
    tc_code_kernel_head.tc_gen_code_Kernel_Head(f, name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, l_input_strides, l_t3_idx, l_internal_idx, l_inputs_addr,
                                                opt_internal, opt_pre_computed, opt_data_type)

    #   Open for the kernel
    f.write("{\n")

    #
    #   Step 1: Initialization
    #
    tc_gen_code_Kernel_Initial(f,   size_sm_p7,         size_sm_a,      size_sm_b,
                                    l_t3_mapping_tb_2D, l_t3_idx,       l_t3_slices,
                                    size_reg_x,         size_reg_y,
                                    opt_gen_p7,         opt_gen_full,   opt_pre_computed,
                                    opt_shared_padding, opt_data_type,
                                    idx_kernel)

    #
    #   For Each Tensor Contraction,
    #   Depends on # of tensor contractions which will be fused in this kernel.
    #
    idx_contractions = 1
    for tensor_contraction in l_inputs_addr:
        #
        f.write("\t// tensor contraction: " + str(tensor_contraction) + "\n")

        #
        if opt_gen_p7 == 1 and idx_contractions > 1:
            f.write("\tinternal_upperbound = 0;\n")

        f.write("\t#pragma unroll 1\n")         #   THIS MIGHT BE FOR THE CONSISTENT PERFORMANCE.
        f.write("\tfor (int l = 0; l < size_internal; l += SIZE_INT_UNIT_" + str(idx_kernel) + ")\n")
        f.write("\t{\n")

        #   For Generalizing p7b,
        if opt_gen_p7 == 1:
            f.write("\t\t// Part: Generalized Contraction Index (p7b)\n")
            f.write("\t\tinternal_offset = (l + SIZE_INT_UNIT_" + str(idx_kernel) + ") - size_internal;\n")
            f.write("\t\tif (internal_offset > 0) internal_upperbound = internal_offset;\n")
            f.write("\n")

        #print ("l_input_strides: ", l_input_strides)
        if len(l_input_strides) == 0:
            tc_code_kernel_load_inputs.tc_gen_code_kernel_load_inputs_base(f, opt_gen_full, opt_gen_p7, 
            opt_load_t2, opt_load_v2, opt_internal,
            tensor_contraction,
            l_t3_slices,
            l_internal_idx,
            l_t3_mapping_tb_2D,
            l_t3_mapping_reg,
            size_sm_p7,
            size_tb_x, size_tb_y,
            idx_kernel)
        else:
            #
            #   Step 2: Loading Inputs from Global Memory to Shared Memory
            #
            tc_code_kernel_load_inputs.tc_gen_code_kernel_load_inputs_base(f, opt_gen_full, opt_gen_p7, 
            opt_load_t2, opt_load_v2, opt_internal,
            tensor_contraction,
            l_t3_slices,
            l_internal_idx,
            l_t3_mapping_tb_2D,
            l_t3_mapping_reg,
            size_sm_p7,
            size_tb_x, size_tb_y,
            idx_kernel)

        #
        #   Computes (Cross-Product: (1x1) * (1x4) four-times)
        #
        f.write("\t\t// Part: Generalized Threads\n")
        f.write("\t\tfor (int ll = 0; ll < SIZE_INT_UNIT_" + str(idx_kernel))

        if opt_gen_p7 == 1:
            f.write(" - internal_upperbound")

        f.write("; ll++)\n")
        f.write("\t\t{\n")

        #
        #   Step 3:
        #       Contracts Input Tensors on Shared Memory
        #       Then, Stores the Results to Registers.
        tc_code_kernel_compute.tc_gen_code_Kernel_Compute(f, size_tb_x, size_tb_y, size_reg_x, size_reg_y, int_str_t2, int_str_v2, tensor_contraction, l_t3_mapping_tb_2D,
                                    opt_pre_computed)

        # For (ll)
        f.write("\t\t}\n")
        f.write("\t\t__syncthreads();\n")

        # For (tensor_contraction)
        f.write("\t}\n")
        f.write("\n")

        #
        idx_contractions = idx_contractions + 1

    #
    #   Step 4: Register Tiles -> Global Memory
    #
    tc_code_kernel_store_output.tc_code_kernel_Store_Results(f, opt_gen_full, l_t3_mapping_tb_2D, l_t3_mapping_reg, size_reg_x, size_reg_y, idx_kernel, -1)

    #   Close
    f.write("}\n")


#
#   To-Do: Should be checked for sd2 functions.
#
def tc_gen_code_Kernel_Load_Checking_Boundary(f, l_blk_boundary_rng, tensor_contraction):
    upper_left      = 1
    upper_right     = 1
    l_left          = list()
    l_right         = list()

    print ("l_blk_boundary_rng: ", l_blk_boundary_rng)

    #
    for left_idx in tensor_contraction[0][4]:
        if tc_helper.tc_gen_helper_find(l_blk_boundary_rng, left_idx) != -1:
            upper_left = upper_left * tc_helper.tc_gen_helper_find(l_blk_boundary_rng, left_idx)
            l_left.append(left_idx);

    #
    for right_idx in tensor_contraction[1][4]:
        if tc_helper.tc_gen_helper_find(l_blk_boundary_rng, right_idx) != -1:
            upper_right = upper_right * tc_helper.tc_gen_helper_find(l_blk_boundary_rng, right_idx)
            l_right.append(right_idx)

    return upper_left, upper_right, l_left, l_right

#
#
#
def tc_gen_code_Kernel_Initial(f,   size_sm_p7,         size_sm_a,      size_sm_b,          # For Shared memory
                                    l_t3_mapping_tb_2D, l_t3_idx,       l_t3_slices,        # For T3 (Output)
                                    size_reg_x,         size_reg_y,                         # For Register-Tiling
                                    opt_gen_p7,         opt_gen_full,   opt_pre_computed,   # Options
                                    opt_shared_padding, opt_data_type,                      # Options
                                    idx_kernel):    # For Options for Generalizing
    #   Shared Memory
    #   Dependson SIZE_P7_UNIT, SIZE_REG_T, SIZE_SLICE (or SIZE_TB)
    f.write("\t// For Shared Memory,\n")
    if opt_shared_padding == 1:
        #
        if opt_data_type == "DOUBLE":
            f.write("\t__shared__ double sm_a[" + str(size_sm_p7) + "][" + str(size_sm_a) + " + 1];\n")
            f.write("\t__shared__ double sm_b[" + str(size_sm_p7) + "][" + str(size_sm_b) + " + 1];\n")
        else:
            f.write("\t__shared__ float sm_a[" + str(size_sm_p7) + "][" + str(size_sm_a) + " + 1];\n")
            f.write("\t__shared__ float sm_b[" + str(size_sm_p7) + "][" + str(size_sm_b) + " + 1];\n")
    else:
        #
        if opt_data_type == "DOUBLE":
            f.write("\t__shared__ double sm_a[" + str(size_sm_p7) + "][" + str(size_sm_a) + "];\n")
            f.write("\t__shared__ double sm_b[" + str(size_sm_p7) + "][" + str(size_sm_b) + "];\n")
        else:
            f.write("\t__shared__ float sm_a[" + str(size_sm_p7) + "][" + str(size_sm_a) + "];\n")
            f.write("\t__shared__ float sm_b[" + str(size_sm_p7) + "][" + str(size_sm_b) + "];\n")
    f.write("\n")

    #
    #   basic variables (used for pre-computed arrays)
    #
    if opt_pre_computed != -1:
        f.write("\tint l_idx_t3         = threadIdx.x + threadIdx.y * SIZE_TB_" + str(idx_kernel) + "_X;\n")                      # (Required)
        f.write("\tint t3_base_thread   = dev_t3_output_base_" + str(idx_kernel) + "[blockIdx.x] + dev_t3_output_offset_" + str(idx_kernel) + "[l_idx_t3];\n")  # Based on inputs (t3_output_base, t3_output_offset) (Required)
    f.write("\n")

    #   Generalized for "Internal Indice"
    if opt_gen_p7 == 1:
        f.write("\tint internal_upperbound   = 0;\n")                            # Based on p7b's size (Generalized)
        f.write("\tint internal_offset;\n")                                      # Based on p7b's size (Generalized)
        f.write("\n")

    #
    #   "-1": pre_computed is off
    #   " 1": pre_computed is on
    #
    if opt_pre_computed == -1:
        #
        numIdx_TB_X = len(l_t3_mapping_tb_2D[0])
        numIdx_TB_Y = len(l_t3_mapping_tb_2D[1])
        
        #
        f.write("\t// when opt_pre_computed == -1, all indices will be calculated manually\n")
        f.write("\t// # of indices mapped on TB_X: " + str(numIdx_TB_X) + "\n")
        f.write("\t// # of indices mapped on TB_Y: " + str(numIdx_TB_Y) + "\n")

        #
        #   TB_X
        #
        if numIdx_TB_X == 1:
            f.write("\tint idx_" + l_t3_mapping_tb_2D[0][0] + " = threadIdx.x;\n")    
        elif numIdx_TB_X == 2:
            f.write("\tint idx_" + l_t3_mapping_tb_2D[0][0] + " = threadIdx.x % SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[0][0].capitalize() + ";\n")
            f.write("\tint idx_" + l_t3_mapping_tb_2D[0][1] + " = threadIdx.x / SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[0][0].capitalize() + ";\n")
        else:
            #
            #   [To-Do]
            #
            list_strides    = list()
            idx_count       = 0
            prev_stride     = ""
            for each_idx in l_t3_mapping_tb_2D[0]:
                if idx_count != len(l_t3_mapping_tb_2D[0]) -1:
                    if prev_stride == "":
                        list_strides.append("SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize())
                        prev_stride = "SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize()
                    else:
                        list_strides.append("SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " * " + prev_stride)
                #
                idx_count += 1

            #
            #
            #
            rev_l_idx_tb_x  = list(reversed(l_t3_mapping_tb_2D[0]))
            idx_count       = 0
            rev_idx_count   = len(rev_l_idx_tb_x) - 1
            for each_rev_idx in rev_l_idx_tb_x:
                #
                rev_idx_count -= 1

                #
                if idx_count == 0:
                    if idx_count == len(rev_l_idx_tb_x) - 1:
                        f.write("\tint idx_" + each_rev_idx + " = threadIdx.x;\n")
                    else:
                        f.write("\tint idx_" + each_rev_idx + " = threadIdx.x / (" + list_strides[rev_idx_count] + ");\n")
                        f.write("\tint remaining_idx = threadIdx.x % (" + list_strides[rev_idx_count] + ");\n")
                else:
                    if idx_count == len(rev_l_idx_tb_x) - 1:
                        f.write("\tint idx_" + each_rev_idx + " = remaining_idx;\n")
                    else:
                        f.write("\tint idx_" + each_rev_idx + " = remaining_idx / (" + list_strides[rev_idx_count] + ");\n")
                        f.write("\tremaining_idx = remaining_idx % (" + list_strides[rev_idx_count] + ");\n")

                #
                idx_count += 1

        #
        #   TB_Y
        #
        if numIdx_TB_Y == 1:
            f.write("\tint idx_" + l_t3_mapping_tb_2D[1][0] + " = threadIdx.y;\n")
        elif numIdx_TB_Y == 2:
            f.write("\tint idx_" + l_t3_mapping_tb_2D[1][0] + " = threadIdx.y % SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[1][0].capitalize() + ";\n")
            f.write("\tint idx_" + l_t3_mapping_tb_2D[1][1] + " = threadIdx.y / SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[1][0].capitalize() + ";\n")
        else:
            #
            #   [To-Do]
            #
            f.write("\t// not-yet: |TB_Y| > 2, " + str(numIdx_TB_Y) + "\n")
            for each_idx in l_t3_mapping_tb_2D[1]:
                f.write("\tidx_" + each_idx + "\n")

        #
        #   Block Numbers
        #
        f.write("\n")
        f.write("\tint tmp_blkIdx;\n")
        rev_l_t3_idx = reversed(l_t3_idx)
        len_l_t3_idx = len(l_t3_idx)

        #
        idx_count = len_l_t3_idx
        for each_idx in rev_l_t3_idx:
            # 
            str_prod_strides = ""
            for each_num_idx in range(0, idx_count - 1):
                if each_num_idx == 0:
                    str_prod_strides = "numBlk_" + l_t3_idx[each_num_idx]
                else:
                    str_prod_strides = "numBlk_" + l_t3_idx[each_num_idx] + " * " + str_prod_strides
                
            #
            if idx_count == len_l_t3_idx:
                f.write("\tint blk_idx_" + each_idx + " = blockIdx.x / (" + str_prod_strides + ");\n")
                f.write("\ttmp_blkIdx = blockIdx.x % (" + str_prod_strides + ");\n")
            else:
                if idx_count == 1:
                    #
                    #
                    #
                    f.write("\tint  blk_idx_" + each_idx + " = tmp_blkIdx;\n")
                elif idx_count == 2:
                    f.write("\tint blk_idx_" + each_idx + " = tmp_blkIdx / " + str_prod_strides + ";\n")
                    f.write("\ttmp_blkIdx = tmp_blkIdx % (" + str_prod_strides + ");\n")
                else:
                    f.write("\tint blk_idx_" + each_idx + " = tmp_blkIdx / (" + str_prod_strides + ");\n")
                    f.write("\ttmp_blkIdx = tmp_blkIdx % (" + str_prod_strides + ");\n")

            #
            f.write("\n")
            idx_count = idx_count - 1

        #
        #   the output's base address for a thread block
        #
        str_t3_base_addr    = ""
        rev_l_t3_idx        = reversed(l_t3_idx)
        l_tb_idx = list()

        for each_axis in l_t3_mapping_tb_2D:
            for each_idx in each_axis:
                l_tb_idx.append(each_idx)

        idx_count       = 0
        existing_idx    = 0
        for each_idx in rev_l_t3_idx:
            #
            idx_t3_count = 0
            for each_tb_idx in l_tb_idx:
                if each_idx == each_tb_idx:
                    existing_idx = 1
                    l_tb_idx.pop(idx_t3_count)
                    break
                idx_t3_count = idx_t3_count + 1

            #
            if existing_idx == 1:
                if idx_count == 0:
                    str_t3_base_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + idx_" + each_idx
                else:
                    str_t3_base_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + idx_" + each_idx + " + (" + str_t3_base_addr + ") * size_" + each_idx 
            else: 
                if idx_count == 0:
                    str_t3_base_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize()
                else:
                    str_t3_base_addr = "blk_idx_" + each_idx + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize() + " + (" + str_t3_base_addr + ") * size_" + each_idx 
            #
            existing_idx    = 0
            idx_count       = idx_count + 1
        
        f.write("\tint t3_base_thread = " + str_t3_base_addr + ";\n")
        f.write("\n")

        #
        #   Ranges
        # 
        if opt_gen_full == 1:
            #
            f.write("\t// need to support partial tiles\n")
            idx_count = 0
            for each_idx in l_t3_idx:
                if idx_count == 0:
                    f.write("\tint rng_" + each_idx)
                else:
                    f.write(", rng_" + each_idx)
                #
                idx_count = idx_count + 1
            f.write(";\n")

            #
            #
            #
            for each_idx in l_t3_idx:
                #
                f.write("\tif ((size_" + each_idx + " - (blk_idx_" + each_idx + " * SIZE_SLICE_1_" + each_idx.capitalize() + ")) >= SIZE_SLICE_1_" + each_idx.capitalize() + ")\n")
                f.write("\t{\n")
                #
                #   IF
                #
                f.write("\t\trng_" + each_idx + " = SIZE_SLICE_1_" + each_idx.capitalize() + ";\n")
                #
                f.write("\t}\n")
                f.write("\telse\n")
                f.write("\t{\n")
                #
                #   ELSE
                #
                f.write("\t\trng_" + each_idx + " = size_" + each_idx + " % SIZE_SLICE_1_" + each_idx.capitalize() + ";\n")
                #
                f.write("\t}\n")


        #
        #
        #
        del l_tb_idx

    else:
        #
        #   Generalized for non-full tile
        #   not yet generalized....
        if opt_gen_full == 1:
            f.write("\t// should support for non-full tiles\n")


            #
            #   To-Do: It does not support multi-dimensional arrays fully.
            #
            # "x"-axis
            if len(l_t3_mapping_tb_2D) != 2:
                print ("ERROR: This part does not support well when len(l_t3_mapping_tb_2D) != 2!")

            #
            numIdxTB_X = len(l_t3_mapping_tb_2D[0])
            numIdxTB_Y = len(l_t3_mapping_tb_2D[1])

            #
            if numIdxTB_X == 2:
                f.write("\tint idx_" + l_t3_mapping_tb_2D[0][0] + " = threadIdx.x % SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[0][0].capitalize() + ";\n")
                f.write("\tint idx_" + l_t3_mapping_tb_2D[0][1] + " = threadIdx.x / SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[0][0].capitalize() + ";\n")
            elif numIdxTB_X == 1:
                f.write("\tint idx_" + l_t3_mapping_tb_2D[0][0] + " = threadIdx.x;\n")
            else:
                #print ("[ERROR]!!! The number of indices mapped on TB_X: ", numIdxTB_X, " (Not Supported Yet)")
                #print ("[ERROR]!!! TB_X: ", l_t3_mapping_tb_2D[0])
                tc_code_etc.tc_gen_code_write_line(f, 1, "// The # of External Indices mapped on TB_X is equal to or greater than 3")

                #
                l_stride_TB_X   = list()
                tmp_str_stride  = ""
                idx_count       = 0
                for each_idx in l_t3_mapping_tb_2D[0]:
                    if idx_count == 0:
                        tmp_str_stride = "SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize()
                    else:
                        tmp_str_stride = tmp_str_stride + " * SIZE_SLICE_" + str(idx_kernel) + "_" + each_idx.capitalize()
                    #
                    l_stride_TB_X.append(tmp_str_stride)
                    idx_count = idx_count + 1

                #
                idx_first               = 0
                idx_second              = 0
                idx_count               = 0
                str_remainning          = "threadIdx.x"
                l_rev_l_t3_mapping_TB_X = list(reversed(l_t3_mapping_tb_2D[0]))
                l_rev_l_stride_TB_X     = list(reversed(l_stride_TB_X))
                for each_idx in l_rev_l_t3_mapping_TB_X:
                    #
                    #   |T_i| == 1, no need to calculate index. 
                    #
                    if tc_helper.tc_gen_helper_find(l_t3_slices, each_idx) == 1:
                        tc_code_etc.tc_gen_code_write_line(f, 1, "int idx_" + each_idx + " \t= 0;")
                    #
                    #   |T_i| != 1
                    #
                    else:
                        #
                        #
                        #
                        if idx_first == 0:
                            #
                            #   THE FVI && The FIRST
                            #
                            if idx_count == len(l_rev_l_t3_mapping_TB_X) - 1:
                                tc_code_etc.tc_gen_code_write_line(f, 1, "int idx_" + each_idx + " \t= " + str_remainning + ";")
                            #
                            #   NOT THE FVI && The FIRST 
                            #
                            else:
                                tc_code_etc.tc_gen_code_write_line(f, 1, "int idx_" + each_idx + " \t= " + str_remainning + " / " + l_rev_l_stride_TB_X[idx_count - 1] + ";")
                                str_remainning = str_remainning + " % " + l_rev_l_stride_TB_X[idx_count - 1]
                            idx_first = 1
                        else:
                            if idx_second == 0:
                                tc_code_etc.tc_gen_code_write_line(f, 1, "int tmp_remainning \t= " + str_remainning + ";")
                                idx_second = 1
                            else:
                                tc_code_etc.tc_gen_code_write_line(f, 1, "tmp_remainning \t= " + str_remainning + ";")

                            #
                            if idx_count == len(l_rev_l_t3_mapping_TB_X) - 1:
                                tc_code_etc.tc_gen_code_write_line(f, 1, "int idx_" + each_idx + " = tmp_remainning ;")
                            else:
                                tc_code_etc.tc_gen_code_write_line(f, 1, "int idx_" + each_idx + " = tmp_remainning / " + l_rev_l_stride_TB_X[idx_count - 1] + ";")
                    
                    #
                    idx_count = idx_count + 1

            #
            if numIdxTB_Y == 2:
                f.write("\tint idx_" + l_t3_mapping_tb_2D[1][0] + " \t= threadIdx.y % SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[1][0].capitalize() + ";\n")
                f.write("\tint idx_" + l_t3_mapping_tb_2D[1][1] + " \t= threadIdx.y / SIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_tb_2D[1][0].capitalize() + ";\n")
            elif numIdxTB_Y == 1:
                f.write("\tint idx_" + l_t3_mapping_tb_2D[1][0] + " \t= threadIdx.y;\n")
            else:
                print ("[ERROR]!!! The number of indices mapped on TB_Y: ", numIdxTB_Y, " (Not Supported Yet)")
                print ("[ERROR]!!! TB_Y: ", l_t3_mapping_tb_2D[1])
            f.write("\n")

            # block ranges for t3 (in order)
            idx_count = 0
            for t3_idx in l_t3_idx:
                f.write("\tint rng_" + t3_idx + " \t= dev_t3_block_range_" + str(idx_kernel) + "[blockIdx.x * NUM_INDEX + " + str(idx_count) + "];\n")
                idx_count = idx_count + 1
        

    f.write("\n")

    #
    #   Register-Tile
    #   : Depends on SIZE_REG_T
    #
    if opt_data_type == "DOUBLE":
        f.write("\tdouble temp_av;\n")
        if size_reg_y >= size_reg_x:
            f.write("\tdouble temp_bv["  + str(size_reg_y) + "];\n")                             # min(size_reg_y, size_reg_x), basically
            f.write("\tdouble reg_tile[" + str(size_reg_y) + "][" + str(size_reg_x) + "];\n")
        else:
            f.write("\tdouble temp_bv["  + str(size_reg_x) + "];\n")                             # min(size_reg_y, size_reg_x), basically
            f.write("\tdouble reg_tile[" + str(size_reg_y) + "][" + str(size_reg_x) + "];\n")
    else:
        f.write("\tfloat temp_av;\n")
        if size_reg_y >= size_reg_x:
            f.write("\tfloat temp_bv["  + str(size_reg_y) + "];\n")                             # min(size_reg_y, size_reg_x), basically
            f.write("\tfloat reg_tile[" + str(size_reg_y) + "][" + str(size_reg_x) + "];\n")
        else:
            f.write("\tfloat temp_bv["  + str(size_reg_x) + "];\n")                             # min(size_reg_y, size_reg_x), basically
            f.write("\tfloat reg_tile[" + str(size_reg_y) + "][" + str(size_reg_x) + "];\n")
    f.write("\n")

    #   Initializing reg_tile[][]
    f.write("\tfor (int i = 0; i < " + str(size_reg_y) + "; i++)\n")
    f.write("\tfor (int j = 0; j < " + str(size_reg_x) + "; j++)\n")
    f.write("\treg_tile[i][j] = 0.0;\n")
    f.write("\n")

