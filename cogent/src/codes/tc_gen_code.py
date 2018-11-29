#
import src.codes.tc_code_etc                    as tc_code_etc
import src.codes.tc_code_include                as tc_code_include
import src.codes.tc_code_define                 as tc_code_define
import src.codes.tc_code_globalvar              as tc_code_globalvar
#
import src.codes.kernels.tc_code_kernel         as tc_code_kernel
#
import src.codes.others.tc_pre_BasicBlock       as tc_pre_BasicBlock
import src.codes.others.tc_pre_IndirectArray    as tc_pre_IndirectArray
#
import src.codes.kernels.tc_interface           as tc_interface
#
import src.generators.tc_gen                    as tc_gen
import src.generators.tc_helper                 as tc_helper

l_blk_boundary_rng  = list()

#
#   This Function Should 
#
def tc_gen_code_new(tmp_count, str_tmp_count, str_tmp_config, l_inner_groups, l_interface_info, opt_pre_computed, opt_data_type):
    #
    #   FILE: OPEN
    #
    output_name = "temp"
    if str_tmp_config == "-1":
        f = open(output_name + "__" + str_tmp_count + ".cu", "w")
    else:
        f = open(output_name + "__" + str_tmp_count + "__" + str_tmp_config + ".cu", "w")

    #
    #   Includes and Globel Methods
    #
    tc_code_include.tc_code_include(f)
    #tc_code_etc.tc_gen_global_methods(f, len(l_inner_groups))
    
    #
    #   should be changed.
    #
    interface_name  = "sd_t_d2_fusion"
    kernel_name     = "kernel_"

    #
    l_combined_opt_diffs        = list()
    l_combined_opt_gen_fulls    = list()
    l_combined_opt_gen_internal = list()
    l_combined_t3_slices_size   = list()
    l_combined_mappings         = list()

    l_combined_t3_d_decl_var    = list()
    l_combined_t2_d_decl_var    = list()
    l_combined_v2_d_decl_var    = list()
    l_combined_t3_parameters    = list()
    l_combined_t2_parameters    = list()
    l_combined_v2_parameters    = list()
    l_combined_device_dynamic   = list()
    l_combined_host_dynamic     = list()
    l_combined_cuda_malloc      = list()

    #
    #   To Support Multiple Inner-Groups
    #
    for each_inner_group in l_inner_groups:
        l_combined_t3_slices_size.append(each_inner_group[8])
        l_combined_mappings.append([each_inner_group[1], each_inner_group[2]])

    #
    #   Inputs: T3-Slices, T3-Mappings, External Index, Internal Index
    #
    tc_code_define.tc_gen_definition_new(f, l_combined_t3_slices_size, l_combined_mappings, l_inner_groups[0][4], l_inner_groups[0][5])

    #
    #   Each Inner-Group Corresponds to A Kernel (There are Three Types per Kernel)
    #       Type #1: External (Full)    & Internal (Full)
    #       Type #2: External (Full)    & Internal (Partial)
    #       Type #3: External (Partial) & Internal (Full)
    #       Type #4: External (Partial) & Internal (Partial)
    #
    l_var_outputs                   = list()
    l_var_input_internal            = list()

    l_combined_register_mappings    = list()

    l_combined_var_input_left       = list()
    l_combined_var_input_right      = list()
    l_combined_var_outputs_helpers  = list()
    l_combined_var_thread_block     = list()

    l_combined_t3_d_decl_var        = list()
    l_combined_t2_d_decl_var        = list()
    l_combined_v2_d_decl_var        = list()

    l_combined_t3_parameters        = list()
    l_combined_t2_parameters        = list()
    l_combined_v2_parameters        = list()

    l_combined_inputs_int_strides   = list()

    l_cuda_malloc                   = list()
    l_device_dynamic                = list()
    l_host_dynamic                  = list()

    #
    #   To Handle Multiple Tensor Contractions in an Inner-Group
    #
    kernel_number = 1
    for each_inner_group in l_inner_groups:
        #
        '''
        idx_count = 0
        for each_info in each_inner_group:
            print ("> ", idx_count, ": ", each_info)
            idx_count += 1
        '''
        #
        l_var_tensor_block      = list()
        l_var_input_left        = list()
        l_var_input_right       = list()
        l_var_outputs_helpers   = list()

        l_t3_d_decl_var     = list()
        l_t2_d_decl_var     = list()
        l_v2_d_decl_var     = list()

        l_t3_parameters     = list()
        l_t2_parameters     = list()
        l_v2_parameters     = list()

        l_input_strides     = list()

        #   Inputs:     kernel_number, each_inner_group[6](l_input_tensors), each_inner_group[4](l_extenral_index), each_inner_group[5](l_internal_index)
        #   Outputs:    l_t3_d_decl_var, l_t3_parameters, l_t2_d_decl_var, l_t2_parameters, l_v2_d_decl_var, l_v2_parameters,
        #               l_cuda_malloc, l_device_dynami,c
        #               l_var_tensor_block, l_var_outputs, l_var_outputs_helpers, l_var_input_left, l_var_input_right, l_var_input_internal
        tc_code_globalvar.tc_gen_variables(kernel_number, l_interface_info,
                                                        each_inner_group[6],    each_inner_group[4],    each_inner_group[5],
                                                        l_t3_d_decl_var,        l_t3_parameters, 
                                                        l_t2_d_decl_var,        l_t2_parameters,
                                                        l_v2_d_decl_var,        l_v2_parameters,
                                                        l_input_strides,
                                                        l_cuda_malloc,          l_device_dynamic,
                                                        l_var_tensor_block,     l_var_outputs,          l_var_outputs_helpers, l_var_input_left, l_var_input_right, l_var_input_internal,
                                                        opt_data_type)

        #
        #   Variables are Used in Functions for Pre-Computed Arrays and In-Direction Arrays. (Finally, Kernels)
        #   : We need to differentiate them to be used in these functions.
        #

        #
        #   Tile-Appoach:
        #       Inputs: kernel_number, l_interface_info, each_inner_group[4](l_external_index), each_inner_group[8](l_t3_slices), each_inner_group[5](l_internal_index), each_inner_group[3](l_idx_size)
        #       Output: l_host_dynamic 
        #
        if opt_pre_computed != -1:
            tc_pre_BasicBlock.tc_gen_code_pre_TileApproach(f,   kernel_number,          l_interface_info, 
                                                                each_inner_group[4],    each_inner_group[8], 
                                                                each_inner_group[3],    each_inner_group[5],
                                                                l_host_dynamic)

            #
            #   Pre-Compuated Arrays and In-Direct Arrays
            #
            tc_pre_IndirectArray.tc_gen_code_driver_PreComputedArray(f, kernel_number,  l_interface_info,    l_var_outputs_helpers, 
                                                                                        l_var_input_left,    l_var_input_right,     l_var_tensor_block,
                                                                                        each_inner_group[6], l_host_dynamic,        each_inner_group[4],    
                                                                                        each_inner_group[4], each_inner_group[2],   each_inner_group[0])

        #   Related to Kernel(s)
        size_smem_left, size_smem_right, str_left, str_right = tc_interface.tc_interface_SMEM_Size(each_inner_group[6], each_inner_group[4], each_inner_group[5], each_inner_group[8], each_inner_group[2])

        #
        size_TB_X, size_TB_Y        = tc_interface.tc_interface_TB_Size(each_inner_group[1][0], each_inner_group[1][1], each_inner_group[8])
        size_smem_internal          = tc_helper.tc_gen_helper_CheckingIntUnit(each_inner_group[4], each_inner_group[8], each_inner_group[5])
        size_REG_X                  = tc_helper.tc_gen_helper_find(each_inner_group[8], each_inner_group[2][0])	# 0 -> REG_X
        size_REG_Y                  = tc_helper.tc_gen_helper_find(each_inner_group[8], each_inner_group[2][1])	# 1 -> REG_Y
        opt_load_t2, opt_load_v2    = tc_helper.tc_gen_helper_CheckingInternalFVI(each_inner_group[6], each_inner_group[5])

        #print ("size_smem_internal: ", size_smem_internal)

        #
        #   Constratins:""
        #
        tc_gen.tc_gen_Constraints(f, size_TB_X, size_TB_Y, size_smem_left, size_smem_right, size_smem_internal)

        #
        opt_shared_padding = 0

        #
        #   Kernels: Different Types (External, Internal)
        #               (1) Full,       Full
        #               (2) Full,       Partial
        #               (3) Partial,    Full
        #               (4) Partial,    Partial
        #
        
        tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_1_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                            l_input_strides,
                                            each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                            size_smem_left, size_smem_right, size_smem_internal,
                                            size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                            l_blk_boundary_rng,
                                            -1, -1, opt_load_t2, opt_load_v2, opt_pre_computed, 1, opt_data_type,
                                            opt_shared_padding,
                                            kernel_number)
        
        tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_2_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                            l_input_strides,
                                            each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                            size_smem_left, size_smem_right, size_smem_internal,
                                            size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                            l_blk_boundary_rng,
                                            1, -1, opt_load_t2, opt_load_v2, opt_pre_computed, 1, opt_data_type,
                                            opt_shared_padding,
                                            kernel_number)

        tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_3_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                            l_input_strides,
                                            each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                            size_smem_left, size_smem_right, size_smem_internal,
                                            size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                            l_blk_boundary_rng,
                                            -1, 1, opt_load_t2, opt_load_v2, opt_pre_computed, 1, opt_data_type,
                                            opt_shared_padding,
                                            kernel_number)
        
        tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_4_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                            l_input_strides,
                                            each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                            size_smem_left, size_smem_right, size_smem_internal,
                                            size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                            l_blk_boundary_rng,
                                            1, 1, opt_load_t2, opt_load_v2, opt_pre_computed, 1, opt_data_type,
                                            opt_shared_padding,
                                            kernel_number)
        

        #
        #   multiple internal indices.
        #
        if len(each_inner_group[5]) > 1:
            
            tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_1_tex_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                                l_input_strides,
                                                each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                                size_smem_left, size_smem_right, size_smem_internal,
                                                size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                                l_blk_boundary_rng,
                                                -1, -1, opt_load_t2, opt_load_v2, opt_pre_computed, 2, opt_data_type,
                                                opt_shared_padding,
                                                kernel_number)
            
            tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_2_tex_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                                l_input_strides,
                                                each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                                size_smem_left, size_smem_right, size_smem_internal,
                                                size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                                l_blk_boundary_rng,
                                                1, -1, opt_load_t2, opt_load_v2, opt_pre_computed, 2, opt_data_type,
                                                opt_shared_padding,
                                                kernel_number)

            tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_3_tex_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                                l_input_strides,
                                                each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                                size_smem_left, size_smem_right, size_smem_internal,
                                                size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                                l_blk_boundary_rng,
                                                -1, 1, opt_load_t2, opt_load_v2, opt_pre_computed, 2, opt_data_type,
                                                opt_shared_padding,
                                                kernel_number)
            
            tc_code_kernel.tc_gen_code_Kernel(f, kernel_name + "_4_tex_" + str(kernel_number), l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                                l_input_strides,
                                                each_inner_group[7], each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[5], each_inner_group[8],
                                                size_smem_left, size_smem_right, size_smem_internal,
                                                size_REG_Y, size_REG_X, size_TB_Y, size_TB_X, str_left, str_right,
                                                l_blk_boundary_rng,
                                                1, 1, opt_load_t2, opt_load_v2, opt_pre_computed, 2, opt_data_type,
                                                opt_shared_padding,
                                                kernel_number)


        # 
        #   For the Interface
        #
        l_combined_var_input_left.append(l_var_input_left)
        l_combined_var_input_right.append(l_var_input_right)
        l_combined_var_outputs_helpers.append(l_var_outputs_helpers)
        l_combined_var_thread_block.append(l_var_tensor_block)
        l_combined_t3_d_decl_var.append(l_t3_d_decl_var)
        l_combined_t2_d_decl_var.append(l_t2_d_decl_var)
        l_combined_v2_d_decl_var.append(l_v2_d_decl_var)
        l_combined_t3_parameters.append(l_t3_parameters)
        l_combined_t2_parameters.append(l_t2_parameters)
        l_combined_v2_parameters.append(l_v2_parameters)
        l_combined_register_mappings.append(each_inner_group[2])
        l_combined_inputs_int_strides.append(l_input_strides)

        #
        kernel_number = kernel_number + 1
        #
        #   End of For-Statement: l_innter_groups
        #

    #
    #   Parts of Kernels for Register Transpose
    #   (Currently) It assumes that all inner groups can be grouped.
    #
    if tc_gen.tc_gen_Check_RegisterTranspose(l_inner_groups) == 1:
        print ("[Code Generator][Code] Register Transpose [Possible] according to the given mappings")
        f.write("\n")
        f.write("// This part is for Kernels which support Register Transpose\n")

        #
        #   This is also 4-different types.   
        #   
        tc_code_kernel_fusion.tc_gen_code_Kernel_Register_Transpose(f, kernel_name + "_4", l_inner_groups, l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, 1, 1)
        tc_code_kernel_fusion.tc_gen_code_Kernel_Register_Transpose(f, kernel_name + "_3", l_inner_groups, l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, 1, -1)
        tc_code_kernel_fusion.tc_gen_code_Kernel_Register_Transpose(f, kernel_name + "_2", l_inner_groups, l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, -1, 1)
        tc_code_kernel_fusion.tc_gen_code_Kernel_Register_Transpose(f, kernel_name + "_1", l_inner_groups, l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, -1, -1)
    else:
        print ("[Code Generator][Code] Register Transpose [Impossible] according to the given mappings")

    #
    #   Temp...... For |K| > 1,
    #
    l_internal_addrs = list()
    for each_inner_group in l_inner_groups:
        #
        #   Each Inner-Group can have multiple Tensor Contractions which will be Fused.
        #   Among Inner-Groups, there might be possible to group them by using Register-Transposition.
        #
        print  ("each_inner_group: ", each_inner_group)
        l_tensor_contraction_in_inner_group = each_inner_group[6]
        
        #   To-Do: Need To Suport Multiple-Tensor Contractions
        for each_tensor_contraction in l_tensor_contraction_in_inner_group:
            info_input_left     = each_tensor_contraction[0]
            info_input_right    = each_tensor_contraction[1]
            
            # LEFT
            l_idx_info_left     = list()
            for idx_tensor in info_input_left[1]:
                #
                if tc_helper.tc_gen_helper_find_1d(l_inner_groups[0][5], idx_tensor) != -1:
                    l_idx_info_left.append([1, idx_tensor])
                else:
                    l_idx_info_left.append([0, idx_tensor])

            # 
            l_rev_idx_info_left = list(reversed(l_idx_info_left))
            str_addr_left       = ""
            idx_count           = 0
            for each_idx_info in l_rev_idx_info_left:
                #
                if each_idx_info[0] == 1:       # internal index
                    if idx_count == 0:
                        str_addr_left = "idx_" + each_idx_info[1]
                    else:
                        str_addr_left = "idx_" + each_idx_info[1] + " + (" + str_addr_left + ") * size_" + each_idx_info[1]
                    #
                    idx_count = idx_count + 1
                else:                           # external index
                    if idx_count != 0:
                        str_addr_left = "(" + str_addr_left + ") * size_" + each_idx_info[1]

            # RIGHT
            str_addr_right      = ""
            l_idx_info_right    = list()
            for idx_tensor in info_input_right[1]:
                #
                if tc_helper.tc_gen_helper_find_1d(l_inner_groups[0][5], idx_tensor) != -1:
                    l_idx_info_right.append([1, idx_tensor])
                else:
                    l_idx_info_right.append([0, idx_tensor])

            #
            l_rev_idx_info_right    = list(reversed(l_idx_info_right))
            str_addr_right          = ""
            idx_count               = 0
            for each_idx_info in l_rev_idx_info_right:
                #
                if each_idx_info[0] == 1:       # internal index
                    if idx_count == 0:
                        str_addr_right = "idx_" + each_idx_info[1]
                    else:
                        str_addr_right = "idx_" + each_idx_info[1] + " + (" + str_addr_right + ") * size_" + each_idx_info[1]
                    #
                    idx_count = idx_count + 1
                else:                           # external index
                    if idx_count != 0:
                        str_addr_right = "(" + str_addr_right + ") * size_" + each_idx_info[1]

            #
            l_internal_addrs.append([str_addr_left, str_addr_right])
        #
    
    #
    #   Drivers
    #
    tc_interface.tc_gen_code_kernel_caller(f,   interface_name,         kernel_name,        l_interface_info,       
                                                l_inner_groups[0][4],   l_inner_groups[0][5],
                                                l_inner_groups[0][10],
                                                l_var_tensor_block,     l_var_outputs,      l_var_outputs_helpers,  
                                                l_var_input_left,       l_var_input_right,  l_var_input_internal,
                                                l_combined_var_input_left, 
                                                l_combined_var_input_right, 
                                                l_combined_var_outputs_helpers, 
                                                l_combined_var_thread_block,
                                                l_combined_register_mappings,
                                                l_internal_addrs,               ## ADDEDEDEDE
                                                l_combined_inputs_int_strides,  ##
                                                l_cuda_malloc,          l_device_dynamic,   l_host_dynamic,    
                                                l_t3_parameters,        l_t2_parameters,    l_v2_parameters,
                                                l_combined_t3_parameters, l_combined_t2_parameters, l_combined_v2_parameters,
                                                opt_pre_computed, opt_data_type)

    #
    '''
    idx_inner_count = 0
    for each_inner in l_inner_groups:
        idx_count = 0
        for each_info in each_inner:
            print ("[", idx_inner_count, "][", idx_count, "] each_info: ", each_info)
            idx_count = idx_count + 1
        idx_inner_count = idx_inner_count + 1
    '''
    
    #
    l_tile_sizes                        = l_inner_groups[0][8]
    l_split_representative_problem_size = l_inner_groups[0][9]

    #l_split_representative_problem_size = l_inner_groups[0][9]
    #print ("l_split_representative_problem_size: ", l_split_representative_problem_size)
    #print ("l_tile_sizes: ", l_tile_sizes)
    
    #
    #   "Interface"
    #
    tc_interface.tc_gen_code_interface(f, interface_name, l_interface_info, l_tile_sizes, l_split_representative_problem_size, opt_data_type)

    #
    #   FILE: CLOSE
    #
    f.close()

#
#   This function should be per a Inner-Group.
#
def tc_gen_code(tmp_count, inner_groups):
    #
    #   This is for calculating cost function based on a given input.
    #
    output_name = "temp"            # Depends on # of groups.

    #
    #   FILE OPEN: A LIST of Inner-Groups is for A SINGLE CUDA File.
    #            : Each Inner-Group has multiple Tensor Contractions which will be fused.
    #            : Thus, each Inner-Group is for A SINGLE KERNEL.
    #
    f = open(output_name + "_" + str(tmp_count) + ".cu", "w")

    #
    #   Per Inner-Groups,
    #
    tc_code_include.tc_code_include(f)
    #tc_code_etc.tc_gen_global_methods(f, len(inner_groups))

    #
    l_combined_opt_diffs        = list()
    l_combined_opt_gen_fulls    = list()
    l_combined_opt_gen_internal = list()
    l_combined_input_tensors    = list()
    l_combined_t3_slices_size   = list()
    l_combined_mappings         = list()

    l_combined_t3_d_decl_var    = list()
    l_combined_t2_d_decl_var    = list()
    l_combined_v2_d_decl_var    = list()
    l_combined_t3_parameters    = list()
    l_combined_t3_parameters_f  = list()
    l_combined_t3_parameters_nf = list()
    l_combined_t2_parameters    = list()
    l_combined_t2_parameters_f  = list()
    l_combined_t2_parameters_nf = list()
    l_combined_v2_parameters    = list()
    l_combined_v2_parameters_f  = list()
    l_combined_v2_parameters_nf = list()
    l_combined_device_dynamic   = list()
    l_combined_host_dynamic     = list()
    l_combined_cuda_malloc      = list()

    #
    #   To Support Multiple Inner-Groups
    #
    for each_inner_group in inner_groups:
        l_combined_input_tensors.append(each_inner_group[6])
        l_combined_t3_slices_size.append(each_inner_group[8])
        l_combined_mappings.append([each_inner_group[1], each_inner_group[2]])

    #
    tc_code_define.tc_gen_definition(f,     l_combined_t3_slices_size,  inner_groups[0][3], inner_groups[0][4],
                                            l_combined_mappings,        inner_groups[0][4], inner_groups[0][5],
                                            l_combined_input_tensors)
    #
    #   Check Types
    #
    idx_kernel = 1
    for each_inner_group in inner_groups:
        opt_gen_full, opt_gen_p7, possible_diff = tc_helper.tc_gen_helper_CheckingTypes(each_inner_group[3], each_inner_group[8], each_inner_group[4])

        # possible_diff, opt_gen_full, and opt_gen_p7 are created at here.
        #l_combined_opt_diffs.append(possible_diff)
        l_combined_opt_diffs.append(-1)
        l_combined_opt_gen_fulls.append(opt_gen_full)
        l_combined_opt_gen_internal.append(opt_gen_p7) 

        print ("[Code Generator][tc_gen_code] Kernel #", idx_kernel, ">>> opt_diff:", possible_diff, "(but -1), opt_gen_full:", opt_gen_full, ", opt_gen_p7:", opt_gen_p7)
        idx_kernel = idx_kernel + 1
        
    #
    #   Code: Global Variables (Common)
    #
    tc_code_globalvar.tc_gen_global_variables_common(f)

    #
    idx_kernel = 0
    for each_inner_group in inner_groups:
        #
        l_t3_d_decl_var     = list()
        l_t2_d_decl_var     = list()
        l_v2_d_decl_var     = list()

        l_t3_parameters     = list()
        l_t2_parameters     = list()
        l_v2_parameters     = list()

        l_t3_parameters_nf  = list()
        l_t2_parameters_nf  = list()
        l_v2_parameters_nf  = list()

        l_t3_parameters_f   = list()
        l_t2_parameters_f   = list()
        l_v2_parameters_f   = list()

        #
        #   Code: Global Variables (Tensor Contractions)
        #
        tc_code_globalvar.tc_gen_global_variables(f,    each_inner_group[6],        each_inner_group[4],    each_inner_group[5],
                                                        l_t3_d_decl_var,            l_t3_parameters,        l_t3_parameters_nf,     l_t2_parameters_nf, l_v2_parameters_nf,
                                                                                    l_t3_parameters_f,      l_t2_parameters_f,      l_v2_parameters_f,
                                                        l_device_dynamic,           l_t2_d_decl_var,        l_v2_d_decl_var,        l_t2_parameters,    l_v2_parameters,
                                                        l_cuda_malloc,              
                                                        l_combined_opt_diffs[idx_kernel], idx_kernel + 1)
        #
        l_combined_t3_d_decl_var.append(l_t3_d_decl_var)
        l_combined_t2_d_decl_var.append(l_t2_d_decl_var)
        l_combined_v2_d_decl_var.append(l_v2_d_decl_var)
        l_combined_t3_parameters.append(l_t3_parameters)
        l_combined_t2_parameters.append(l_t2_parameters)
        l_combined_v2_parameters.append(l_v2_parameters)
        l_combined_t3_parameters_f.append(l_t3_parameters_f)
        l_combined_t2_parameters_f.append(l_t2_parameters_f)
        l_combined_v2_parameters_f.append(l_v2_parameters_f)
        l_combined_t3_parameters_nf.append(l_t3_parameters_nf)
        l_combined_t2_parameters_nf.append(l_t2_parameters_nf)
        l_combined_v2_parameters_nf.append(l_v2_parameters_nf)

        #
        idx_kernel = idx_kernel + 1
        
    #
    #   Code: SD2 Functions
    #
    idx_kernel = 1
    for each_inner_group in inner_groups:
        tc_pre_SD2_Functions.tc_gen_code_pre_SD2_Functions(f, each_inner_group[4], each_inner_group[5], each_inner_group[6], l_host_dynamic, idx_kernel)
        idx_kernel = idx_kernel + 1
    
    #
    #   Code: CUDA Malloc | Memcpy
    #
    tc_pre_CUDA_Malloc.tc_gen_code_pre_CUDA_Malloc(f, l_cuda_malloc, l_t3_parameters, l_t2_parameters, l_v2_parameters, l_device_dynamic, inner_groups[0][5])

    #
    #   Code: Pre-Computed Arrays
    #
    idx_kernel = 1
    for each_inner_group in inner_groups:
        tc_pre_IndirectArray.tc_gen_code_pre_IndirectArray(f,   each_inner_group[4],  each_inner_group[0],  each_inner_group[6],    each_inner_group[2],
                                                                each_inner_group[4],  each_inner_group[5],  l_host_dynamic,         l_combined_opt_diffs[idx_kernel - 1],          
                                                                idx_kernel)
        idx_kernel = idx_kernel + 1
    
    #
    #   Code: BasicBlock
    #
    idx_kernel = 1
    for each_inner_group in inner_groups:
        tc_pre_BasicBlock.tc_gen_code_pre_BasicBlock(f, each_inner_group[4], each_inner_group[3], each_inner_group[8],                  l_host_dynamic,   
                                                        each_inner_group[4], each_inner_group[5], l_combined_opt_diffs[idx_kernel - 1], idx_kernel)
        idx_kernel = idx_kernel + 1

    #
    print ("[Code Generator][tc_gen_code] # of Inner-Groups: ", len(inner_groups))

    #
    #   Each Inner-Group Corresponds to A Kernel.
    #
    idx_kernel = 1
    for an_inner_group in inner_groups:
        print ("[Code Generator][tc_gen_code] Creating Kernel --- #", idx_kernel)
        #
        kernel_name     = "kernel_ccsdT_" + str(idx_kernel)    # Depends on # of groups.

        #   Options for Each Kernel
        possible_diff   = l_combined_opt_diffs[idx_kernel - 1]
        opt_gen_full    = l_combined_opt_gen_fulls[idx_kernel - 1]
        opt_gen_p7      = l_combined_opt_gen_internal[idx_kernel -1]

        #
        #   To-Do: check if boundaries are needed by all tensor contractions, or not.
        #
        if opt_gen_full != -1:
            #   Inputs:     l_blk_boundary_rng, l_idx_size, l_t3_slices, l_t3_mapping_reg, l_t3_mapping_tb_2D, info_left_index, info_right_index
            #   Outputs:    l_blk_boundary_rng
            tc_helper.tc_gen_helper_CheckingBoundary(l_blk_boundary_rng, an_inner_group[3], an_inner_group[8], an_inner_group[2], an_inner_group[1], an_inner_group[6][0][0][1], an_inner_group[6][0][1][1])
            print (">>> Boundaries for External Indices: ", l_blk_boundary_rng)

        # (....)
        #possible_diff = -1  # (To-Do: )

        #
        #   Inputs: l_input_tensors, l_internal_idx, l_external_idx, l_t3_slices, l_t3_mapping_reg,
        #   Outputs: int_size_sm_a, int_size_sm_b, int_str_t2, int_str_v2
        #
        for each_tc in an_inner_group[6]:
            #
            int_size_sm_a   = 1
            int_size_sm_b   = 1
            int_str_t2      = 1
            int_str_v2      = 1

            #
            bool_found      = 1
            for each_idx in each_tc[0][1]:
                # for size_sm_a
                if tc_helper.tc_gen_helper_find_1d(an_inner_group[5], each_idx) == -1:
                    int_size_sm_a = int_size_sm_a * tc_helper.tc_gen_helper_find(an_inner_group[8], each_idx)

                # for str_str_t2
                if tc_helper.tc_gen_helper_find_1d(an_inner_group[4], each_idx) != -1:                          # external indices (all)
                    if tc_helper.tc_gen_helper_find_1d(an_inner_group[2], each_idx) == -1:                      # external indices mapped on ! regiter tiling
                        int_str_t2 = int_str_t2 * tc_helper.tc_gen_helper_find(an_inner_group[8], each_idx)

            bool_found = 1
            for each_idx in each_tc[1][1]:
                # for size_sm_b
                if tc_helper.tc_gen_helper_find_1d(an_inner_group[5], each_idx) == -1:
                    int_size_sm_b = int_size_sm_b * tc_helper.tc_gen_helper_find(an_inner_group[8], each_idx)

                # for str_str_v2
                if tc_helper.tc_gen_helper_find_1d(an_inner_group[4], each_idx) != -1:                          # external indices (all)
                    if tc_helper.tc_gen_helper_find_1d(an_inner_group[2], each_idx) == -1:                      # external indices mapped on ! register tiling
                        int_str_v2 = int_str_v2 * tc_helper.tc_gen_helper_find(an_inner_group[8], each_idx)

        #
        #   Inputs:     l_t3_mapping_tb_2D, l_t3_slices
        #   Outputs:    int_size_tb_x, int_size_tb_y
        #
        int_size_tb_x = 1
        int_size_tb_y = 1
        for each_idx in an_inner_group[1][0]:  # "x"-axis
            int_size_tb_x = int_size_tb_x * tc_helper.tc_gen_helper_find(an_inner_group[8], each_idx)

        for each_idx in an_inner_group[1][1]:  # "y"-axis_idx
            int_size_tb_y = int_size_tb_y * tc_helper.tc_gen_helper_find(an_inner_group[8], each_idx)

        # the below information should be produced by "tc_gen_input()"
        size_sm_a       = int_size_sm_a #+ 1 (padding)
        size_sm_b       = int_size_sm_b #+ 1
        size_tb_x       = int_size_tb_x
        size_tb_y       = int_size_tb_y
        size_sm_p7      = tc_helper.tc_gen_helper_CheckingIntUnit(an_inner_group[4], an_inner_group[8], an_inner_group[5])
        size_reg_y      = tc_helper.tc_gen_helper_find(an_inner_group[8], an_inner_group[2][1])
        size_reg_x      = tc_helper.tc_gen_helper_find(an_inner_group[8], an_inner_group[2][0])

        #
        #   Options for two inputs
        #   Inputs: l_input_tensors, l_internal_idx
        #
        opt_load_t2, opt_load_v2 = tc_helper.tc_gen_helper_CheckingInternalFVI(an_inner_group[6], an_inner_group[5])

        #
        #   Constraints
        #   Inputs: f, size_tb_x, size_tb_y, size_sm_a, size_sm_b, size_sm_p7
        #
        tc_gen_Constraints(f, size_tb_x, size_tb_y, size_sm_a, size_sm_b, size_sm_p7)

        #
        l_t3_d_decl_var     = l_combined_t3_d_decl_var[idx_kernel - 1]
        l_t2_d_decl_var     = l_combined_t2_d_decl_var[idx_kernel - 1]
        l_v2_d_decl_var     = l_combined_v2_d_decl_var[idx_kernel - 1]
        l_t3_parameters     = l_combined_t3_parameters[idx_kernel - 1]
        l_t2_parameters     = l_combined_t2_parameters[idx_kernel - 1]
        l_v2_parameters     = l_combined_v2_parameters[idx_kernel - 1]
        l_t3_parameters_f   = l_combined_t3_parameters_f[idx_kernel - 1]
        l_t2_parameters_f   = l_combined_t2_parameters_f[idx_kernel - 1]
        l_v2_parameters_f   = l_combined_v2_parameters_f[idx_kernel - 1]
        l_t3_parameters_nf  = l_combined_t3_parameters_nf[idx_kernel - 1]
        l_t2_parameters_nf  = l_combined_t2_parameters_nf[idx_kernel - 1]
        
        #
        #   >>> Create Kernels <<<
        #
        if possible_diff == -1:
            tc_code_kernel.tc_gen_code_Kernel(  f,                  kernel_name,        l_t3_d_decl_var,    l_t2_d_decl_var,    l_v2_d_decl_var,
                                                an_inner_group[7],  an_inner_group[1],  an_inner_group[2],  an_inner_group[4],  an_inner_group[5],  an_inner_group[8],
                                                size_sm_a,          size_sm_b,          size_sm_p7,
                                                size_reg_y,         size_reg_x,         size_tb_y,          size_tb_x,          int_str_t2,         int_str_v2,
                                                l_blk_boundary_rng,
                                                opt_gen_p7,         opt_gen_full,       opt_load_t2,        opt_load_v2,        idx_kernel)
        else:
            tc_code_kernel.tc_gen_code_Kernel(  f,                  kernel_name,        l_t3_d_decl_var,    l_t2_d_decl_var,    l_v2_d_decl_var,
                                                an_inner_group[7],  an_inner_group[1],  an_inner_group[2],  an_inner_group[4],  an_inner_group[5],  an_inner_group[8],
                                                size_sm_a,          size_sm_b,          size_sm_p7,
                                                size_reg_y,         size_reg_x,         size_tb_y,          size_tb_x,          int_str_t2,         int_str_v2,
                                                l_blk_boundary_rng,
                                                opt_gen_p7,         opt_gen_full,       opt_load_t2,        opt_load_v2,        idx_kernel)
            opt_gen_full    = -1    # 1 or -1
            tc_code_kernel.tc_gen_code_Kernel(  f,                  kernel_name + "_full",  l_t3_d_decl_var,    l_t2_d_decl_var,    l_v2_d_decl_var,
                                                an_inner_group[7],  an_inner_group[1],      an_inner_group[2],  an_inner_group[4],  an_inner_group[5],  an_inner_group[8],
                                                size_sm_a,          size_sm_b,              size_sm_p7,
                                                size_reg_y,         size_reg_x,             size_tb_y,          size_tb_x,          int_str_t2,         int_str_v2,
                                                l_blk_boundary_rng,
                                                opt_gen_p7,         opt_gen_full,           opt_load_t2,        opt_load_v2,        idx_kernel)

        #
        idx_kernel = idx_kernel + 1

    #
    #   Code: Function to Call Kernels.
    #
    kernel_name     = "kernel_ccsdT"
    tc_code_etc.tc_gen_code_fusedKernels(f, kernel_name,    l_combined_t3_parameters,       l_combined_t2_parameters,    l_combined_v2_parameters,
                                                            l_combined_t3_parameters_nf,    l_combined_t2_parameters_nf, l_combined_v2_parameters_nf,
                                                            l_combined_t3_parameters_f,     l_combined_t2_parameters_f,  l_combined_v2_parameters_f,
                                                            l_combined_opt_diffs,           len(inner_groups))

    #
    #   Code: Function for Correctness Check.
    #
    tc_post_Correctness.tc_gen_code_Post_Correctness(f, an_inner_group[4], an_inner_group[5], l_combined_input_tensors)

    #
    #   Code: Delete Device Memory Allocated Dynamically.
    #
    tc_post_HostDevice_Free.tc_gen_code_post_CUDA_Free(f, l_cuda_malloc)

    #
    #   Code: Delete Host Memory Allocated Dynamically.
    #
    tc_post_HostDevice_Free.tc_gen_code_post_HostFree(f, l_host_dynamic)

    #
    #   Code: Main Function
    #
    f.write("// # of Inner-Groups: " + str(len(inner_groups)) + "\n")
    tc_code_etc.tc_gen_code_main(f, len(inner_groups))

    # FILE CLOSE
    f.close()
    #
    #   END of "tc_gen_code"
    #
