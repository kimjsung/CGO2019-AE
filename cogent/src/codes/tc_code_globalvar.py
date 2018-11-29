import src.generators.tc_helper     as tc_helper

#
def tc_gen_global_variables_common(f):
    #
    #   This is for Common Global Variables among Inner-Groups
    #
    f.write("// Common Global Variables\n")
    #
    #   To-Do: The Size of "Unit" for Internal Indices might be different among Inner-Groups as well as Tensor Contractions in an Inner-Group.
    #
    tc_gen_code_helper_varible(f, "int", "size_internal")

    # for output
    tc_gen_code_helper_varible(f, "int", "size_T3")

#
#   
#
def tc_gen_variables(kernel_number,         l_interface_info,  
                    l_input_tensors,        l_external_idx,         l_internal_idx,
                    l_t3_d_decl_var,        l_t3_parameters, 
                    l_t2_d_decl_var,        l_t2_parameters,
                    l_v2_d_decl_var,        l_v2_parameters,
                    l_input_strides,
                    l_cuda_malloc,          l_device_dynamic,
                    l_var_thread_blocks,    l_var_outputs,      l_var_outputs_helpers,  l_var_input_left, l_var_input_right, l_var_internal,
                    opt_data_type):
    #
    #
    #   1. # of Thread-Blocks
    l_var_thread_blocks.append(["int", "num_thread_blocks_kernel_" + str(kernel_number)])

    #   2. Outputs
    tc_gen_variables_outputs(kernel_number, l_interface_info, l_external_idx, l_t3_d_decl_var, l_t3_parameters, l_cuda_malloc, l_device_dynamic, l_var_outputs, opt_data_type)

    #   3. Outputs-Helpers
    tc_gen_variables_outputs_helpers(kernel_number, l_t3_d_decl_var, l_t3_parameters, l_cuda_malloc, l_device_dynamic, l_var_outputs_helpers)
    
    #   4. Inputs
    for each_input in l_input_tensors:
        #print ("each_input:", each_input)
        #
        #   Left
        #
        tc_gen_variables_input_left(kernel_number, each_input, l_external_idx, l_t2_d_decl_var, l_t2_parameters, l_cuda_malloc, l_device_dynamic, l_var_input_left, opt_data_type)

        #
        #   Right
        #
        tc_gen_variables_input_right(kernel_number, each_input, l_external_idx, l_v2_d_decl_var, l_v2_parameters, l_cuda_malloc, l_device_dynamic, l_var_input_right, opt_data_type)

        #
        #   |K| > 1
        #
        if len(l_internal_idx) > 1:
            tc_gen_variables_input_internal(l_internal_idx, l_var_internal)
        #
        #   |K| == 1
        #
        elif len(l_internal_idx) == 1:
            #
            #
            #
            tmp_input_left      = each_input[0]
            tmp_input_right     = each_input[1]
            str_stride_left     = ""
            str_stride_right    = ""

            #
            idx_count = 0
            for each_idx in tmp_input_left[1]:
                if each_idx == l_internal_idx[0]:
                    break
                else:
                    if idx_count == 0:
                        str_stride_left = "size_" + each_idx
                    else:
                        str_stride_left = str_stride_left + " * size_" + each_idx
                    idx_count = idx_count + 1
            
            #
            if idx_count == 0:
                str_stride_left = "1"

            #                
            idx_count = 0
            for each_idx in tmp_input_right[1]:
                if each_idx == l_internal_idx[0]:
                    break
                else:
                    if idx_count == 0:
                        str_stride_right = "size_" + each_idx
                    else:
                        str_stride_right = str_stride_right + " * size_" + each_idx
                    idx_count = idx_count + 1

            #
            if idx_count == 0:
                str_stride_right = "1"

            #
            #   Assumption: Inputs' name are different.
            #
            l_input_strides.append(["stride_int_" + tmp_input_left[0], str_stride_left, "stride_int_" + tmp_input_right[0], str_stride_right])

#   To-Do: Need to differentiate all parameters as inputs and outputs.
def tc_gen_global_variables(f,  l_input_tensors,    l_external_idx,     l_internal_idx,
                                l_t3_d_decl_var,    l_t3_parameters,
                                l_t3_parameters_nf, l_t2_parameters_nf, l_v2_parameters_nf,
                                l_t3_parameters_f,  l_t2_parameters_f,  l_v2_parameters_f,
                                l_device_dynamic,   l_t2_d_decl_var,    l_v2_d_decl_var,    l_t2_parameters, l_v2_parameters,
                                l_cuda_malloc,      possible_diff,      kernel_number):
    #
    str_size_T3_all = ""
    str_size_T3_blk = ""
    idx_count       = 0
    for each_idx in l_external_idx:
        if idx_count != 0:
            str_size_T3_blk = str_size_T3_blk + " * "
            str_size_T3_all = str_size_T3_all + " * "
        str_size_T3_blk = str_size_T3_blk + "SIZE_SLICE_" + str(kernel_number) + "_" + each_idx.capitalize()
        str_size_T3_all = str_size_T3_all + "SIZE_IDX_" + each_idx.capitalize()
        idx_count = idx_count + 1

    # Global - Variables
    f.write("\n")
    f.write("// created by tc_gen_global_variables()\n")

    #
    #   (Global) Variables for Sizes
    #
    tc_gen_global_variables_sizes(f, l_input_tensors, possible_diff, kernel_number)

    #
    #   (Global) Variables for Output Inself
    #
    tc_gen_global_variables_outputs(f,  possible_diff,      kernel_number,                              # Input
                                        l_t3_d_decl_var,                                                # Outputs
                                        l_t3_parameters,    l_t3_parameters_nf, l_t3_parameters_f,      # Outputs
                                        l_cuda_malloc,      l_device_dynamic)                           # Outputs

    #
    #   (Global) Variables for Arrays related to Output
    #
    tc_gen_global_variables_outputs_helpers(f,  possible_diff,      kernel_number,                          # Input
                                                l_t3_parameters,    l_t3_parameters_nf, l_t3_parameters_f,  # Outputs
                                                l_cuda_malloc,      l_device_dynamic,   l_t3_d_decl_var)    # Outputs

    # >>>>>>>>>>>>> To-Do: Inner-Group
    #. For Each Tensor Contraction
    #. Data Structure: l_input_tensors.append(((("t2_1"), ("p4","p7","h1","h2")), (("v2_1"), ("p6","p7","h3","p5"))))
    for each_input in l_input_tensors:
        #
        #   (Global) Variables For Left
        #
        tc_gen_global_variables_outputs_input_left(f,   each_input,         l_external_idx,     possible_diff,
                                                        l_t2_d_decl_var,    l_t2_parameters,    l_t2_parameters_nf, l_t2_parameters_f,
                                                        l_cuda_malloc,      l_device_dynamic,   kernel_number)
        #
        #   (Global) Variables For Right
        #
        tc_gen_global_variables_outputs_input_right(f,  each_input,         l_external_idx,     possible_diff,
                                                        l_v2_d_decl_var,    l_v2_parameters,    l_v2_parameters_nf, l_v2_parameters_f,
                                                        l_cuda_malloc,      l_device_dynamic,   kernel_number)
        #
        #
        #
        if len(l_internal_idx) > 1:
            tc_gen_global_variables_outputs_input_internal(f, l_internal_idx)

#
def tc_gen_global_variables_outputs_input_internal(f, l_internal_idx):
    f.write("// Global Variables for Internal Indices\n")
    tc_gen_code_helper_varible(f, "int*", "d_internal_t2_1" + "_offset")
    tc_gen_code_helper_varible(f, "int*", "h_internal_t2_1" + "_offset")
    tc_gen_code_helper_varible(f, "int*", "d_internal_v2_1" + "_offset")
    tc_gen_code_helper_varible(f, "int*", "h_internal_v2_1" + "_offset")

    # Create Constant Memory
    str_size_internal   = ""
    idx_count           = 0
    for each_idx in l_internal_idx:
        if idx_count == 0:
            str_size_internal = "SIZE_IDX_" + each_idx.capitalize()
        else:
            str_size_internal = str_size_internal + " * SIZE_IDX_" + each_idx.capitalize()
        idx_count = idx_count + 1

    f.write("\n")
    tc_gen_code_helper_varible(f, "__constant__ int", "const_internal_t2_1_offset[" + str_size_internal + "]")
    tc_gen_code_helper_varible(f, "__constant__ int", "const_internal_v2_1_offset[" + str_size_internal + "]")

#
def tc_gen_variables_input_internal(l_internal_idx, l_var_internal):
    #
    #
    #     
    l_var_internal.append(["int*", "host_internal_left_offset"])
    l_var_internal.append(["int*", "host_internal_right_offset"])

   # Create Constant Memory
    str_size_internal   = ""
    idx_count           = 0
    for each_idx in l_internal_idx:
        if idx_count == 0:
            str_size_internal = "SIZE_IDX_" + each_idx.capitalize()
        else:
            str_size_internal = str_size_internal + " * SIZE_IDX_" + each_idx.capitalize()
        idx_count = idx_count + 1

    # >> To-Do??
    #f.write("\n")
    #tc_gen_code_helper_varible(f, "__constant__ int", "const_internal_left_offset[" + str_size_internal + "]")
    #tc_gen_code_helper_varible(f, "__constant__ int", "const_internal_left_offset[" + str_size_internal + "]") 

#
def tc_gen_variables_input_left(kernel_number, each_input, l_external_idx, l_t2_d_decl_var, l_t2_parameters, l_cuda_malloc, l_device_dynamic, l_var_input_left, opt_data_type):
    #
    #
    #
    # Left Input
    d_input_name    =  "dev_" + each_input[0][0]
    h_input_name    = "host_" + each_input[0][0]
    input_f_size    = ""
    input_s_size    = ""

    #
    idx_s_count = 0
    idx_f_count = 0
    for each_index in each_input[0][1]:
        if tc_helper.tc_gen_helper_find_1d(l_external_idx, each_index) != -1:
            if idx_f_count == 0:
                input_f_size = "size_"      + each_index
            else:
                input_f_size = "size_"      + each_index + " * " + input_f_size

            if idx_s_count == 0:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize()
            else:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize() + " * " + input_s_size
            idx_s_count = idx_s_count + 1
            idx_f_count = idx_f_count + 1
        else:
            if idx_f_count == 0:
                input_f_size = "size_"      + each_index
            else:
                input_f_size = "size_"      + each_index + " * " + input_f_size
            idx_f_count = idx_f_count + 1 
    
    #
    if opt_data_type == "DOUBLE":
        l_var_input_left.append(["double*", d_input_name])
    else:
        l_var_input_left.append(["float*", d_input_name])
    #l_var_input_left.append(["double*", h_input_name])
    l_var_input_left.append(["int*",    d_input_name + "_addr"])
    l_var_input_left.append(["int*",    h_input_name + "_addr"])
    l_var_input_left.append(["int*",    d_input_name + "_offset"])
    l_var_input_left.append(["int*",    h_input_name + "_offset"])

    #
    if opt_data_type == "DOUBLE":
        l_t2_d_decl_var.append("double* " + d_input_name)
    else:
        l_t2_d_decl_var.append("float* " + d_input_name)
    l_t2_d_decl_var.append("const int* __restrict__ " + d_input_name + "_addr")
    l_t2_d_decl_var.append("const int* __restrict__ " + d_input_name + "_offset")
    
    #
    if opt_data_type == "DOUBLE":
        l_cuda_malloc.append([d_input_name, "double", input_f_size])
    else:
        l_cuda_malloc.append([d_input_name, "float",  input_f_size])
    l_cuda_malloc.append([d_input_name + "_addr",   "int",      input_s_size + " * num_thread_blocks_kernel_" + str(kernel_number)])
    l_cuda_malloc.append([d_input_name + "_offset", "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"])

    #
    if opt_data_type == "DOUBLE":
        l_t2_parameters.append([d_input_name, "double",   input_f_size])
    else:
        l_t2_parameters.append([d_input_name, "float",   input_f_size])
    l_t2_parameters.append([d_input_name + "_addr",         "int",      input_s_size + " * num_thread_blocks_kernel_" + str(kernel_number)])
    l_t2_parameters.append([d_input_name + "_offset",       "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"])

    #
    if opt_data_type == "DOUBLE":
        l_device_dynamic.append(["double",  d_input_name,                h_input_name,               input_f_size])
    else:
        l_device_dynamic.append(["float",  d_input_name,                h_input_name,               input_f_size])
    l_device_dynamic.append(["int",     d_input_name + "_addr",      h_input_name + "_addr",     input_s_size + " * num_thread_blocks_kernel_" + str(kernel_number)])
    l_device_dynamic.append(["int",     d_input_name + "_offset",    h_input_name + "_offset",   "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"])

#
def tc_gen_global_variables_outputs_input_left(f,   each_input,         l_external_idx,     possible_diff,
                                                    l_t2_d_decl_var,    l_t2_parameters,    l_t2_parameters_nf, l_t2_parameters_f,
                                                    l_cuda_malloc,      l_device_dynamic,   kernel_number, 
                                                    opt_data_type):
    f.write("// Global Variables for Left Input\n")
    # Left Input
    d_input_name    = "d_" + each_input[0][0]
    h_input_name    = "h_" + each_input[0][0]
    input_f_size    = ""
    input_s_size    = ""

    #
    idx_s_count = 0
    idx_f_count = 0
    for each_index in each_input[0][1]:
        if tc_helper.tc_gen_helper_find_1d(l_external_idx, each_index) != -1:
            if idx_f_count == 0:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize()
            else:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize() + " * " + input_f_size

            if idx_s_count == 0:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize()
            else:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize() + " * " + input_s_size
            idx_s_count = idx_s_count + 1
            idx_f_count = idx_f_count + 1
        else:
            if idx_f_count == 0:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize()
            else:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize() + " * " + input_f_size
            idx_f_count = idx_f_count + 1

    #
    if opt_data_type == "DOUBLE":
        tc_gen_code_helper_varible(f, "double*",    d_input_name)
        tc_gen_code_helper_varible(f, "double*",    h_input_name)
    else:
        tc_gen_code_helper_varible(f, "float*",    d_input_name)
        tc_gen_code_helper_varible(f, "float*",    h_input_name)

    tc_gen_code_helper_varible(f, "int*",       d_input_name + "_addr")
    tc_gen_code_helper_varible(f, "int*",       h_input_name + "_addr")

    tc_gen_code_helper_varible(f, "int*",       d_input_name + "_offset")
    tc_gen_code_helper_varible(f, "int*",       h_input_name + "_offset")

    #
    if opt_data_type == "DOUBLE":
        l_t2_d_decl_var.append("double* "   + d_input_name)
    else:
        l_t2_d_decl_var.append("float* "   + d_input_name)

    l_t2_d_decl_var.append("const int* __restrict__ "      + d_input_name + "_addr")
    l_t2_d_decl_var.append("const int* __restrict__ "      + d_input_name + "_offset")

    #
    if opt_data_type == "DOUBLE":
        l_cuda_malloc.append((d_input_name,             "double",   input_f_size))
    else:
        l_cuda_malloc.append((d_input_name,             "float",   input_f_size))

    l_cuda_malloc.append((d_input_name + "_addr",   "int",      input_s_size + " * n_blks_" + str(kernel_number)))
    l_cuda_malloc.append((d_input_name + "_offset", "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

    #
    if opt_data_type == "DOUBLE":
        l_t2_parameters.append((d_input_name,             "double",   input_f_size))
    else:
        l_t2_parameters.append((d_input_name,             "float",   input_f_size))

    l_t2_parameters.append((d_input_name + "_addr",   "int",      input_s_size + " * n_blks"))
    l_t2_parameters.append((d_input_name + "_offset", "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

    #
    if opt_data_type == "DOUBLE":
        l_device_dynamic.append(("double",  d_input_name,             h_input_name,               input_f_size))
    else:
        l_device_dynamic.append(("float",  d_input_name,             h_input_name,               input_f_size))

    l_device_dynamic.append(("int",     d_input_name + "_addr",   h_input_name + "_addr",     input_s_size + " * n_blks_" + str(kernel_number)))
    l_device_dynamic.append(("int",     d_input_name + "_offset", h_input_name + "_offset",   "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

    #
    #
    #
    if possible_diff == 1:
        tc_gen_code_helper_varible(f, "int*",       d_input_name + "_addr_full")
        tc_gen_code_helper_varible(f, "int*",       h_input_name + "_addr_full")
        tc_gen_code_helper_varible(f, "int*",       d_input_name + "_addr_non_full")
        tc_gen_code_helper_varible(f, "int*",       h_input_name + "_addr_non_full")

        #
        if opt_data_type == "DOUBLE":
            l_t2_parameters_nf.append((d_input_name,                    "double",   input_f_size))
        else:
            l_t2_parameters_nf.append((d_input_name,                    "float",   input_f_size))

        l_t2_parameters_nf.append((d_input_name + "_addr_non_full", "int",      input_s_size + " * num_blk_non_full_" + str(kernel_number)))
        l_t2_parameters_nf.append((d_input_name + "_offset",        "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

        #
        if opt_data_type == "DOUBLE":
            l_t2_parameters_f.append((d_input_name,                     "double",   input_f_size))
        else:
            l_t2_parameters_f.append((d_input_name,                     "float",   input_f_size))
        l_t2_parameters_f.append((d_input_name + "_addr_full",      "int",      input_s_size + " * num_blk_full_" + str(kernel_number)))
        l_t2_parameters_f.append((d_input_name + "_offset",         "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

        l_device_dynamic.append(("int", d_input_name + "_addr_full", h_input_name + "_addr_full", input_s_size + " * num_blk_full_" + str(kernel_number)))
        l_device_dynamic.append(("int", d_input_name + "_addr_non_full", h_input_name + "_addr_non_full", input_s_size + " * num_blk_non_full_" + str(kernel_number)))
        l_cuda_malloc.append((d_input_name + "_addr_full",      "int",  input_s_size + " * num_blk_full_" + str(kernel_number)))
        l_cuda_malloc.append((d_input_name + "_addr_non_full",  "int",  input_s_size + " * num_blk_non_full_" + str(kernel_number)))

    f.write("\n")

#
def tc_gen_variables_input_right(kernel_number, each_input, l_external_idx, l_v2_d_decl_var, l_v2_parameters, l_cuda_malloc, l_device_dynamic, l_var_input_right, opt_data_type):
    # Right Input
    d_input_name    =  "dev_" + each_input[1][0]
    h_input_name    = "host_" + each_input[1][0]
    input_f_size    = ""
    input_s_size    = ""

    #

    idx_f_count = 0
    idx_s_count = 0
    for each_index in each_input[1][1]:
        if tc_helper.tc_gen_helper_find_1d(l_external_idx, each_index) != -1:
            if idx_f_count == 0:
                input_f_size = "size_"      + each_index
            else:
                input_f_size = "size_"      + each_index + " * " + input_f_size

            if idx_s_count == 0:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize()
            else:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize() + " * " + input_s_size
            idx_f_count = idx_f_count + 1
            idx_s_count = idx_s_count + 1
        else:
            if idx_f_count == 0:
                input_f_size = "size_"      + each_index
            else:
                input_f_size = "size_"      + each_index + " * " + input_f_size
            idx_f_count = idx_f_count + 1
    
    #
    if opt_data_type == "DOUBLE":
        l_var_input_right.append(["double*", d_input_name])
    else:
        l_var_input_right.append(["float*", d_input_name])

    l_var_input_right.append(["int*",    d_input_name + "_addr"])
    l_var_input_right.append(["int*",    h_input_name + "_addr"])
    l_var_input_right.append(["int*",    d_input_name + "_offset"])
    l_var_input_right.append(["int*",    h_input_name + "_offset"])

    #
    if opt_data_type == "DOUBLE":
        l_v2_d_decl_var.append("double* " + d_input_name)
    else:
        l_v2_d_decl_var.append("float* " + d_input_name)

    l_v2_d_decl_var.append("const int* __restrict__ " + d_input_name + "_addr")
    l_v2_d_decl_var.append("const int* __restrict__ " + d_input_name + "_offset")
    
    #
    if opt_data_type == "DOUBLE":
        l_cuda_malloc.append([d_input_name,             "double",   input_f_size])
    else:
        l_cuda_malloc.append([d_input_name,             "float",   input_f_size])

    l_cuda_malloc.append([d_input_name + "_addr",   "int",      input_s_size + " * num_thread_blocks_kernel_" + str(kernel_number)])
    l_cuda_malloc.append([d_input_name + "_offset", "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"])

    #
    if opt_data_type == "DOUBLE":
        l_v2_parameters.append([d_input_name,                   "double",   input_f_size])
    else:
        l_v2_parameters.append([d_input_name,                   "float",   input_f_size])

    l_v2_parameters.append([d_input_name + "_addr",         "int",      input_s_size + " * num_thread_blocks_kernel_" + str(kernel_number)])
    l_v2_parameters.append([d_input_name + "_offset",       "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"])

    #
    if opt_data_type == "DOUBLE":
        l_device_dynamic.append(["double",  d_input_name,                h_input_name,               input_f_size])
    else:
        l_device_dynamic.append(["float",  d_input_name,                h_input_name,               input_f_size])

    l_device_dynamic.append(["int",     d_input_name + "_addr",      h_input_name + "_addr",     input_s_size + " * num_thread_blocks_kernel_" + str(kernel_number)])
    l_device_dynamic.append(["int",     d_input_name + "_offset",    h_input_name + "_offset",   "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"])

#
def tc_gen_global_variables_outputs_input_right(f,  each_input,         l_external_idx,     possible_diff,
                                                    l_v2_d_decl_var,    l_v2_parameters,    l_v2_parameters_nf, l_v2_parameters_f,
                                                    l_cuda_malloc,      l_device_dynamic,   kernel_number,
                                                    opt_data_type):
    f.write("// Global Variables for Right Input\n")
    # Right Input
    d_input_name    = "d_" + each_input[1][0]
    h_input_name    = "h_" + each_input[1][0]
    input_f_size    = ""
    input_s_size    = ""

    #
    idx_f_count = 0
    idx_s_count = 0
    for each_index in each_input[1][1]:
        if tc_helper.tc_gen_helper_find_1d(l_external_idx, each_index) != -1:
            if idx_f_count == 0:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize()
            else:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize() + " * " + input_f_size

            if idx_s_count == 0:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize()
            else:
                input_s_size = "SIZE_SLICE_" + str(kernel_number) + "_"    + each_index.capitalize() + " * " + input_s_size
            idx_f_count = idx_f_count + 1
            idx_s_count = idx_s_count + 1
        else:
            if idx_f_count == 0:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize()
            else:
                input_f_size = "SIZE_IDX_"      + each_index.capitalize() + " * " + input_f_size
            idx_f_count = idx_f_count + 1

    #
    if opt_data_type == "DOUBLE":
        tc_gen_code_helper_varible(f, "double*",    d_input_name)
        tc_gen_code_helper_varible(f, "double*",    h_input_name)
    else:
        tc_gen_code_helper_varible(f, "float*",    d_input_name)
        tc_gen_code_helper_varible(f, "float*",    h_input_name)

    tc_gen_code_helper_varible(f, "int*",       d_input_name + "_addr")
    tc_gen_code_helper_varible(f, "int*",       h_input_name + "_addr")

    tc_gen_code_helper_varible(f, "int*",       d_input_name + "_offset")
    tc_gen_code_helper_varible(f, "int*",       h_input_name + "_offset")

    #
    if opt_data_type == "DOUBLE":
        l_v2_d_decl_var.append("double* "   + d_input_name)
    else:
        l_v2_d_decl_var.append("float* "   + d_input_name)
    l_v2_d_decl_var.append("const int* __restrict__ "      + d_input_name + "_addr")
    l_v2_d_decl_var.append("const int* __restrict__ "      + d_input_name + "_offset")

    #
    if opt_data_type == "DOUBLE":
        l_v2_parameters.append((d_input_name, "double", input_f_size))
    else:
        l_v2_parameters.append((d_input_name, "float", input_f_size))

    l_v2_parameters.append((d_input_name + "_addr",   "int",      input_s_size + " * n_blks_" + str(kernel_number)))
    l_v2_parameters.append((d_input_name + "_offset", "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

    #
    if opt_data_type == "DOUBLE":
        l_cuda_malloc.append((d_input_name,             "double",   input_f_size))
    else:
        l_cuda_malloc.append((d_input_name,             "float",   input_f_size))

    l_cuda_malloc.append((d_input_name + "_addr",   "int",      input_s_size + " * n_blks_" + str(kernel_number)))
    l_cuda_malloc.append((d_input_name + "_offset", "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

    #
    if opt_data_type == "DOUBLE":
        l_device_dynamic.append(("double",  d_input_name,             h_input_name,               input_f_size))
    else:
        l_device_dynamic.append(("float",  d_input_name,             h_input_name,               input_f_size))

    l_device_dynamic.append(("int",     d_input_name + "_addr",   h_input_name + "_addr",     input_s_size + " * n_blks_" + str(kernel_number)))
    l_device_dynamic.append(("int",     d_input_name + "_offset", h_input_name + "_offset",   "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

    #
    #
    #
    if possible_diff == 1:
        tc_gen_code_helper_varible(f, "int*",       d_input_name + "_addr_full")
        tc_gen_code_helper_varible(f, "int*",       h_input_name + "_addr_full")
        tc_gen_code_helper_varible(f, "int*",       d_input_name + "_addr_non_full")
        tc_gen_code_helper_varible(f, "int*",       h_input_name + "_addr_non_full")

        #
        if opt_data_type == "DOUBLE":
            l_v2_parameters_nf.append((d_input_name,                    "double",   input_f_size))
        else:
            l_v2_parameters_nf.append((d_input_name,                    "float",   input_f_size))
        l_v2_parameters_nf.append((d_input_name + "_addr_non_full", "int",      input_s_size + " * num_blk_non_full_" + str(kernel_number)))
        l_v2_parameters_nf.append((d_input_name + "_offset",        "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

        #
        if opt_data_type == "DOUBLE":
            l_v2_parameters_f.append((d_input_name,                     "double",   input_f_size))
        else:
            l_v2_parameters_f.append((d_input_name,                     "float",   input_f_size))
        l_v2_parameters_f.append((d_input_name + "_addr_full",      "int",      input_s_size + " * num_blk_full"))
        l_v2_parameters_f.append((d_input_name + "_offset",         "int",      "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

        l_device_dynamic.append(("int", d_input_name + "_addr_full",        h_input_name + "_addr_full", input_s_size + " * num_blk_full_" + str(kernel_number)))
        l_device_dynamic.append(("int", d_input_name + "_addr_non_full",    h_input_name + "_addr_non_full", input_s_size + " * num_blk_non_full_" + str(kernel_number)))
        l_cuda_malloc.append((d_input_name + "_addr_full",      "int",  input_s_size + " * num_blk_full_" + str(kernel_number)))
        l_cuda_malloc.append((d_input_name + "_addr_non_full",  "int",  input_s_size + " * num_blk_non_full_" + str(kernel_number)))

    f.write("\n")

#
def tc_gen_variables_outputs_helpers(kernel_number, l_t3_d_decl_var, l_t3_parameters, l_cuda_malloc, l_device_dynamic, l_var_outputs_helpers):
    #
    #
    #

    #   1. Block Index
    #l_var_outputs_helpers.append(["int*",  "dev_t3_block_index_" + str(kernel_number)])
    l_var_outputs_helpers.append(["int*", "host_t3_block_index_" + str(kernel_number)])

    #   2. Block Range
    str_name = "t3_block_range_" + str(kernel_number)
    str_size = "num_thread_blocks_kernel_" + str(kernel_number) + " * NUM_INDEX"
    l_var_outputs_helpers.append(["int*",  "dev_" + str_name])
    l_var_outputs_helpers.append(["int*", "host_" + str_name])
    
    l_t3_d_decl_var.append( "const int* __restrict__ dev_" + str_name)
    l_t3_parameters.append( ["dev_" + str_name, "int", str_size])
    l_cuda_malloc.append(   ["dev_" + str_name, "int", str_size])
    l_device_dynamic.append(["int", "dev_" + str_name, "host_" + str_name, str_size])

    #   3. Output-Base
    str_name = "t3_output_base_" + str(kernel_number)
    str_size = "num_thread_blocks_kernel_" + str(kernel_number)
    l_var_outputs_helpers.append(["int*",  "dev_" + str_name])
    l_var_outputs_helpers.append(["int*", "host_" + str_name])

    l_t3_d_decl_var.append( "const int* __restrict__ dev_" + str_name)
    l_t3_parameters.append( ["dev_" + str_name, "int", str_size])
    l_cuda_malloc.append(   ["dev_" + str_name, "int", str_size])
    l_device_dynamic.append(["int", "dev_" + str_name, "host_" + str_name, str_size])

    #   4. Output-Offset
    str_name = "t3_output_offset_" + str(kernel_number)
    str_size = "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y" 
    l_var_outputs_helpers.append(["int*",  "dev_t3_output_offset_" + str(kernel_number)])
    l_var_outputs_helpers.append(["int*", "host_t3_output_offset_" + str(kernel_number)])

    l_t3_d_decl_var.append( "const int* __restrict__ dev_" + str_name)
    l_t3_parameters.append( ["dev_" + str_name, "int", str_size])
    l_cuda_malloc.append(   ["dev_" + str_name, "int", str_size])
    l_device_dynamic.append(["int", "dev_" + str_name, "host_" + str_name, str_size])

#
def tc_gen_global_variables_outputs_helpers(f,  possible_diff,      kernel_number,                          # Input
                                                l_t3_parameters,    l_t3_parameters_nf, l_t3_parameters_f,  # Outputs
                                                l_cuda_malloc,      l_device_dynamic,   l_t3_d_decl_var):   # Outputs
    #
    #   Depends on # of Fused Kernel.
    #
    if possible_diff == 1:
        tc_gen_code_helper_varible(f, "int*", "t3_blk_idx_full_"        + str(kernel_number))
        tc_gen_code_helper_varible(f, "int*", "t3_blk_idx_non_full_"    + str(kernel_number))
        f.write("\n")

    tc_gen_code_helper_varible(f, "int*",       "d_t3_blk_rng_" + str(kernel_number))
    tc_gen_code_helper_varible(f, "int*",       "h_t3_blk_idx_" + str(kernel_number))     # only for host
    tc_gen_code_helper_varible(f, "int*",       "h_t3_blk_rng_" + str(kernel_number))

    if possible_diff == 1:
        tc_gen_code_helper_varible(f, "int*",       "d_t3_blk_rng_nf_" + str(kernel_number))
        tc_gen_code_helper_varible(f, "int*",       "h_t3_blk_rng_nf_" + str(kernel_number))
        l_t3_parameters_nf.append(( "d_t3_blk_rng_nf_"  + str(kernel_number), "int", "num_blk_non_full_"    + str(kernel_number) + " * NUM_INDEX"))
        l_t3_parameters_f.append((  "d_t3_blk_rng_"     + str(kernel_number), "int", "n_blks_"              + str(kernel_number) + " * NUM_INDEX"))
        l_cuda_malloc.append((      "d_t3_blk_rng_nf_" + str(kernel_number),  "int", "num_blk_non_full_"    + str(kernel_number) + " * NUM_INDEX"))
        l_device_dynamic.append(("int", "d_t3_blk_rng_nf_" + str(kernel_number), "h_t3_blk_rng_nf_" + str(kernel_number), "num_blk_non_full_" + str(kernel_number) + " * NUM_INDEX"))

    l_t3_d_decl_var.append("const int* __restrict__ t3_blk_rng_" + str(kernel_number))
    l_t3_parameters.append( ("d_t3_blk_rng_" + str(kernel_number), "int", "n_blks_" + str(kernel_number) + " * NUM_INDEX"))
    l_cuda_malloc.append(   ("d_t3_blk_rng_" + str(kernel_number), "int", "n_blks_" + str(kernel_number) + " * NUM_INDEX"))
    l_device_dynamic.append(("int", "d_t3_blk_rng_" + str(kernel_number), "h_t3_blk_rng_" + str(kernel_number), "n_blks_" + str(kernel_number) + " * NUM_INDEX"))
    f.write("\n")

    tc_gen_code_helper_varible(f, "int*",       "d_t3_output_base_" + str(kernel_number))
    tc_gen_code_helper_varible(f, "int*",       "h_t3_output_base_" + str(kernel_number))
    l_t3_d_decl_var.append("const int* __restrict__ t3_output_base_" + str(kernel_number))
    l_t3_parameters.append( ("d_t3_output_base_" + str(kernel_number), "int", "n_blks_" + str(kernel_number)))
    l_cuda_malloc.append(   ("d_t3_output_base_" + str(kernel_number), "int", "n_blks_" + str(kernel_number)))
    l_device_dynamic.append(("int", "d_t3_output_base_" + str(kernel_number), "h_t3_output_base_" + str(kernel_number), "n_blks_" + str(kernel_number)))

    if possible_diff == 1:
        tc_gen_code_helper_varible(f, "int*",       "d_t3_output_base_full_"        + str(kernel_number))
        tc_gen_code_helper_varible(f, "int*",       "d_t3_output_base_non_full_"    + str(kernel_number))
        tc_gen_code_helper_varible(f, "int*",       "h_t3_output_base_full_"        + str(kernel_number))
        tc_gen_code_helper_varible(f, "int*",       "h_t3_output_base_non_full_"    + str(kernel_number))

        l_t3_parameters_f.append(   ("d_t3_output_base_full_"       + str(kernel_number), "int", "num_blk_full_"      + str(kernel_number)))
        l_t3_parameters_nf.append(  ("d_t3_output_base_non_full_"   + str(kernel_number), "int", "num_blk_non_full_"  + str(kernel_number)))

        l_device_dynamic.append(("int", "d_t3_output_base_full_"        + str(kernel_number), "h_t3_output_base_full_"      + str(kernel_number), "num_blk_full_"       + str(kernel_number)))
        l_device_dynamic.append(("int", "d_t3_output_base_non_full_"    + str(kernel_number), "h_t3_output_base_non_full_"  + str(kernel_number), "num_blk_non_full_"   + str(kernel_number)))
        l_cuda_malloc.append(("d_t3_output_base_full_"      + str(kernel_number), "int", "num_blk_full_"        + str(kernel_number)))
        l_cuda_malloc.append(("d_t3_output_base_non_full_"  + str(kernel_number), "int", "num_blk_non_full_"    + str(kernel_number)))

    f.write("\n")
    tc_gen_code_helper_varible(f, "int*",       "d_t3_output_offset_" + str(kernel_number))
    tc_gen_code_helper_varible(f, "int*",       "h_t3_output_offset_" + str(kernel_number))
    l_t3_d_decl_var.append("const int* __restrict__ t3_output_offset_" + str(kernel_number))
    l_cuda_malloc.append(   ("d_t3_output_offset_" + str(kernel_number), "int", "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))
    l_t3_parameters.append( ("d_t3_output_offset_" + str(kernel_number), "int", "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))
    l_device_dynamic.append(("int", "d_t3_output_offset_" + str(kernel_number), "h_t3_output_offset_" + str(kernel_number), "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))
    f.write("\n")                                                       # this should be stored in a table to be used in future.

    if possible_diff == 1:
        l_t3_parameters_nf.append(  ("d_t3_output_offset_" + str(kernel_number), "int", "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))
        l_t3_parameters_f.append(   ("d_t3_output_offset_" + str(kernel_number), "int", "SIZE_TB_" + str(kernel_number) + "_X * SIZE_TB_" + str(kernel_number) + "_Y"))

#
def tc_gen_global_variables_outputs(f,  possible_diff,      kernel_number,                          # Input
                                        l_t3_d_decl_var,                                            # output
                                        l_t3_parameters,    l_t3_parameters_nf, l_t3_parameters_f,  # output
                                        l_cuda_malloc,      l_device_dynamic,                       # output 
                                        opt_data_type):
    #
    #   To-Do: Should Support Multiple Outputs
    #   Depends on # of Fused Kernel.
    #
    if kernel_number == 1:
        # To-Do (Oct. 12)
        #if possible_diff == 1:
        #    tc_gen_code_helper_varible(f, "int*", "t3_blk_idx_full_" + str(kernel_number))
        #    tc_gen_code_helper_varible(f, "int*", "t3_blk_idx_non_full_" + str(kernel_number))

        #. Common for a single inner-group
        f.write("\n")
        f.write("// Depends on # of Fused Kernels\n")

        if opt_data_type == "DOUBLE":
            tc_gen_code_helper_varible(f, "double*",    "d_t3")         # this will be used for pre_SD2_Functions().
            tc_gen_code_helper_varible(f, "double*",    "h_t3")         # this will be used for pre_SD2_Functions().
            tc_gen_code_helper_varible(f, "double*",    "h_t3_chk")     # this will be used for pre_SD2_Functions().
        else:
            tc_gen_code_helper_varible(f, "float*",    "d_t3")         # this will be used for pre_SD2_Functions().
            tc_gen_code_helper_varible(f, "float*",    "h_t3")         # this will be used for pre_SD2_Functions().
            tc_gen_code_helper_varible(f, "float*",    "h_t3_chk")     # this will be used for pre_SD2_Functions().

        #
        if opt_data_type == "DOUBLE":
            l_cuda_malloc.append(("d_t3", "double", "size_T3"))
            l_device_dynamic.append(("double", "d_t3", "h_t3", "size_T3"))
        else:
            l_cuda_malloc.append(("d_t3", "float", "size_T3"))
            l_device_dynamic.append(("float", "d_t3", "h_t3", "size_T3"))

    #
    if possible_diff == 1:
        if opt_data_type == "DOUBLE":
            l_t3_parameters_nf.append(("d_t3", "double", "size_T3"))
            l_t3_parameters_f.append(("d_t3", "double", "size_T3"))
        else:
            l_t3_parameters_nf.append(("d_t3", "float", "size_T3"))
            l_t3_parameters_f.append(("d_t3", "float", "size_T3"))
    else:
        if opt_data_type == "DOUBLE":
            l_t3_parameters.append(("d_t3", "double", "size_T3"))
        else:
            l_t3_parameters.append(("d_t3", "float", "size_T3"))

    #
    if opt_data_type == "DOUBLE":
        l_t3_d_decl_var.append("double* t3")
    else:
        l_t3_d_decl_var.append("float* t3")
    f.write("\n")

#
def tc_gen_variables_outputs(kernel_number, l_interface_info,   l_external_idx, 
                                            l_t3_d_decl_var,    l_t3_parameters, l_cuda_malloc, l_device_dynamic, 
                                            l_var_outputs,      opt_data_type):
    #
    #   Because the Output is COMMON among Inner Groups.
    #
    idx_count   = 0
    str_t3_size = ""
    for each_idx in l_external_idx:
        if idx_count == 0:            
            str_t3_size = "size_" + each_idx
        else:
            str_t3_size = str_t3_size + " * size_" + each_idx
        idx_count = idx_count + 1

    #
    if kernel_number == 1:
        if opt_data_type == "DOUBLE":
            l_var_outputs.append(["double*", "dev_t3"])
        else:
            l_var_outputs.append(["float*", "dev_t3"])

        #
        if opt_data_type == "DOUBLE":
            l_cuda_malloc.append(["dev_t3", "double", str_t3_size])
            l_device_dynamic.append(["double", "dev_t3", l_interface_info[0][1], str_t3_size])
        else:
            l_cuda_malloc.append(["dev_t3", "float", str_t3_size])
            l_device_dynamic.append(["float", "dev_t3", l_interface_info[0][1], str_t3_size])

    #
    if opt_data_type == "DOUBLE":
        l_t3_parameters.append(["dev_t3", "double", str_t3_size])
        l_t3_d_decl_var.append("double* dev_t3")
    else:
        l_t3_parameters.append(["dev_t3", "float", str_t3_size])
        l_t3_d_decl_var.append("float* dev_t3")

    #
    #   End of def.
    #

#
def tc_gen_global_variables_sizes(f, l_input_tensors, possible_diff, kernel_number):
    #
    #   To-Do: Inner-Groups
    #
    if possible_diff == 1:
        tc_gen_code_helper_varible(f, "int", "n_blks_"              + str(kernel_number))
        tc_gen_code_helper_varible(f, "int", "num_blk_full_"        + str(kernel_number))
        tc_gen_code_helper_varible(f, "int", "num_blk_non_full_"    + str(kernel_number))
    else:
        tc_gen_code_helper_varible(f, "int", "n_blks_" + str(kernel_number))


    #
    #   To-Do: Inner-Groupse
    #   sizes for two tensor inputs
    #
    f.write("// Each Input Tensor Size\n")
    for each_tc in l_input_tensors:
        tc_gen_code_helper_varible(f, "int", "size_" + each_tc[0][0].capitalize())
        tc_gen_code_helper_varible(f, "int", "size_" + each_tc[1][0].capitalize())
    f.write("\n")

#
def tc_gen_code_helper_varible(f, type, name):
    f.write(type)
    f.write(" ")
    f.write(name)
    f.write(";\n")
