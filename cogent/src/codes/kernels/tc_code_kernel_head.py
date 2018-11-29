#
#
#
def tc_gen_code_Kernel_Head(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, l_input_strides, l_external_idx, l_internal_idx, l_inputs_addr,
                            opt_internal, opt_pre_computed, opt_data_type):
    #
    f.write("\n")
    f.write("// created by tc_gen_code_Kernel()\n")
    f.write("__global__ void ")
    f.write(kernel_name)
    f.write("(")

    # parameters for t3 (output)
    for t3_var in l_t3_d_decl_var:
        if opt_pre_computed == -1:
            if "range"  in t3_var:
                continue
            if "base"   in t3_var:
                continue
            if "offset" in t3_var:
                continue

            #
            f.write(t3_var)
            f.write(", ")
        else:
            f.write(t3_var)
            f.write(", ")
    f.write("\n")

    # parameters for t2 (left)
    for t2_var in l_t2_d_decl_var:
        if opt_pre_computed == -1:
            if "addr"   in t2_var:
                continue
            if "offset" in t2_var:
                continue
            #
            f.write(t2_var)
            f.write(", ")
        else:
            f.write(t2_var)
            f.write(", ")
    f.write("\n")

    # parameters for v2 (right)
    for v2_var in l_v2_d_decl_var:
        if opt_pre_computed == -1:
            if "addr"   in v2_var:
                continue
            if "offset" in v2_var:
                continue
            #
            f.write(v2_var)
            f.write(", ")
        else:
            f.write(v2_var)
            f.write(", ")
    f.write("\n")   

    #
    #   [Rules] Sizes:      External Indinces and Iternal Indices
    #           numBlks:    Externl Indices
    #
    if opt_pre_computed == -1:
        #
        #   [Sizes] External
        #
        for each_idx in l_external_idx:
            f.write("int size_" + each_idx + ", ")

        #
        #   [Sizes] Internal
        #
        for each_idx in l_internal_idx:
            f.write("int size_" + each_idx + ", ")
        f.write("\n")

        #
        #   [Blks] External
        # 
        for each_idx in l_external_idx:
            f.write("int numBlk_" + each_idx + ", ")
        f.write("\n")

    #
    #   (Optional) Strides for LEFT and RIGHT (if internal index is not FVI)
    #   This is for an internal index, because constant-memory will be used for
    #   multiple-internal index.
    #
    if len(l_input_strides) > 0:
        for each_tc in l_input_strides:
            f.write("int " + each_tc[0] + ", ")
            f.write("int " + each_tc[2] + ", ")
        f.write("\n")

    #
    #   (Optional)
    #
    if opt_internal > 1:
        for each_tensor_contraction in l_inputs_addr:
            f.write("int* dev_internal_offset_" + each_tensor_contraction[0][3] + ", ")
            f.write("int* dev_internal_offset_" + each_tensor_contraction[1][3] + ", ")
            f.write("\n")

    #   
    f.write("int stride_reg_x, ")
    f.write("int stride_reg_y, ")
    f.write("\n")

    # the size of |internal indices| = size_internal
    f.write("int size_internal")
    f.write(")\n")

#
#
#
def tc_gen_code_Kernel_Head_RT(f, kernel_name, l_combined_t3_d_decl_var, l_combined_t2_d_decl_var, l_combined_v2_d_decl_var):
    #
    f.write("\n")
    f.write("// created by tc_gen_code_kernel_RT()\n")
    f.write("__global__ void ")
    f.write(kernel_name)
    f.write("(")

    #   T3 (Output)
    #
    idx_count = 0
    for each_inner_group in l_combined_t3_d_decl_var:
        for t3_var in each_inner_group:
            if idx_count != 0:
                idx_count = idx_count - 1   # Need to Get Rid of the Output (Overlapped)
                continue
            f.write(t3_var)
            f.write(", ")
        f.write("\n")
        idx_count = idx_count + 1

    #   T2 (LEFT)
    for each_inner_group in l_combined_t2_d_decl_var:
        for t2_var in each_inner_group:
            f.write(t2_var)
            f.write(", ")
        f.write("\n")

    #   V2 (RIGHT)
    for each_inner_group in l_combined_v2_d_decl_var:
        for v2_var in each_inner_group:
            f.write(v2_var)
            f.write(", ")
        f.write("\n")

    #
    f.write("int stride_reg_x, ")
    f.write("int stride_reg_y, ")

    #
    f.write("int size_internal")
    f.write(")\n")

