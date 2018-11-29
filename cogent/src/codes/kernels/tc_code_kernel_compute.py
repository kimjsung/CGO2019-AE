#
#
#
def tc_gen_code_Kernel_Compute(f,   size_tb_x,  size_tb_y, size_reg_x, size_reg_y, int_str_t2, int_str_v2, 
                                    tensor_contraction, l_t3_mapping_tb_2D,
                                    opt_pre_computed):
    # Load Input Tensors from Shared Memory to registers
    # Depends on SIZE_REG_T
    # idx_offset = 0
    # for tile_idx in size_reg_y:
    l_idx_tb = list()
    for each_axis in l_t3_mapping_tb_2D:
        for each_idx in each_axis:
            l_idx_tb.append(each_idx)

    # sd2_1':  t3[h3,h2,h1,p6,p5,p4] -= t2[p4,p7,h1,h2] *   # p4 (y)
    #                                   v2[p6,p7,h3,p5];    # p5 (x)
    #
    #   should support non-full tiles
    #
    if size_reg_y >= size_reg_x:                # if y = max(x, y)
        #
        #   Input-LEFT
        #
        idx_count           = 0
        str_left_offset     = ""
        #
        #   [To-Do] This should be generalized
        #
        rev_l_idx_left      = list(reversed(tensor_contraction[0][4]))
        for each_idx in rev_l_idx_left:
            #print ("each_idx: ", each_idx)
            #
            for mapped_idx in l_idx_tb:
                if each_idx == mapped_idx:
                    if idx_count == 0:
                        str_left_offset = "idx_" + each_idx
                    else:
                        str_left_offset = "idx_" + each_idx + " + (" + str_left_offset + ") * SIZE_SLICE_1_" + each_idx.capitalize()
                    #
                    idx_count = idx_count + 1
            
        #
        #   Input-RIGHT
        #
        idx_count           = 0
        str_right_offset    = ""
        rev_l_idx_right     = list(reversed(tensor_contraction[1][4]))
        #rev_l_idx_right     = tensor_contraction[1][4]
        for each_idx in rev_l_idx_right:
            #
            for mapped_idx in l_idx_tb:
                if each_idx == mapped_idx:
                    if idx_count == 0:
                        str_right_offset = "idx_" + each_idx
                    else:
                        str_right_offset = "idx_" + each_idx + " + (" + str_right_offset + ") * SIZE_SLICE_1_" + each_idx.capitalize()
                    #
                    idx_count = idx_count + 1

        #
        #
        #
        if tensor_contraction[0][2] == "y":     # "left" is mapped on "y"
            for idx_left in range(0, size_reg_y):
                if opt_pre_computed == -1:
                    f.write("\t\t\ttemp_bv[" + str(idx_left)  + "] = sm_a[ll][" +  str_left_offset + " + " + str(int_str_t2 * idx_left) + "];\n")
                else:
                    f.write("\t\t\ttemp_bv[" + str(idx_left)  + "] = sm_a[ll][dev_" + tensor_contraction[0][3] + "_offset[l_idx_t3] + " + str(int_str_t2 * idx_left) + "];\n")
        else:                                   # "right" is mapped on "y"
            for idx_right in range(0, size_reg_y):
                if opt_pre_computed == -1:
                    f.write("\t\t\ttemp_bv[" + str(idx_right) + "] = sm_b[ll][" + str_right_offset + " + " + str(int_str_v2 * idx_right) + "];\n")
                else:
                    f.write("\t\t\ttemp_bv[" + str(idx_right) + "] = sm_b[ll][dev_" + tensor_contraction[1][3] + "_offset[l_idx_t3] + " + str(int_str_v2 * idx_right) + "];\n")
        f.write("\n")

        # xx for size_reg_x
        # pragma??
        #
        f.write("\t\t\tfor (int xx = 0; xx < " + str(size_reg_x) + "; xx++) // (1)\n")
        f.write("\t\t\t{\n")
        if tensor_contraction[0][2] == "x": # "left" is mapped on "x"
            if opt_pre_computed == -1:
                f.write("\t\t\t\ttemp_av = sm_a[ll][" + str_left_offset + " + (xx * " + str(int_str_t2) + ")];\n")
            else:
                f.write("\t\t\t\ttemp_av = sm_a[ll][dev_" + tensor_contraction[0][3] + "_offset[l_idx_t3] + (xx * " + str(int_str_t2) + ")];\n")
        else:                               # "right" is mapped on "x"
            if opt_pre_computed == -1:
                f.write("\t\t\t\ttemp_av = sm_b[ll][" + str_right_offset + " + (xx * " + str(int_str_v2) + ")];\n")
            else:
                f.write("\t\t\t\ttemp_av = sm_b[ll][dev_" + tensor_contraction[1][3] + "_offset[l_idx_t3] + (xx * " + str(int_str_v2) + ")];\n")
        f.write("\n")

        #
        #   [3] Compute
        #
        for yy in range(0, size_reg_y):
            f.write("\t\t\t\treg_tile[" + str(yy) + "][xx] " + tensor_contraction[2] + " temp_av * temp_bv[" + str(yy) + "];\n")
        f.write("\t\t\t}\n")
    else:                                       # if x = max(x, y)
        #
        #   [2] |REG_X| > |REG_Y|
        #
        #
        #   Input-LEFT
        #
        idx_count           = 0
        str_left_offset     = ""
        #
        #   [To-Do] This should be generalized
        #
        rev_l_idx_left      = tensor_contraction[0][4]
        for each_idx in rev_l_idx_left:
            #print ("each_idx: ", each_idx)
            #
            for mapped_idx in l_idx_tb:
                if each_idx == mapped_idx:
                    if idx_count == 0:
                        str_left_offset = "idx_" + each_idx
                    else:
                        str_left_offset = "idx_" + each_idx + " + (" + str_left_offset + ") * SIZE_SLICE_1_" + each_idx.capitalize()
                    #
                    idx_count = idx_count + 1
            
        #
        #   Input-RIGHT
        #
        idx_count           = 0
        str_right_offset    = ""
        rev_l_idx_right     = list(reversed(tensor_contraction[1][4]))
        #rev_l_idx_right     = tensor_contraction[1][4]
        for each_idx in rev_l_idx_right:
            #
            for mapped_idx in l_idx_tb:
                if each_idx == mapped_idx:
                    if idx_count == 0:
                        str_right_offset = "idx_" + each_idx
                    else:
                        str_right_offset = "idx_" + each_idx + " + (" + str_right_offset + ") * SIZE_SLICE_1_" + each_idx.capitalize()
                    #
                    idx_count = idx_count + 1

        #
        #   "temp_bv" <--- |REG_X|
        #
        if tensor_contraction[0][2] == "x":
            #
            #   LEFT is mapped on "X"
            #
            for idx_right in range(0, size_reg_x):
                if opt_pre_computed == -1:
                    f.write("\t\t\ttemp_bv[" + str(idx_right) + "] = sm_a[ll][" + str_left_offset +  " + " + str(int_str_t2 * idx_right) + "];\n")
                else:
                    f.write("\t\t\ttemp_bv[" + str(idx_right) + "] = sm_a[ll][dev_" + tensor_contraction[0][3] + "_offset[l_idx_t3] + " + str(int_str_t2 * idx_right) + "];\n")
        else:
            #
            #   RIGHT is mapped on "X"
            #
            for idx_right in range(0, size_reg_x):
                if opt_pre_computed == -1:
                    f.write("\t\t\ttemp_bv[" + str(idx_right) + "] = sm_b[ll][" + str_right_offset + " + " + str(int_str_v2 * idx_right) + "];\n")
                else:
                    f.write("\t\t\ttemp_bv[" + str(idx_right) + "] = sm_b[ll][dev_" + tensor_contraction[1][3] + "_offset[l_idx_t3] + " + str(int_str_v2 * idx_right) + "];\n")
        f.write("\n")

        #
        #   
        #
        f.write("\t\t\tfor (int yy = 0; yy < " + str(size_reg_y) + "; yy++) // (2)\n")
        f.write("\t\t\t{\n")

        #
        #   "temp_av" <--- |REG_Y|
        #
        if tensor_contraction[0][2] == "y":
            #
            #   LEFT is mapped on "Y"
            #
            if opt_pre_computed == -1:
                f.write("\t\t\t\ttemp_av = sm_a[ll][" + str_left_offset + " + (yy * " + str(int_str_t2) + ")];\n")    # int_str_t2 was int_str_v2
            else:
                f.write("\t\t\t\ttemp_av = sm_a[ll][dev_" + tensor_contraction[0][3] + "_offset[l_idx_t3] + (yy * " + str(int_str_t2) + ")];\n")    # int_str_t2 was int_str_v2
        else:
            #
            #   RIGHT is mapped on "Y"
            #
            if opt_pre_computed == -1:
                f.write("\t\t\t\ttemp_av = sm_b[ll][" + str_right_offset + " + (yy * " + str(int_str_v2) + ")];\n")
            else:
                f.write("\t\t\t\ttemp_av = sm_b[ll][dev_" + tensor_contraction[1][3] + "_offset[l_idx_t3] + (yy * " + str(int_str_v2) + ")];\n")
        f.write("\n")

        #
        #   [Computations]  |REG_X| is enumerated.
        #
        for xx in range(0, size_reg_x):
            f.write("\t\t\t\treg_tile[yy][" + str(xx) + "] " + tensor_contraction[2] + " temp_av * temp_bv[" + str(xx) + "];\n")
        f.write("\t\t\t}\n")

    #
    #
    #
    del l_idx_tb
