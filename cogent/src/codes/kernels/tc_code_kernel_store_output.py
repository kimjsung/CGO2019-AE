#
#   [To-Do] 1. Loop-Unrolling
#           2. Boundary Checks
#           3. Instuction Reordering
#
def tc_code_kernel_Store_Results(f, opt_gen_full, l_t3_mapping_tb_2D, l_t3_mapping_reg, size_reg_x, size_reg_y, idx_kernel, opt_accumulated):
    #
    #
    #
    f.write("\n")
    f.write("\t// Store Results (Registers) to Global Memory\n");
    f.write("\t// Part: Generalized Threads\n")
    f.write("\t// Part: Generalized Register-Tiling\n")

    #
    #   Option #1: None-Full-Tile (Partial-Tile)
    #
    if opt_gen_full == 1:
        #
        f.write("\tif (")
        #
        axis_count = 0
        for axis_idx in l_t3_mapping_tb_2D:
            #
            if axis_count != 0:
                f.write(" && ")

            # Per Each-Axis
            idx_count = 0
            for each_idx in axis_idx:
                if idx_count == 0:
                    f.write("idx_" + each_idx + " < rng_" + each_idx)
                else:
                    f.write(" && idx_" + each_idx + " < rng_" + each_idx)
                #
                idx_count += 1

            axis_count += 1
        #
        f.write(")\n")
    #
    #   Option #1: Full-Tile
    #
    else:
        f.write("\t#pragma unroll " + str(size_reg_y) + "\n")

    #
    #
    #
    f.write("\tfor (int i = 0; i < ")
    f.write(str(size_reg_y))
    f.write("; i++)\n")
    f.write("\t{\n")

    f.write("\t\tfor (int j = 0; j < ")
    f.write(str(size_reg_x))
    f.write("; j++)\n")
    f.write("\t\t{\n")

    #
    #   if
    #
    if opt_gen_full == 1:
        f.write("\t\t\tif(i < rng_" + l_t3_mapping_reg[1] + " && j < rng_" + l_t3_mapping_reg[0] + ")\n")
        f.write("\t\t\t{\n")

    #. Output:
    if idx_kernel == 1:
        if opt_accumulated == 1:
            #
            #   [To-Do] Instruction Reordering
            #
            f.write("\t\t\t\tdev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];\n")
            #f.write("\t\t\t\tint t3_addr = t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x);\n")
            #f.write("\t\t\t\tdouble tmp_dev_t3 = dev_t3[t3_addr];\n")
            #f.write("\t\t\t\tdev_t3[t3_addr] = tmp_dev_t3 + reg_tile[i][j];\n")
        else:
            f.write("\t\t\tdev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];\n")
    else:
        f.write("\t\t\tdev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] += reg_tile[i][j];\n")

    #
    #   if-end
    #
    if opt_gen_full == 1:
        f.write("\t\t\t}\n")

    f.write("\t\t}\n")
    f.write("\t}\n")
