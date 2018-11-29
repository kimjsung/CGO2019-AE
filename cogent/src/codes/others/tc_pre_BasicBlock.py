#
import src.generators.tc_helper     as tc_helper

#
#
#
def tc_gen_code_pre_TileApproach(f, kernel_number, l_interface_info, l_t3_idx, l_t3_slices, l_idx_size, l_internal_idx, l_host_dynamic):
    #
    #   Head
    #
    f.write("\n")
    f.write("// Created by tc_gen_code_pre_TileApproach()\n")
    f.write("void pre_TileApproach_" + str(kernel_number) + "(")

    #   Parameters #1:
    f.write("int*& host_t3_block_index_" + str(kernel_number) + ", ")
    f.write("int*& host_t3_block_range_" + str(kernel_number) + ", ")

    #   Parameters #2:
    f.write("int* num_thread_blocks_kernel_" + str(kernel_number) + ", ")

    #   Parameters #3: 
    idx_count = 0
    for each_idx in l_interface_info[0][0]:
        if idx_count == 0:
            f.write("int size_" + each_idx)
        else:
            f.write(", int size_" + each_idx)
        idx_count = idx_count + 1

    f.write(")\n")

    #   Open
    f.write("{\n")

    #
    #   Step 1: Initialization
    #
    tc_gen_code_pre_BasicBlock_Initial_new(f, l_t3_idx, l_t3_slices, kernel_number)

    #
    #   Step 2: Calculating Ranges
    #
    tc_gen_code_pre_BasicBlock_Ranges(f, l_t3_idx, l_t3_slices)

    #
    #   Step 3: Initializing Variables for t3
    #
    tc_gen_code_pre_BasicBlock_Init_Variables(f, kernel_number, l_t3_idx, l_host_dynamic)

    #
    #   Step 4: Creating Arrays for t3 (index & range)
    #
    tc_gen_code_pre_BasicBlock_Create_Arrays_Idx_Rng(f, l_t3_idx, -1, kernel_number)

    #
    #   Step 5: Overview
    #
    tc_gen_code_pre_BasicBlock_overview_new(f, l_t3_idx, l_idx_size, l_t3_slices, l_host_dynamic, l_t3_idx, l_internal_idx, -1, kernel_number)

    #   Close
    f.write("}\n")

#
#   (To-Do:)
#
def tc_gen_code_pre_BasicBlock(f, l_t3_idx, l_idx_size, l_t3_slices, l_host_dynamic, l_external_idx, l_internal_idx, possible_diff, idx_kernel):
    # Head
    f.write("\n")
    f.write("// created by tc_gen_code_pre_BasicBlock()\n")
    f.write("void pre_BasicBlock_" + str(idx_kernel) + "()\n")

    # Open
    f.write("{\n")

    #
    #   Step 1: Initialization
    #
    tc_gen_code_pre_BasicBlock_Initial(f, l_t3_idx, l_t3_slices, idx_kernel)

    #
    #   Step 2: Calculating Ranges
    #
    tc_gen_code_pre_BasicBlock_Ranges(f, l_t3_idx, l_t3_slices)

    #
    #   Step 3: initializing (Global) Variables for t3
    #
    tc_gen_code_pre_BasicBlock_Init_Global_Variables(f, l_t3_idx, l_host_dynamic, possible_diff, idx_kernel)

    #
    #   Step 4: Creating (Global) Arrays for t3 (index, range)
    #
    tc_gen_code_pre_BasicBlock_Create_Arrays_Idx_Rng(f, l_t3_idx, possible_diff, idx_kernel)

    #
    #   Step 4-1: For Non-full-Tile
    #
    if possible_diff == 1:
        tc_gen_code_pre_BasicBlock_Diff_Kernels(f, l_t3_idx, idx_kernel)

    #
    #   Step 5: Overview
    #
    tc_gen_code_pre_BasicBlock_overview(f, l_t3_idx, l_idx_size, l_t3_slices, l_host_dynamic, l_external_idx, l_internal_idx, possible_diff, idx_kernel)

    # Close
    f.write("}\n")

#
def tc_gen_code_pre_BasicBlock_Create_Arrays_Idx_Rng(f, l_t3_idx, possible_diff, idx_kernel):
    # offset for t3 (output)
    for t3_idx in l_t3_idx:
        f.write("\tfor (int blk_" + t3_idx + " = 0; blk_" + t3_idx + " < n_blk_" + t3_idx + "; blk_" + t3_idx + "++)\n")
    #
    f.write("\t{\n")

    #rev_left = list(reversed(sd2_func[0][1]))
    rev_l_t3_idx = list(reversed(l_t3_idx))
    str_blk_idx_base = ""
    idx_count = 0
    for each_idx in rev_l_t3_idx:
        if idx_count == 0:
            str_blk_idx_base = "blk_" + each_idx
        else:
            str_blk_idx_base = "blk_" + each_idx + " + (" + str_blk_idx_base + ") * n_blk_" + each_idx
        idx_count = idx_count + 1

    # order is important in here!
    # temporally I made blk_idx_base manually.
    # If indices are sorted according to a mapping, we can directly use "l_t3_idx"
    #f.write("\t\tint blk_idx_base = (blk_h3 + (blk_h2 + (blk_h1 + (blk_p6 + (blk_p5 + (blk_p4) * n_blk_p5) * n_blk_p6) * n_blk_h1) * n_blk_h2) * n_blk_h3) * NUM_INDEX;\n")
    f.write("\t\tint blk_idx_base = (" + str_blk_idx_base +  ") * NUM_INDEX;\n")
    f.write("\n")

    idx = 0
    for t3_idx in l_t3_idx:
        f.write("\t\thost_t3_block_index_" + str(idx_kernel) + "[blk_idx_base + " + str(idx) + "] = blk_" + t3_idx + ";\n")
        idx = idx + 1

    f.write("\n")

    idx = 0
    for t3_idx in l_t3_idx:
        f.write("\t\thost_t3_block_range_" + str(idx_kernel) + "[blk_idx_base + " + str(idx) + "] = blk_" + t3_idx + "_range[blk_" + t3_idx + "];\n")
        idx = idx + 1

    #
    #   Differenciates (1) blocks for full tiles and (2) blocks for non-full tiles.
    #
    if possible_diff == 1:
        f.write("\n")
        f.write("\t\tif(")

        idx = 0
        for t3_idx in l_t3_idx:
            if idx == 0:
                f.write("host_t3_block_range_" + str(idx_kernel) + "[blk_idx_base + " + str(idx) + "] != SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize())
            else:
                f.write(" || host_t3_block_range_" + str(idx_kernel) + "[blk_idx_base + " + str(idx) + "] != SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize())
            idx = idx + 1

        f.write(")\n")
        f.write("\t\t{\n")      # if
        f.write("\t\t\tnum_blk_non_full_" + str(idx_kernel) + "++;\n")
        f.write("\t\t}\n")
        f.write("\t\telse\n")   # else
        f.write("\t\t{\n")
        f.write("\t\t\tnum_blk_full_" + str(idx_kernel) + "++;\n")
        f.write("\t\t}\n")

    #
    f.write("\t}\n")
    f.write("\n")

#
def tc_gen_code_pre_BasicBlock_Init_Variables(f, kernel_number, l_t3_idx, l_host_dynamic):
    #
    #
    #
    str_block_name = "*num_thread_blocks_kernel_" + str(kernel_number)
    f.write("\t// # of blocks\n")
    f.write("\t" + str_block_name + " = ")
    t3_count = 0
    for t3_idx in l_t3_idx:
        f.write("n_blk_" + t3_idx)
        if t3_count == len(l_t3_idx) - 1:
            f.write(";\n")
        else:
            f.write(" * ")
        t3_count = t3_count + 1
    f.write("\n")

    # h_t3_blk_rng
    f.write("\thost_t3_block_index_" + str(kernel_number) + " = (int*)malloc(sizeof(int) * " + str_block_name + " * NUM_INDEX);\n")
    f.write("\thost_t3_block_range_" + str(kernel_number) + " = (int*)malloc(sizeof(int) * " + str_block_name + " * NUM_INDEX);\n")
    l_host_dynamic.append("host_t3_block_index_" + str(kernel_number))
    l_host_dynamic.append("host_t3_block_range_" + str(kernel_number))
    f.write("\n")

#
def tc_gen_code_pre_BasicBlock_Init_Global_Variables(f, l_t3_idx, l_host_dynamic, possible_diff, idx_kernel):
    #
    #   To-Do: Supports multiple outputs for a series of tensor contractions.
    #
    f.write("\t// # of blocks\n")
    f.write("\tn_blks_" + str(idx_kernel) + " = ")
    t3_count = 0
    for t3_idx in l_t3_idx:
        f.write("n_blk_" + t3_idx)
        if t3_count == len(l_t3_idx) - 1:
            f.write(";\n")
        else:
            f.write(" * ")
        t3_count = t3_count + 1
    f.write("\n")

    # h_t3_blk_rng
    f.write("\th_t3_blk_idx_" + str(idx_kernel) + " = (int*)malloc(sizeof(int) * n_blks_" + str(idx_kernel) + " * NUM_INDEX);\n")
    f.write("\th_t3_blk_rng_" + str(idx_kernel) + " = (int*)malloc(sizeof(int) * n_blks_" + str(idx_kernel) + " * NUM_INDEX);\n")
    l_host_dynamic.append("h_t3_blk_idx_" + str(idx_kernel))
    l_host_dynamic.append("h_t3_blk_rng_" + str(idx_kernel))
    f.write("\n")

    #
    #
    #
    if possible_diff == 1:
        f.write("\tnum_blk_full_" + str(idx_kernel) + "     = 0;\n")
        f.write("\tnum_blk_non_full_" + str(idx_kernel) + " = 0;\n")
        f.write("\n")

#
def tc_gen_code_pre_BasicBlock_Ranges(f, l_t3_idx, l_t3_slices):
    # ranges
    for t3_idx in l_t3_idx:
        val = tc_helper.tc_gen_helper_find(l_t3_slices, t3_idx)
        f.write("\tfor (int i = 0; i < n_blk_" + t3_idx + "; i++)\n")
        f.write("\t{\n")
        f.write("\t\tblk_" + t3_idx + "_range[i] = " + str(val) + ";\n")
        f.write("\t\tif (rng_boundary_" + t3_idx + " != 0 && i == n_blk_" + t3_idx + " - 1)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tblk_" + t3_idx + "_range[i] = rng_boundary_" + t3_idx + ";\n")
        f.write("\t\t}\n")
        f.write("\t}\n")
    f.write("\n")

def tc_gen_code_pre_BasicBlock_Initial_new(f, l_t3_idx, l_t3_slices, idx_kernel):
    # for t3,
    # the number of blocks per index
    for t3_idx in l_t3_idx:
        val = tc_helper.tc_gen_helper_find(l_t3_slices, t3_idx)
        f.write("\tint n_blk_" + t3_idx + " = CEIL(size_" + t3_idx + ", " + str(val) + ");\n")
    f.write("\n")

    # block-range per index
    for t3_idx in l_t3_idx:
        f.write("\tint blk_" + t3_idx + "_range[n_blk_" + t3_idx + "];\n")
    f.write("\n")

    # (just-in-case) # of full-tile per index (for debug)
    #for t3_idx in l_t3_idx:
    #    val = tc_helper.tc_gen_helper_find(l_t3_slices, t3_idx)
    #    f.write("\tint n_blk_full_" + t3_idx + " = size_" + t3_idx + " / " + str(val) + ";\n")
    #f.write("\n")

    #
    for t3_idx in l_t3_idx:
        val = tc_helper.tc_gen_helper_find(l_t3_slices, t3_idx)
        f.write("\tint rng_boundary_" + t3_idx + " = size_" + t3_idx + " % " + str(val) + ";\n")
    f.write("\n")

#
def tc_gen_code_pre_BasicBlock_Initial(f, l_t3_idx, l_t3_slices, idx_kernel):
    # for t3,
    # the number of blocks per index
    for t3_idx in l_t3_idx:
        val = tc_helper.tc_gen_helper_find(l_t3_slices, t3_idx)
        f.write("\tint n_blk_" + t3_idx + " = CEIL(SIZE_IDX_" + t3_idx.capitalize() + ", " + str(val) + ");\n")
    f.write("\n")

    # block-range per index
    for t3_idx in l_t3_idx:
        f.write("\tint blk_" + t3_idx + "_range[n_blk_" + t3_idx + "];\n")
    f.write("\n")

    # (just-in-case) # of full-tile per index (for debug)
    for t3_idx in l_t3_idx:
        val = tc_helper.tc_gen_helper_find(l_t3_slices, t3_idx)
        f.write("\tint n_blk_full_" + t3_idx + " = SIZE_IDX_" + t3_idx.capitalize() + " / " + str(val) + ";\n")
    f.write("\n")

    #
    for t3_idx in l_t3_idx:
        val = tc_helper.tc_gen_helper_find(l_t3_slices, t3_idx)
        f.write("\tint rng_boundary_" + t3_idx + " = SIZE_IDX_" + t3_idx.capitalize() + " % " + str(val) + ";\n")
    f.write("\n")

#
def tc_gen_code_pre_BasicBlock_Diff_Kernels(f, l_t3_idx, idx_kernel):
    #
    f.write("\tt3_blk_idx_full_" + str(idx_kernel) + "      = (int*)malloc(sizeof(int) * num_blk_full_" + str(idx_kernel) + ");\n")
    f.write("\tt3_blk_idx_non_full_" + str(idx_kernel) + "  = (int*)malloc(sizeof(int) * num_blk_non_full_" + str(idx_kernel) + ");\n")
    f.write("\th_t3_blk_rng_nf_" + str(idx_kernel) + "      = (int*)malloc(sizeof(int) * num_blk_non_full_" + str(idx_kernel) + " * NUM_INDEX);\n")
    f.write("\n")

    #
    f.write("\tnum_blk_full_" + str(idx_kernel) + "     = 0;\n")
    f.write("\tnum_blk_non_full_" + str(idx_kernel) + " = 0;\n")
    f.write("\n")

    #
    # offset for t3 (output)
    #
    for t3_idx in l_t3_idx:
        f.write("\tfor (int blk_" + t3_idx + " = 0; blk_" + t3_idx + " < n_blk_" + t3_idx + "; blk_" + t3_idx + "++)\n")
    f.write("\t{\n")

    #
    rev_l_t3_idx = list(reversed(l_t3_idx))
    str_blk_idx_base = ""
    idx_count = 0
    for each_idx in rev_l_t3_idx:
        if idx_count == 0:
            str_blk_idx_base = "blk_" + each_idx
        else:
            str_blk_idx_base = "blk_" + each_idx + " + (" + str_blk_idx_base + ") * n_blk_" + each_idx
        idx_count = idx_count + 1

    #
    f.write("\t\tint blk_idx_base = (" + str_blk_idx_base  + ") * NUM_INDEX;\n")
    f.write("\n")

    #
    f.write("\t\tif(")
    idx = 0
    for t3_idx in l_t3_idx:
        if idx == 0:
            f.write("h_t3_blk_rng_" + str(idx_kernel) + "[blk_idx_base + " + str(idx) + "] != SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize())
        else:
            f.write(" || h_t3_blk_rng_" + str(idx_kernel) + "[blk_idx_base + " + str(idx) + "] != SIZE_SLICE_" + str(idx_kernel) + "_" + t3_idx.capitalize())
        idx = idx + 1

    f.write(")\n")
    f.write("\t\t{\n")

    #
    f.write("\t\t\th_t3_blk_rng_nf_" + str(idx_kernel) + "[num_blk_non_full_" + str(idx_kernel) + " * NUM_INDEX + 0] = blk_h3_range[blk_h3];\n")
    f.write("\t\t\th_t3_blk_rng_nf_" + str(idx_kernel) + "[num_blk_non_full_" + str(idx_kernel) + " * NUM_INDEX + 1] = blk_h2_range[blk_h2];\n")
    f.write("\t\t\th_t3_blk_rng_nf_" + str(idx_kernel) + "[num_blk_non_full_" + str(idx_kernel) + " * NUM_INDEX + 2] = blk_h1_range[blk_h1];\n")
    f.write("\t\t\th_t3_blk_rng_nf_" + str(idx_kernel) + "[num_blk_non_full_" + str(idx_kernel) + " * NUM_INDEX + 3] = blk_p6_range[blk_p6];\n")
    f.write("\t\t\th_t3_blk_rng_nf_" + str(idx_kernel) + "[num_blk_non_full_" + str(idx_kernel) + " * NUM_INDEX + 4] = blk_p5_range[blk_p5];\n")
    f.write("\t\t\th_t3_blk_rng_nf_" + str(idx_kernel) + "[num_blk_non_full_" + str(idx_kernel) + " * NUM_INDEX + 5] = blk_p4_range[blk_p4];\n")
    f.write("\n")

    #
    f.write("\t\t\tt3_blk_idx_non_full_" + str(idx_kernel) + "[num_blk_non_full_" + str(idx_kernel) + "++] = blk_idx_base;\n")

    f.write("\t\t}\n")
    f.write("\t\telse\n")
    f.write("\t\t{\n")

    #
    f.write("\t\t\tt3_blk_idx_full_" + str(idx_kernel) + "[num_blk_full_" + str(idx_kernel) + "++] = blk_idx_base;\n")

    f.write("\t\t}\n")

    #
    f.write("\t}\n")
    f.write("\n")

#
def tc_gen_code_pre_BasicBlock_overview_new(f, l_t3_idx, l_idx_size, l_t3_slices, l_host_dynamic, l_external_idx, l_internal_idx, possible_diff, idx_kernel):
    #
    f.write("\tprintf (\"==========================================================================================================\\n\");\n")
    f.write("\tprintf (\" >>> %s <<<\\n\", __func__);\n")

    #   t3: indices
    f.write("\tprintf (\"               t3 (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write(each_idx)
        else:
            f.write("," + each_idx)
        idx_count = idx_count + 1
    f.write(")\\n\");\n")

    #   t3: sizes
    f.write("\tprintf (\"                  (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("size_" + each_idx)
        else:
            f.write(", size_" + each_idx)
        idx_count = idx_count + 1
    f.write(");\n")

    #   slices
    f.write("\tprintf (\"           Slices (")
    idx_count = 0
    for t_slice in l_t3_slices:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for t_slice in l_t3_slices:
        if idx_count == len(l_t3_slices) - 1:
            f.write(" " + str(t_slice[1]))
        else:
            f.write(" " + str(t_slice[1]) + ",")
        idx_count = idx_count + 1
    f.write(");\n")
    '''
    #
    f.write("\tprintf (\"             Full (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("n_blk_full_" + each_idx)
        else:
            f.write(", n_blk_full_" + each_idx)
        idx_count = idx_count + 1
    f.write(");\n")

    #
    f.write("\tprintf (\"         Boundary (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("rng_boundary_" + each_idx)
        else:
            f.write(", rng_boundary_" + each_idx)
        idx_count = idx_count + 1
    f.write(");\n")
    '''
    f.write("\tprintf (\" # of Differences (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(") >>> %6d Slices for t3\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("n_blk_" + each_idx)
        else:
            f.write(", n_blk_" + each_idx)
        idx_count = idx_count + 1
    f.write(", *num_thread_blocks_kernel_" + str(idx_kernel) + ");\n")

    f.write("\tprintf (\"==========================================================================================================\\n\");\n")

#
def tc_gen_code_pre_BasicBlock_overview(f, l_t3_idx, l_idx_size, l_t3_slices, l_host_dynamic, l_external_idx, l_internal_idx, possible_diff, idx_kernel):
    #
    f.write("\tprintf (\"==========================================================================================================\\n\");\n")
    f.write("\tprintf (\" >>> %s <<<\\n\", __func__);\n")

    #   t3: indices
    f.write("\tprintf (\"               t3 (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write(each_idx)
        else:
            f.write("," + each_idx)
        idx_count = idx_count + 1
    f.write(")\\n\");\n")

    #   t3: sizes
    f.write("\tprintf (\"                  (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("SIZE_IDX_" + each_idx.capitalize())
        else:
            f.write(", SIZE_IDX_" + each_idx.capitalize())
        idx_count = idx_count + 1
    f.write(");\n")

    #   slices
    f.write("\tprintf (\"           Slices (")
    idx_count = 0
    for t_slice in l_t3_slices:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for t_slice in l_t3_slices:
        if idx_count == len(l_t3_slices) - 1:
            f.write(" " + str(t_slice[1]))
        else:
            f.write(" " + str(t_slice[1]) + ",")
        idx_count = idx_count + 1
    f.write(");\n")

    #
    f.write("\tprintf (\"             Full (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("n_blk_full_" + each_idx)
        else:
            f.write(", n_blk_full_" + each_idx)
        idx_count = idx_count + 1
    f.write(");\n")

    #
    f.write("\tprintf (\"         Boundary (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(")\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("rng_boundary_" + each_idx)
        else:
            f.write(", rng_boundary_" + each_idx)
        idx_count = idx_count + 1
    f.write(");\n")

    f.write("\tprintf (\" # of Differences (")
    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("%2d")
        else:
            f.write(",%2d")
        idx_count = idx_count + 1
    f.write(") >>> %6d Slices for t3\\n\", ")

    idx_count = 0
    for each_idx in l_external_idx:
        if idx_count == 0:
            f.write("n_blk_" + each_idx)
        else:
            f.write(", n_blk_" + each_idx)
        idx_count = idx_count + 1
    f.write(", n_blks_" + str(idx_kernel) + ");\n")

    if possible_diff == 1:
        f.write("\tprintf (\" # of Blocks for     Full Tiles: %4d\\n\", num_blk_full_" + str(idx_kernel) + ");\n")
        f.write("\tprintf (\" # of Blocks for Non-Full Tiles: %4d\\n\", num_blk_non_full_" + str(idx_kernel) + ");\n")

    f.write("\tprintf (\"==========================================================================================================\\n\");\n")
