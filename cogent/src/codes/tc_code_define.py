import src.generators.tc_helper     as tc_helper
import src.codes.tc_code_etc        as tc_code_etc

#
#
#
def tc_gen_definition_new(f, l_combined_t3_slices, l_combined_mappings, l_external_idx, l_internal_idx):
    #
    #
    #
    #print (l_combined_mappings)
    #print (l_combined_t3_slices)
    
    tc_code_etc.tc_gen_code_write_line(f, 0, "// created by tc_gen_definition_new()")

    #
    idx_kernel = 1
    for each_inner_group in l_combined_t3_slices:
        tc_gen_definition_slices(f, idx_kernel, each_inner_group)
        tc_gen_definition_internal_indices_new(f, idx_kernel, l_internal_idx)
        idx_kernel = idx_kernel + 1

    #
    idx_kernel = 1
    for each_inner_group in l_combined_mappings:
        tc_gen_definition_mappings_ext(f, idx_kernel, each_inner_group[0], each_inner_group[1])
        idx_kernel = idx_kernel + 1

    #
    tc_gen_code_helper_define(f, "NUM_INDEX",   "\t\t" + str(len(l_external_idx)))

    #
    tc_gen_code_helper_define(f, "CEIL(a, b)",  "\t\t(((a) + (b) - 1) / (b))")
    #

    #
    #   |K| > 1: Constant Memory, ()
    #
    if len(l_internal_idx) > 1:
        f.write("\n")
        f.write("// Not Yet: Multiple Tensor Contractions.\n")
        f.write("// |Constant Memory| = 64KB, 16K Words(Integer), which means |K| <= 8192\n")
        tc_gen_code_helper_define(f, "MAX_CONST_LEN", "\t\t8192")
        f.write("__constant__ int const_internal_t2_offset[MAX_CONST_LEN];\n")
        f.write("__constant__ int const_internal_v2_offset[MAX_CONST_LEN];\n")

#
#
#                        >> Inner <<                                    >> Inner <<                                           >> Inner <<
def tc_gen_definition(f, l_combined_t3_slices, l_idx_size, l_t3_idx, l_combined_mappings, l_external_idx, l_internal_idx, l_combined_input_tensors):
    # definitions
    f.write("\n")
    f.write("// created by tc_gen_definition()\n")

    #   To-Do: Inner-Groups
    #   Definitions for Sizes of Slices
    #
    idx_kernel = 1
    for each_inner_group in l_combined_t3_slices:
        tc_gen_definition_slices(f, idx_kernel, each_inner_group)
        idx_kernel = idx_kernel + 1

    #
    #   Defintions for Sizes for external and internal indices
    #
    tc_gen_definition_indices(f, l_idx_size)

    #
    #   Definitions for multiple contraction indices
    #
    tc_gen_definition_internal_indices(f, l_internal_idx)   # 4: l_internal_idx

    #   To-Do: Inner-Groups
    #   Definitions for Mapping and External Indices
    #
    idx_kernel = 1
    for each_inner_group in l_combined_mappings:
        tc_gen_definition_mappings_ext(f, idx_kernel, each_inner_group[0], each_inner_group[1])
        idx_kernel = idx_kernel + 1

    #
    tc_gen_code_helper_define(f, "NUM_INDEX",   "\t" + str(len(l_external_idx)))
    f.write("\n")

    #
    #   Definitions for Strides of Ouput: T3
    #
    tc_gen_definition_strides_output(f, l_idx_size, l_t3_idx)

    #   To-Do: Inner-Groups
    #   Definitions for Strides of Inputs: T2, and V2
    #
    for each_input_tensors in l_combined_input_tensors:
        tc_gen_definition_strides_input(f, each_input_tensors)

    #
    #   Macros used in the code
    #
    tc_gen_code_helper_define(f, "CEIL(a, b)", "(((a) + (b) - 1) / (b))")
    f.write("\n")


#
def tc_gen_definition_strides_output(f, l_idx_size, l_t3_idx):
    f.write("// t3 for output\n")
    # for t3,
    val_prev = 1
    str_prev = ""
    for t3_idx in l_t3_idx:
        if val_prev == 1:
            tc_gen_code_helper_define(f, "STR_SD2_T3_" + str(t3_idx.capitalize()),
                                         "1")
        else:
            tc_gen_code_helper_define(f, "STR_SD2_T3_" + str(t3_idx.capitalize()),
                                         "STR_SD2_T3_" + str_prev.capitalize() + " * " + "SIZE_IDX_" + str_prev.capitalize())
        str_prev = t3_idx
        val_prev = tc_helper.tc_gen_helper_find(l_idx_size, t3_idx)
    f.write("\n")

#
def tc_gen_definition_strides_input(f, l_input_tensors):
    # for t2 and v2
    # l_input_tensors.append(((("t2_1"), ("p4","p7","h1","h2")), (("v2_1"), ("p6","p7","h3","p5"))))
    for single_tc in l_input_tensors:
        val_prev = 1
        str_prev = ""
        f.write("// t2 for inputs\n")
        for t2_idx in single_tc[0][1]:
            def_name = "STR_SD2_" + single_tc[0][0].capitalize() + "_" + t2_idx.capitalize()
            if val_prev == 1:
                tc_gen_code_helper_define(f, def_name, "1")
            else:
                tc_gen_code_helper_define(f, def_name, str_prev + " * " + val_prev)
            str_prev = def_name
            val_prev = "SIZE_IDX_" + t2_idx.capitalize()
        f.write("\n")

        val_prev = 1
        str_prev = ""
        f.write("// v2 for inputs\n")
        for v2_idx in single_tc[1][1]:
            def_name = "STR_SD2_" + single_tc[1][0].capitalize() + "_" + v2_idx.capitalize()
            if val_prev == 1:
                tc_gen_code_helper_define(f, def_name, "1")
            else:
                tc_gen_code_helper_define(f, def_name, str_prev + " * " + val_prev)
            str_prev = def_name
            val_prev = "SIZE_IDX_" + v2_idx.capitalize()
        f.write("\n")

#
def tc_gen_definition_mappings_ext(f, idx_kernel, l_t3_mapping_tb_2D, l_t3_mapping_reg):
    #
    idx_count       = 0
    str_size_TB_X   = "\t"
    for x_idx in l_t3_mapping_tb_2D[0]:
        if idx_count != 0:
            str_size_TB_X = str_size_TB_X + " * "
        str_size_TB_X = str_size_TB_X + "SIZE_SLICE_" + str(idx_kernel) + "_" + x_idx.capitalize()
        idx_count = idx_count + 1

    idx_count       = 0
    str_size_TB_Y   = "\t"
    for y_idx in l_t3_mapping_tb_2D[1]:
        if idx_count != 0:
            str_size_TB_Y = str_size_TB_Y + " * "
        str_size_TB_Y = str_size_TB_Y + "SIZE_SLICE_" + str(idx_kernel) + "_" + y_idx.capitalize()
        idx_count = idx_count + 1

    tc_gen_code_helper_define(f, "SIZE_TB_"  + str(idx_kernel) + "_X",   str_size_TB_X)
    tc_gen_code_helper_define(f, "SIZE_TB_"  + str(idx_kernel) + "_Y",   str_size_TB_Y)
    tc_gen_code_helper_define(f, "SIZE_REG_" + str(idx_kernel) + "_X",  "\tSIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_reg[0].capitalize())
    tc_gen_code_helper_define(f, "SIZE_REG_" + str(idx_kernel) + "_Y",  "\tSIZE_SLICE_" + str(idx_kernel) + "_" + l_t3_mapping_reg[1].capitalize())
    f.write("\n")

#
def tc_gen_definition_internal_indices(f, l_internal_idx):
    #
    #   To-Do: Should be fixed Correctly
    #   To-Do: Temporally and Manually, I put "SIZE_SLICE_1_P7"
    #
    str_internal_indices    = ""
    if len(l_internal_idx) > 1:
        str_internal_indices = "16"
    else:
        idx_count               = 0
        for int_idx in l_internal_idx:
            if idx_count == 0:
                str_internal_indices = "SIZE_SLICE_1_" + int_idx.capitalize()
            else:
                str_internal_indices = str_internal_indices + " * SIZE_SLICE_1_" + int_idx.capitalize()
            idx_count = idx_count + 1

    #
    tc_gen_code_helper_define(f, "SIZE_INT_UNIT", str_internal_indices)
    f.write("\n")

#
def tc_gen_definition_internal_indices_new(f, idx_kernel, l_internal_idx):
    #
    #   To-Do: Should be fixed Correctly
    #   To-Do: Temporally and Manually, I put "SIZE_SLICE_1_P7"
    #
    str_internal_indices    = ""
    if len(l_internal_idx) > 1:
        #str_internal_indices = "16"
        idx_count               = 0
        for int_idx in l_internal_idx:
            if idx_count == 0:
                str_internal_indices = "SIZE_SLICE_" + str(idx_kernel) + "_" + int_idx.capitalize()
            else:
                str_internal_indices = str_internal_indices + " * SIZE_SLICE_" + str(idx_kernel) + "_" + int_idx.capitalize()
            idx_count = idx_count + 1
    else:
        idx_count               = 0
        for int_idx in l_internal_idx:
            if idx_count == 0:
                str_internal_indices = "SIZE_SLICE_" + str(idx_kernel) + "_" + int_idx.capitalize()
            else:
                str_internal_indices = str_internal_indices + " * SIZE_SLICE_" + str(idx_kernel) + "_" + int_idx.capitalize()
            idx_count = idx_count + 1

    #
    tc_gen_code_helper_define(f, "SIZE_INT_UNIT_" + str(idx_kernel), str_internal_indices)
    f.write("\n")


#
#   Common for Tensor Contractions in an Inner-Group
#
def tc_gen_definition_indices(f, l_idx_size):
    for idx in l_idx_size:
        tc_gen_code_helper_define(f, "SIZE_IDX_" + idx[0].capitalize(), idx[1])
    f.write("\n")

#
#   Common for Tensor Contractions in an Inner-Group
#
def tc_gen_definition_slices(f, idx_kernel, l_t3_slices):
    #print (">>>> l_t3_slices: ", l_t3_slices)
    for idx in l_t3_slices:
        tc_gen_code_helper_define(f, "SIZE_SLICE_" + str(idx_kernel) + "_" + idx[0].capitalize(), idx[1])
    f.write("\n")

#
def tc_gen_code_helper_define(f, name, value):
    f.write("#define ")
    f.write(name)
    f.write(" ")
    f.write(str(value))
    f.write("\n")
