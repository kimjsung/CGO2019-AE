import src.generators.tc_helper     as tc_helper

#
#
#
def tc_gen_code_pre_SD2_Functions(f, l_t3_idx, l_internal_idx, l_input_tensors, l_host_dynamic, idx_kernel):
    f.write("\n")
    f.write("// created by tc_gen_code_pre_SD2_Functions()\n")
    f.write("void pre_SD2_Functions_" + str(idx_kernel) + "()\n")
    f.write("{\n")
    f.write("\tsetlocale(LC_ALL,\"\");\n")

    if idx_kernel == 1:
        #   Init "Size"
        tc_pre_SD2_Functions_Init_Sizes(f, l_internal_idx, l_t3_idx)

        #   Init "Output"
        tc_pre_SD2_Functions_Init_Output(f, l_host_dynamic)

    #
    #   Input Tensors   >>> Inner-Groups <<<
    #
    idx_tc = 1
    for each_func in l_input_tensors:
        #   Init "Left"
        tc_pre_SD2_Functions_Init_Left(f, each_func, l_host_dynamic, idx_tc)

        #   Init "Right"
        tc_pre_SD2_Functions_Init_Right(f, each_func, l_host_dynamic, idx_tc)
        idx_tc = idx_tc + 1

    #
    #   >>> Inner-Groups <<<
    #
    tc_pre_SD2_Functions_Overview(f, l_input_tensors)

    # End of "pre_SD2_Functions();""
    f.write("}\n")

#
def tc_pre_SD2_Functions_Init_Left(f, each_func, l_host_dynamic, idx_tc):
    #   Left
    f.write("\tsize_" + each_func[0][0].capitalize() + " = ")
    idx_count = 0
    for t2_idx in each_func[0][1]:
        f.write("SIZE_IDX_" + t2_idx.capitalize())
        if idx_count == len(each_func[0][1]) - 1:
            f.write(";\n")
        else:
            f.write(" * ")
        idx_count = idx_count + 1
    #
    f.write("\th_" + each_func[0][0] + " = (double*)malloc(sizeof(double) * size_" + each_func[0][0].capitalize() + ");\n")
    l_host_dynamic.append("h_" + each_func[0][0])
    f.write("\n")

    #   initializing LEFT
    f.write("\tfor (int j = 0; j < size_" + each_func[0][0].capitalize() + "; j++)\n")
    f.write("\t{\n")
    f.write("\t\th_" + each_func[0][0] + "[j] = ((double)rand() / RAND_MAX);\n")
    f.write("\t}\n")
    f.write("\n")

#
def tc_pre_SD2_Functions_Init_Right(f, each_func, l_host_dynamic, idx_tc):
    #   RIGHT
    f.write("\tsize_" + each_func[1][0].capitalize() + " = ")
    idx_count = 0
    for v2_idx in each_func[1][1]:
        f.write("SIZE_IDX_" + v2_idx.capitalize())
        if idx_count == len(each_func[1][1]) - 1:
            f.write(";\n")
        else:
            f.write(" * ")
        idx_count = idx_count + 1

    #   initializing RIGHT
    f.write("\th_" + each_func[1][0] + " = (double*)malloc(sizeof(double) * size_" + each_func[1][0].capitalize() + ");\n")
    l_host_dynamic.append("h_" + each_func[1][0])
    f.write("\n")

    f.write("\tfor (int j = 0; j < size_" + each_func[1][0].capitalize() + "; j++)\n")
    f.write("\t{\n")
    f.write("\t\th_" + each_func[1][0] + "[j] = ((double)rand() / RAND_MAX);\n")
    f.write("\t}\n")
    f.write("\n")

#
def tc_pre_SD2_Functions_Init_Output(f, l_host_dynamic):
    # for t3,
    f.write("\th_t3     = (double*)malloc(sizeof(double) * size_T3);\n")
    l_host_dynamic.append("h_t3")
    f.write("\th_t3_chk = (double*)malloc(sizeof(double) * size_T3);\n")
    l_host_dynamic.append("h_t3_chk")
    f.write("\n")

    # initializing t3
    f.write("\t// initializing t3\n")
    f.write("\tfor (int j = 0; j < size_T3; j++)\n")
    f.write("\t{\n")
    f.write("\t\th_t3[j]        = 0.0;\n")
    f.write("\t\th_t3_chk[j]    = 0.0;\n")
    f.write("\t}\n")
    f.write("\n")

#
def tc_pre_SD2_Functions_Init_Sizes(f, l_internal_idx, l_t3_idx):
    #
    idx_count               = 0
    size_product_int_idx    = ""
    #print (l_internal_idx)

    for int_idx in l_internal_idx:
        if idx_count == 0:
            size_product_int_idx = "SIZE_IDX_" + int_idx.capitalize()
        else:
            size_product_int_idx = size_product_int_idx + " * SIZE_IDX_" + int_idx.capitalize()
        idx_count = idx_count + 1

    # |internal indices|
    f.write("\tsize_internal = " + size_product_int_idx + ";\n")
    f.write("\tsize_T3 = ")
    idx = 0
    for t3_idx in l_t3_idx:
        f.write("SIZE_IDX_" + t3_idx.capitalize())
        if idx == len(l_t3_idx) - 1:
            f.write(";\n")
        else:
            f.write(" * ")
        idx = idx + 1
    f.write("\n")

#
def tc_pre_SD2_Functions_Overview(f, l_input_tensors):

        f.write("\tprintf (\"==========================================================================================================\\n\");\n")
        f.write("\tprintf (\" >>> %s <<<\\n\", __func__);\n")
        f.write("\tprintf (\"   T3: %'12d\\n\", size_T3);\n")

        for each_input in l_input_tensors:
            f.write("\tprintf (\" " + each_input[0][0].capitalize() + ": %'12d, ")
            f.write(each_input[1][0].capitalize() + ": %'12d\\n\", size_" + each_input[0][0].capitalize() + ", size_" + each_input[1][0].capitalize())
            f.write(");\n")

        f.write("\tprintf (\"==========================================================================================================\\n\");\n")
