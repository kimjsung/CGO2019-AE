#
import src.generators.tc_helper     as tc_helper

#
def tc_gen_code_Post_Correctness(f, l_external_idx, l_internal_idx, l_combined_input_tensors):
    #
    tc_gen_code_Post_Correctness_Head(f)

    #
    f.write("\tlong long int    tmp_ops = 0;\n")
    f.write("\tint              ops     = 0;\n")
    f.write("\n")

    #
    for ext_idx in l_external_idx:
        f.write("\tfor (int t3_" + ext_idx + " = 0; t3_" + ext_idx + " < SIZE_IDX_" + ext_idx.capitalize() + "; t3_" + ext_idx + "++)\n")

    # inner-external-indices
    f.write("\t{\n")

    f.write("\t\tint tmp_r_idx = ")
    ext_count = 0
    for ext_idx in l_external_idx:
        f.write("t3_" + ext_idx + " * STR_SD2_T3_" + ext_idx.capitalize())
        if ext_count == len(l_external_idx) - 1:
            f.write(";\n")
        else:
            f.write(" + ")
        ext_count = ext_count + 1

    for int_idx in l_internal_idx:
        f.write("\t\tfor (int t3_" + int_idx + " = 0; t3_" + int_idx + " < SIZE_IDX_" + int_idx.capitalize() + "; t3_" + int_idx + "++, ops = 0)\n")

    # inner-internal-indices
    f.write("\t\t{\n")

    #. based on fused group (len(l_t2_groups) == len(l_v2_groups) (Assumptions))
    for each_inner_group in l_combined_input_tensors:
        for sd2_func in each_inner_group:
            #print (">>> ", sd2_func[0], ",", sd2_func[1], ",", sd2_func[2], ":", type(sd2_func[2]))
            f.write("\t\t\th_t3_chk[tmp_r_idx] " + sd2_func[2] + " ")
            f.write("h_" + sd2_func[0][0] + "[")
            idx_count = 0
            for func_idx in sd2_func[0][1]:
                f.write("t3_" + func_idx + " * STR_SD2_" + sd2_func[0][0].capitalize() + "_" + func_idx.capitalize())
                if idx_count == len(sd2_func[0][1]) - 1:
                    f.write("]")
                else:
                    f.write(" + ")
                idx_count = idx_count + 1

            f.write(" * h_" + sd2_func[1][0] + "[")
            idx_count = 0
            for func_idx in sd2_func[1][1]:
                f.write("t3_" + func_idx + " * STR_SD2_" + sd2_func[1][0].capitalize() + "_" + func_idx.capitalize())
                if idx_count == len(sd2_func[1][1]) - 1:
                    f.write("];\n")
                else:
                    f.write(" + ")
                idx_count = idx_count + 1

            f.write("\t\t\tops++;\n")
            f.write("\n")

    f.write("\t\t\ttmp_ops = tmp_ops + ops;\n")
    f.write("\t\t}\n")

    f.write("\t}\n")
    f.write("\n")

    # To compare two outputs
    tc_gen_code_Post_Correctness_Print_Overview(f)

    f.write("}\n")  # End of "post_Correctness();"

#
def tc_gen_code_Post_Correctness_Head(f):
    f.write("\n")
    f.write("// created by tc_gen_code_Post_Correctness()\n")
    f.write("void post_Correctness()\n")
    f.write("{\n")

# 
def tc_gen_code_Post_Correctness_Print_Overview(f):
    f.write("\tprintf (\"======================================= Correctness Check ==========================================\\n\");\n")
    f.write("\tdouble   epsilon = 0.00000001;\n")
    f.write("\tint      diff    = 0;\n")
    f.write("\tint      same    = 0;\n")
    f.write("\tfor (int i = 0; i < size_T3; i++)\n")
    f.write("\t{\n")
    f.write("\t\tdouble check = h_t3_chk[i] - h_t3[i];\n")
    f.write("\t\tif (check < 0) check *= -1;\n")
    #f.write("\t\tif (check/max(h_t3_chk[i], h_t3[i]) > epsilon)\n")
    f.write("\t\tif (check > epsilon)\n")
    f.write("\t\t{\n")
    f.write("\t\t\tdiff++;\n")
    f.write("\t\t\tif (diff < 8)\n")
    f.write("\t\t\tprintf (\"Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\\n\", i, h_t3_chk[i], h_t3[i], std::abs(h_t3_chk[i] - h_t3[i]));\n")
    #f.write("\t\t\tprintf (\"Index: %5d, (Host) %f, (Dev.) %f >> (Diff.) %f\\n\", i, h_t3_chk[i] * 1000000000000, h_t3[i] * 1000000000000, std::abs(h_t3_chk[i] - h_t3[i]));\n")
    f.write("\t\t}\n")
    f.write("\t\telse\n")
    f.write("\t\t{\n")
    f.write("\t\t\tsame++;\n")
    f.write("\t\t}\n")
    f.write("\t}\n")
    f.write("\n")

    f.write("\tprintf (\" >>> PASSED: %'10d among %'10d in t3\\n\", same, size_T3);\n")
    f.write("\tprintf (\" >>> ERROR : %'10d among %'10d in t3\\n\", diff, size_T3);\n")
    f.write("\tprintf (\" >>> Total Operations: %'lld\\n\", tmp_ops * 2);\n")
    f.write("\tprintf (\"====================================================================================================\\n\");\n")
