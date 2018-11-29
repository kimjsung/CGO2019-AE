

#
def print_input_tensor_contractions(f, l_input_tensors):
    tc_count = 1
    for each_tc in l_input_tensors:
        f.write("\tprintf (\"#: %3d, " + each_tc[0][0] + "[")
        idx_count = 0
        for each_idx in each_tc[0][1]:
            if idx_count == 0:
                f.write(each_idx)
            else:
                f.write("," + each_idx)
            idx_count = idx_count + 1
        f.write("] * " + each_tc[1][0] + "[")
        idx_count = 0
        for each_idx in each_tc[1][1]:
            if idx_count == 0:
                f.write(each_idx)
            else:
                f.write("," + each_idx)
            idx_count = idx_count + 1

        f.write("]\\n\", " + str(tc_count) + ");\n")
        tc_count = tc_count + 1
