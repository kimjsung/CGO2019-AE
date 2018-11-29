#
#
#
#
#   An Interface to Call the Above Caller. 
#
def tc_gen_code_interface(f, interface_name, l_interface_info, l_tile_sizes):
    #
    #
    f.write("\n")
    f.write("// This is written by tc_interface.tc_gen_code_interface()\n")
    f.write("// This Interface Should be Called to Run the Kernels\n")
    f.write("extern \"C\"\n")
    f.write("void " + interface_name + "_(")

    #
    #   l_interface_info: [0] All Index, [1] Output, [2] Inputs, [3] Conditions, [4] Options
    #   If the code generator supports multiple-output-groups, this should be corrected.
    #
    #   [0] All Index
    #
    l_split_info = l_interface_info[0][5]
    if len(l_interface_info[0][5]) > 0:
        #
        idx_count = 0
        for each_index in l_interface_info[0][0]:
            #
            #   Checking if this index is used for Split or not
            #
            is_skip = -1
            for each_split_info in l_split_info:
                #
                if each_index == each_split_info[1]:
                    each_index = each_split_info[0]
                elif each_index == each_split_info[2]:
                    is_skip = 1
            #
            #
            #
            if is_skip != 1:
                if idx_count == 0:
                    f.write("int size_" + each_index)
                else:
                    f.write(", int size_" + each_index)
                idx_count = idx_count + 1
            else:
                is_skip == -1   # reset
    else:
        idx_count = 0
        for each_index in l_interface_info[0][0]:
            if idx_count == 0:
                f.write("int size_" + each_index)
            else:
                f.write(", int size_" + each_index)
            idx_count = idx_count + 1

    #   [1] Output
    f.write(", double* " + l_interface_info[0][1])

    #   [2] Inputs
    for each_pair_inputs in l_interface_info[0][2]:
        for each_input in each_pair_inputs:
            f.write(", double* " + each_input)
    
    #   [3] Conditions
    for each_condition in l_interface_info[0][3]:
        f.write(", int " + each_condition)

    #   [4] Option(s): (Currently) Only One Option for Register Transpose
    f.write(", int " + l_interface_info[0][4])

    f.write(")\n")
    f.write("{\n")

    #   [>] Pre-Processing
    f.write("\t// Pre-Processing for Split\n")
    f.write("\t// Based on Tile-Sizes and Problem-Size\n")
    f.write("\t// Currently, one index can be split into two indices\n")

    # l_split_info
    for each_split in l_split_info:
        print (">> each_split: ", each_split)
        print (">> l_tile_sizes: ", l_tile_sizes)
        #
        #   Each Split-Infomation
        #
        tc_code_etc.tc_gen_code_write_line(f, 1, "int size_" + each_split[1] + ";")
        tc_code_etc.tc_gen_code_write_line(f, 1, "int size_" + each_split[2] + ";")

        str_first_tile_size = str(tc_helper.tc_gen_helper_find(l_tile_sizes, each_split[1]))

        #
        #   Pre-Processing
        #
        tc_code_etc.tc_gen_code_write_line(f, 0, "")
        tc_code_etc.tc_gen_code_write_line(f, 1, "if (size_" + each_split[0] + " % " + str_first_tile_size + " == 0)")
        tc_code_etc.tc_gen_code_write_line(f, 1, "{")

        #
        tc_code_etc.tc_gen_code_write_line(f, 2, "//")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[1] + " = " + str_first_tile_size + ";")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[2] + " = size_" + each_split[0] + " / " + str_first_tile_size + ";")

        #
        tc_code_etc.tc_gen_code_write_line(f, 1, "}")
        tc_code_etc.tc_gen_code_write_line(f, 1, "else")
        tc_code_etc.tc_gen_code_write_line(f, 1, "{")

        #
        tc_code_etc.tc_gen_code_write_line(f, 2, "//")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[1] + " = size_" + each_split[0] + ";")
        tc_code_etc.tc_gen_code_write_line(f, 2, "size_" + each_split[2] + " = 1;")
        
        #
        tc_code_etc.tc_gen_code_write_line(f, 1, "}")
        
    f.write("\n")

    f.write("\t// Call An Application\n")
    f.write("\t" + interface_name + "(")
    
    #
    #   [0] All Index (Split-Version)
    #
    idx_count = 0
    for each_index in l_interface_info[0][0]:
        if idx_count == 0:
            f.write("size_" + each_index)
        else:
            f.write(", size_" + each_index)
        idx_count = idx_count + 1

    
    #   [1] Output
    f.write(", " + l_interface_info[0][1])

    #   [2] Inputs
    for each_pair_inputs in l_interface_info[0][2]:
        for each_input in each_pair_inputs:
            f.write(", " + each_input)
    
    #   [3] Conditions
    for each_condition in l_interface_info[0][3]:
        f.write(", " + each_condition)

    #   [4] Option(s):
    f.write(", " + l_interface_info[0][4])
    f.write(");\n")


    f.write("}\n")