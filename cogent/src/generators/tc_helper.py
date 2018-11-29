# to add tab(s)
def tc_gen_helper_tabs(f, num_tabs):
    for each_tab in range(num_tabs):
        f.write("\t")

# to write a line--- code
def tc_gen_helper_code_a_line(f, num_tabs, str_code, opt_linebreak):
    #
    for each_tab in range(num_tabs):
        f.write("\t")
    #
    f.write(str_code)
    #
    if opt_linebreak == 1:
        f.write("\n")

# find value by using index from a list ("idx", value)
def tc_gen_helper_find(list, index):
    for temp in list:
        if temp[0] == index:
            return temp[1]
    return -1

def tc_gen_helper_find_1d(list, index):
    for temp in list:
        if temp == index:
            return temp
    return -1
#
def tc_gen_helper_list_pop_str(list, str_target):
    idx_count = 0;
    for each_idx in list:
        if each_idx == str_target:
            list.pop(idx_count)
        idx_count = idx_count + 1

#
def tc_gen_helper_list_pair_pop_str(list, str_target, offset_target):
    idx_count = 0;
    for each_pair in list:
        if each_pair[offset_target] == str_target:
            list.pop(idx_count)
        idx_count = idx_count + 1

#
def tc_gen_helper_list_offset_str(list, str_target):
    idx_count = 0;
    for each_idx in list:
        if each_idx == str_target:
            return idx_count
        idx_count = idx_count + 1
    #
    return -1

def tc_gen_helper_find_exist(list, index):
    for temp in list:
        if temp == index:
            return 1
    return -1

def tc_gen_helper_perms_inputs(tmp, output):
    n = len(tmp)
    a = list(tmp)
    tc_gen_helper_perm(a, 0, n - 1, output)

def toString(List):
    return ''.join(List)

def toPrint(List):
    string = ""
    for element in List:
        string = string + ", " + element
    print (string)

def toPrintWithCount(List, idx_count):
    string = ""
    for element in List:
        string = string + ", " + element
    print (str(idx_count) + ": " + string)

def tc_gen_helper_perm(a, l, r, output):
    if l == r:
        output.append((a[0], a[1], a[2], a[3], a[4]))

    else:
        for i in range(l, r + 1):
            a[l], a[i] = a[i], a[l]
            tc_gen_helper_perm(a, l + 1, r, output)
            a[l], a[i] = a[i], a[l]

#
def tc_gen_helper_decl_varible(f, type, name):
    f.write(type)
    f.write(" ")
    f.write(name)
    f.write(";\n")

#
def tc_gen_helper_CheckingInternalFVI(l_input_tensors, l_internal_idx):
    #
    FVI_left    = 1
    FVI_right   = 1

    #
    if len(l_internal_idx) > 1:
        return FVI_left, FVI_right
    else:
        #
        if l_input_tensors[0][0][1][0] == l_internal_idx[0]:
            FVI_left = -1

        #
        if l_input_tensors[0][1][1][0] == l_internal_idx[0]:
            FVI_right = -1

        return FVI_left, FVI_right
#
def tc_gen_helper_CheckingIntUnit(l_idx_size, l_t3_slices, l_internal_idx):
    internal_unit_size  = 1
    temp_size           = 1

    #
    #   |K| > 1
    #
    if len(l_internal_idx) > 1:
        #print ("len(l_internal_idx): ", len(l_internal_idx), ", ", l_internal_idx)
        for int_idx in l_internal_idx:
            temp_size = temp_size * tc_gen_helper_find(l_t3_slices, int_idx)

        #
        if temp_size >= 16:
            internal_unit_size = 16
        else:
            internal_unit_size = temp_size
        
    #
    #   |K| == 1
    #
    else:
        internal_unit_size = tc_gen_helper_find(l_t3_slices, l_internal_idx[0])

    return internal_unit_size

# l_input_tensors.append(((("t2_195"), ("p7","h1","p4","h2")), (("v2_195"), ("h3","p6","p7","p5")), ("-=")))
# l_inputs_addr.append((((16), ("STR_SD2_T2_195_P7"), ("y"), ("t2_195")), ((16), ("STR_SD2_V2_195_P7"), ("x"), ("v2_195")),   ("-=")))
def tc_gen_helper_CartesianProduct_Inputs(left, right, output_input, output_addr):
    idx_count = 1
    for left_idx in left:
        for right_idx in right:
            #print (str(idx_count) + ": " + left_idx[0] + "," + left_idx[1] + "," + left_idx[2] + "," + left_idx[3] + ",,,"
            #                             + right_idx[0] + "," + right_idx[1] + "," + right_idx[2] + "," + right_idx[3])
            output_input.append ( (
                                    (("t2_" + str(idx_count)), (left_idx[0], left_idx[1], left_idx[2], left_idx[3])),
                                    (("v2_" + str(idx_count)), (right_idx[0], right_idx[1], right_idx[2], right_idx[3])),
                                    ("-=")
                                ) )
            output_addr.append  ( (
                                    ((16), ("STR_SD2_T2_" + str(idx_count) + "_P7"), ("y"), ("t2_" + str(idx_count))),
                                    ((16), ("STR_SD2_V2_" + str(idx_count) + "_P7"), ("x"), ("v2_" + str(idx_count))),
                                    ("-=")
                                ) )
            idx_count = idx_count + 1

#
def tc_gen_helper_CartesianProduct_Tiles(perm_t3_slices, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7):
    possible_tile_sizes = (1, 2, 4, 8, 16)
    total_count = 0
    for t_h3 in possible_tile_sizes:
        for t_h2 in possible_tile_sizes:
            for t_h1 in possible_tile_sizes:
                for t_p6 in possible_tile_sizes:
                    for t_p7 in possible_tile_sizes:
                        total_count = total_count + 1
                        perm_t3_slices.append( ((idx_1, t_h3), (idx_2, t_h2), (idx_3, t_h1), (idx_4, t_p6), (idx_5, 4), (idx_6, 4), (idx_7, t_p7)) )

    print ("Total: " + str(total_count))

#
def tc_gen_helper_CartesianProduct_Tiles_7D(perm_t3_slices, l_t3_mapping_reg, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7):
    possible_tile_sizes = (1, 2, 4, 8, 16)
    total_count         = 0
    is_1 = -1
    is_2 = -1
    is_3 = -1
    is_4 = -1
    is_5 = -1
    is_6 = -1
    is_7 = -1

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_1) != -1:
        is_1 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_2) != -1:
        is_2 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_3) != -1:
        is_3 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_4) != -1:
        is_4 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_5) != -1:
        is_5 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_6) != -1:
        is_6 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_7) != -1:
        is_7 = 0

    #print (is_1, is_2, is_3, is_4, is_5, is_6, is_7)

    total_count = 0
    for t_1 in possible_tile_sizes:
        if is_1 == 0:
            if t_1 != 4:
                continue
        for t_2 in possible_tile_sizes:
            if is_2 == 0:
                if t_2 != 4:
                    continue
            for t_3 in possible_tile_sizes:
                if is_3 == 0:
                    if t_3 != 4:
                        continue
                for t_4 in possible_tile_sizes:
                    if is_4 == 0:
                        if t_4 != 4:
                            continue
                    for t_5 in possible_tile_sizes:
                        if is_5 == 0:
                            if t_5 != 4:
                                continue
                        for t_6 in possible_tile_sizes:
                            if is_6 == 0:
                                if t_6 != 4:
                                    continue
                            for t_7 in possible_tile_sizes:
                                if is_7 == 0:
                                    if t_7 != 4:
                                        continue
                                perm_t3_slices.append(((idx_1, t_1), (idx_2, t_2), (idx_3, t_3), (idx_4, t_4), (idx_5, t_5), (idx_6, t_6), (idx_7, t_7)))
                                total_count = total_count + 1

    #print ("Total: ", total_count)

#
def tc_gen_helper_CartesianProduct_Tiles_6D(perm_t3_slices, l_t3_mapping_reg, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6):
    possible_tile_sizes = (1, 2, 4, 8, 16)
    total_count         = 0
    is_1 = -1
    is_2 = -1
    is_3 = -1
    is_4 = -1
    is_5 = -1
    is_6 = -1

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_1) != -1:
        is_1 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_2) != -1:
        is_2 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_3) != -1:
        is_3 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_4) != -1:
        is_4 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_5) != -1:
        is_5 = 0

    if tc_gen_helper_find_1d(l_t3_mapping_reg, idx_6) != -1:
        is_6 = 0

    #print (is_1, is_2, is_3, is_4, is_5, is_6)
    total_count = 0
    for t_1 in possible_tile_sizes:
        if is_1 == 0:
            if t_1 != 4:
                continue
        for t_2 in possible_tile_sizes:
            if is_2 == 0:
                if t_2 != 4:
                    continue
            for t_3 in possible_tile_sizes:
                if is_3 == 0:
                    if t_3 != 4:
                        continue
                for t_4 in possible_tile_sizes:
                    if is_4 == 0:
                        if t_4 != 4:
                            continue
                    for t_5 in possible_tile_sizes:
                        if is_5 == 0:
                            if t_5 != 4:
                                continue
                        for t_6 in possible_tile_sizes:
                            if is_6 == 0:
                                if t_6 != 4:
                                    continue
                            #print (total_count, t_1, t_2, t_3, t_4, t_5, t_6)
                            perm_t3_slices.append(((idx_1, t_1), (idx_2, t_2), (idx_3, t_3), (idx_4, t_4), (idx_5, t_5), (idx_6, t_6)))
                            total_count = total_count + 1

    print ("Total: ", total_count)

#
def tc_gen_helper_CartesianProduct_Tiles_5D(perm_t3_slices, l_t3_mapping_reg, idx_1, idx_2, idx_3, idx_4, idx_5):
    possible_tile_sizes = (1, 2, 4, 8, 16)
    total_count = 0
    for t_h3 in possible_tile_sizes:
        for t_h2 in possible_tile_sizes:
            for t_p6 in possible_tile_sizes:
                for t_p7 in possible_tile_sizes:
                    perm_t3_slices.append( ((idx_1, t_h3), (idx_2, t_h2), (idx_3, 4), (idx_4, t_p6), (idx_5, 4), (idx_6, t_p7)) )
                    total_count = total_count + 1
    #print ("Total: " + str(total_count))

#
def tc_gen_helper_CartesianProduct_Tiles_General(perm_t3_slices, l_idx_size, l_t3_mapping_reg):
    # Assumption: each size of index is equals to or greater than 16.
    # Otherwise, need to consider constraints.
    if len(l_idx_size) == 7:
        #print ("7D Slices")
        tc_gen_helper_CartesianProduct_Tiles_7D(perm_t3_slices, l_t3_mapping_reg,   l_idx_size[0][0], l_idx_size[1][0], l_idx_size[2][0],
                                                                                    l_idx_size[3][0], l_idx_size[4][0], l_idx_size[5][0], l_idx_size[6][0])
    elif len(l_idx_size) == 6:
        print ("6D Slices: " + l_idx_size[0][0] + "," + l_idx_size[1][0] + "," + l_idx_size[2][0] + "," + l_idx_size[3][0] + "," + l_idx_size[4][0] + "," + l_idx_size[5][0])
        tc_gen_helper_CartesianProduct_Tiles_6D(perm_t3_slices, l_t3_mapping_reg,   l_idx_size[0][0], l_idx_size[1][0], l_idx_size[2][0],
                                                                                    l_idx_size[3][0], l_idx_size[4][0], l_idx_size[5][0])
    elif len(l_idx_size) == 5:
        print ("5D Slices: " + l_idx_size[0][0] + "," + l_idx_size[1][0] + "," + l_idx_size[2][0] + "," + l_idx_size[3][0] + "," + l_idx_size[4][0])
        tc_gen_helper_CartesianProduct_Tiles_5D(perm_t3_slices, l_t3_mapping_reg,   l_idx_size[0][0], l_idx_size[1][0], l_idx_size[2][0],
                                                                                    l_idx_size[3][0], l_idx_size[4][0])
    elif len(l_idx_size) == 4:
        print ("4D Slices")
    elif len(l_idx_size) == 3:
        print ("3D Slices")
    else:
        print ("ERROR: Need to Support " + str(len(l_idx_size)))

#
def tc_gen_helper_CheckingTypes(l_idx_size, l_t3_slices, l_external_idx):
    # Assumption: The Order is Identical.
    # need to check
    opt_gen_full    = -1    # Default: -1 (full), 1 (non-full)
    opt_gen_p7      = -1    # Default: -1 (full), 1 (non-full)
    if len(l_idx_size) == len(l_t3_slices):
        for idx_count in range(0, len(l_idx_size)):
            if tc_gen_helper_find_1d(l_external_idx, l_idx_size[idx_count][0]) != -1:
                if l_idx_size[idx_count][1] % l_t3_slices[idx_count][1] != 0:
                    opt_gen_full = 1
            else:
                if l_idx_size[idx_count][1] % l_t3_slices[idx_count][1] != 0:
                    opt_gen_p7 = 1

    '''
    if opt_gen_full == -1:
        print (">>> Kernel for Full Tiles is needed (opt_gen_full: " + str(opt_gen_full) + ")")
    else:
        print (">>> Kernel for Non-Full Tiles is needed (opt_gen_full: " + str(opt_gen_full) + ")")

    if opt_gen_p7 == -1:
        print (">>> Steps of Internal Index for Full-Tile (opt_gen_p7: " + str(opt_gen_p7) + ")")
    else:
        print (">>> Steps of Internal Index for Non-Full-Tile (opt_gen_p7: " + str(opt_gen_p7) + ")")
    '''

    # For checking Non-Full vs Full
    if opt_gen_full == 1:
        possible_diff = 1
    else:
        possible_diff = -1

    #
    return opt_gen_full, opt_gen_p7, possible_diff

#
def tc_gen_helper_CheckingBoundary(l_blk_boundary_rng, l_idx_size, l_t3_slices, l_t3_mapping_reg, l_t3_mapping_tb_2D, info_left_index, info_right_index):
    '''
    print ("empty: ", l_blk_boundary_rng)
    print ("l_idx_size: ", l_idx_size)
    print ("l_t3_slices: ", l_t3_slices)
    print ("l_t3_mapping_reg: ", l_t3_mapping_reg)
    print ("l_t3_mapping_tb_2D: ", l_t3_mapping_tb_2D)
    print ("left_idx: ", info_left_index)
    print ("right_idx: ", info_right_index)
    '''
    #
    #
    for x_axis in l_t3_mapping_tb_2D[0]:
        if tc_gen_helper_find_1d(info_left_index, x_axis) != -1:
            l_blk_boundary_rng.append((x_axis, tc_gen_helper_find(l_idx_size, x_axis) % tc_gen_helper_find(l_t3_slices, x_axis)))
        else:
            if tc_gen_helper_find_1d(info_right_index, x_axis) != -1:
                l_blk_boundary_rng.append((x_axis, tc_gen_helper_find(l_idx_size, x_axis) % tc_gen_helper_find(l_t3_slices, x_axis)))
            else:
                print ("ERROR: tc_helper.tc_gen_helper_CheckingBoundary()")

    #
    #
    for y_axis in l_t3_mapping_tb_2D[1]:
        if tc_gen_helper_find_1d(info_left_index, y_axis) != -1:
            l_blk_boundary_rng.append((y_axis, tc_gen_helper_find(l_idx_size, y_axis) % tc_gen_helper_find(l_t3_slices, y_axis)))
        else:
            if tc_gen_helper_find_1d(info_right_index, y_axis) != -1:
                l_blk_boundary_rng.append((y_axis, tc_gen_helper_find(l_idx_size, y_axis) % tc_gen_helper_find(l_t3_slices, y_axis)))
            else:
                print ("ERROR: tc_helper.tc_gen_helper_CheckingBoundary()")

#
def tc_gen_helper_CompareTwoLists2D(l_src, l_dest):
    #
    # 
    #
    for each_src in l_src:
        for each_dest in l_dest:
            print ("")

    
