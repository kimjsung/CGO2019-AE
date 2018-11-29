#
import copy
#
import src.generators.tc_helper             as tc_helper

#
#
#
def tc_gen_perms_exclusive_REG_X(list_sizes_REG, list_sizes_TB,
                                                list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                list_internal_indices,
                                                list_representative_problem_size,
                                                list_TB_K, list_TB_X,
                                                list_CLASS_configuration,
                                                opt_print):
    #
    #
    #
    num_ext_idx = 0
    num_int_idx = 0
    for each_left_idx in list_given_input_tensor_left:
        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_left_idx) == -1:
            num_ext_idx += 1
        else:
            num_int_idx += 1
    #
    len_tensor_left = len(list_given_input_tensor_left)
    
    if opt_print == 1:
        print ("========================================== [Enumerations-REG_X] ===================================================")
        print ("========================================== [Exclusive]          ===================================================")
        print ("Tensor (LEFT): ", list_given_input_tensor_left)
        print ("len(LEFT): ", len_tensor_left, ", # of External Indices: ", num_ext_idx, ", # of Internal Indices: ", num_int_idx)
        print ("list_representative_problem_size: ", list_representative_problem_size)

    #
    #   For Each Tile-Size for REG_X
    #
    for size_REG_X in list_sizes_REG:
        if opt_print == 1:
            print ("|REG_X| = ", size_REG_X)

        #
        #
        #
        for start_index in range(0, len_tensor_left):
            #   
            REG_X_Vol           = 1
            REG_X_Vol_Prev      = 1
            list_REG_X          = []    # will be inherited
            list_Tile_Sizes     = []    # will be inherited
            done_mapping_REG_X  = -1    # not done

            #
            #
            #
            for target_index in range(start_index, len_tensor_left):
                str_start_index = list_given_input_tensor_left[target_index]
                if opt_print == 1:
                    print ("idx: ", str_start_index)

                #
                #   #1. Internal Index
                #
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, str_start_index) != -1:
                    continue

                #
                #   #2. The FVI in the Output Tensor
                #
                if str_start_index == list_given_output_tensor[0]:
                    continue

                if opt_print == 1:
                    print (">> idx: ", str_start_index)
                #
                #   |REG_X'|
                #
                REG_X_Vol *= tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)

                if opt_print == 1:
                    print (">> idx: ", str_start_index, ", REG_X_Vol: ", REG_X_Vol)

                #
                #
                #
                if REG_X_Vol >= size_REG_X:
                    #
                    #   |REG_X'| > |REG_X|
                    #
                    if REG_X_Vol > size_REG_X:
                        #
                        #   Need to Split (REG and BX)
                        #
                        if done_mapping_REG_X == -1:
                            blocking_tile_size = size_REG_X / REG_X_Vol_Prev
                            list_REG_X.append(str_start_index)
                            list_Tile_Sizes.append([str_start_index, int(blocking_tile_size)])
                            done_mapping_REG_X = 1
                        else:
                            list_Tile_Sizes.append([str_start_index, 1])
                    #
                    #   |REG_X'| = |REG_X|
                    #
                    else: 
                        #
                        #
                        #
                        if done_mapping_REG_X == -1:
                            list_REG_X.append(str_start_index)
                            list_Tile_Sizes.append([str_start_index, tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)])
                            done_mapping_REG_X = 1
                        else:
                            list_Tile_Sizes.append([str_start_index, 1])
                    #
                    #
                    #
                    break
                else:
                    list_REG_X.append(str_start_index)
                    list_Tile_Sizes.append([str_start_index, tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)])
                
                #
                #
                #
                REG_X_Vol_Prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)
            #
            #
            #
            if done_mapping_REG_X == 1:
                if opt_print == 1:
                    print ("list_REG_X: ", list_REG_X)
                    print ("list_Tile_sizes: ", list_Tile_Sizes)
                tc_gen_perms_exclusive_REG_Y(list_sizes_REG, list_sizes_TB,
                                        list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                        list_internal_indices, list_representative_problem_size,
                                        list_TB_K, list_TB_X,
                                        list_REG_X,
                                        list_Tile_Sizes,
                                        list_CLASS_configuration,
                                        opt_print)


#
#
#
def tc_gen_perms_exclusive_REG_Y(list_sizes_REG, list_sizes_TB,
                                                list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                list_internal_indices,
                                                list_representative_problem_size,
                                                list_TB_K, list_TB_X,
                                                list_REG_X,
                                                list_inherited_Tile_Sizes,
                                                list_CLASS_configuration,
                                                opt_print):
    #
    #
    #
    num_ext_idx = 0
    num_int_idx = 0
    for each_right_idx in list_given_input_tensor_right:
        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_right_idx) == -1:
            num_ext_idx += 1
        else:
            num_int_idx += 1
    #
    len_tensor_right = len(list_given_input_tensor_right)
    
    if opt_print == 1:
        print ("========================================== [Enumerations-REG_Y] ===================================================")
        print ("========================================== [Exclusive]          ===================================================")
        print ("Tensor (LEFT): ", list_given_input_tensor_right)
        print ("len(LEFT): ", len_tensor_right, ", # of External Indices: ", num_ext_idx, ", # of Internal Indices: ", num_int_idx)
        print ("list_representative_problem_size: ", list_representative_problem_size)
        print ("Given Tile-Sizes: ", list_inherited_Tile_Sizes)
        print ("Given list_REG_X: ", list_REG_X)

    #
    #   For Each Tile-Size for REG_X
    #
    for size_REG_Y in list_sizes_REG:
        if opt_print == 1:
            print ("|REG_Y| = ", size_REG_Y)

        #
        #
        #
        for start_index in range(0, len_tensor_right):
            #   
            REG_Y_Vol               = 1
            REG_Y_Vol_Prev          = 1
            list_REG_Y              = []    # inherited
            duplicated_Tile_Sizes   = copy.deepcopy(list_inherited_Tile_Sizes)    # inherited
            done_mapping_REG_Y      = -1    # not done

            #
            #
            #
            for target_index in range(start_index, len_tensor_right):
                str_start_index = list_given_input_tensor_right[target_index]

                #
                #   #1. Internal Index
                #
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, str_start_index) != -1:
                    continue

                #
                #   #2. The FVI in the Output Tensor
                #
                if str_start_index == list_given_output_tensor[0]:
                    continue

                #
                #   |REG_Y'|
                #
                REG_Y_Vol *= tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)

                #
                #   |REG_Y'| >= |REG_Y|
                #
                if REG_Y_Vol >= size_REG_Y:
                    #
                    #   |REG_Y'| > |REG_Y|
                    #
                    if REG_Y_Vol > size_REG_Y:
                        #
                        #   Need to SPlit (REG and BX)
                        #
                        if done_mapping_REG_Y == -1:
                            blocking_tile_size = size_REG_Y / REG_Y_Vol_Prev
                            list_REG_Y.append(str_start_index)
                            duplicated_Tile_Sizes.append([str_start_index, int(blocking_tile_size)])
                            done_mapping_REG_Y = 1
                        else:
                            duplicated_Tile_Sizes.append([str_start_index, 1])  # ?
                    #
                    #   |REG_Y'| = |REG_Y|
                    #
                    else:
                        #
                        #
                        #
                        if done_mapping_REG_Y == -1:
                            list_REG_Y.append(str_start_index)
                            duplicated_Tile_Sizes.append([str_start_index, tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)])
                            done_mapping_REG_Y = 1
                        else:
                            duplicated_Tile_Sizes.append([str_start_index, 1])
                    #
                    #
                    #
                    break
                #
                #   |REG_Y'| < |REG_Y|
                #
                else:
                    list_REG_Y.append(str_start_index)
                    duplicated_Tile_Sizes.append([str_start_index, tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)])
    
                #
                #
                #
                REG_Y_Vol_Prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)
            #
            #
            #
            if done_mapping_REG_Y == 1:
                tc_gen_perms_exclusive_TB_X(list_sizes_TB,
                                                    list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                    list_internal_indices,
                                                    list_representative_problem_size,
                                                    list_TB_K, list_TB_X,
                                                    list_REG_X, list_REG_Y,
                                                    duplicated_Tile_Sizes,
                                                    list_CLASS_configuration,
                                                    opt_print)


#
#
#
def tc_gen_perms_exclusive_TB_X(list_sizes_TB,
                                                list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                list_internal_indices,
                                                list_representative_problem_size,
                                                list_TB_K, list_TB_X,
                                                list_REG_X, list_REG_Y,
                                                list_inherited_Tile_Sizes,
                                                list_CLASS_configuration,
                                                opt_print):
    #
    #
    #
    num_ext_idx = 0
    num_int_idx = 0
    for each_right_idx in list_given_input_tensor_left:
        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_right_idx) == -1:
            num_ext_idx += 1
        else:
            num_int_idx += 1
    #
    len_tensor_left = len(list_given_input_tensor_left)

    if opt_print == 1:
        print ("========================================== [Enumerations-TB_X]  ===================================================")
        print ("========================================== [Exclusive] [START]  ===================================================")
        print ("Tensor (LEFT): ", list_given_input_tensor_left)
        print ("len(LEFT): ", len_tensor_left, ", # of External Indices: ", num_ext_idx, ", # of Internal Indices: ", num_int_idx)
        print ("list_representative_problem_size: ", list_representative_problem_size)
        print ("Given Tile-Sizes: ", list_inherited_Tile_Sizes)
        print ("Given REG_X: ", list_REG_X)
        print ("Given REG_Y: ", list_REG_Y)
        print ("Given TB_X:  ", list_TB_X)
        print ("========================================== [Exclusive]   [END]  ===================================================")

    #
    #
    #
    for size_TB_X in list_sizes_TB:
        if opt_print == 1:
            print ("|TB_X| = ", size_TB_X)

        #
        #   Assumption: Input Tensor whose index is the FVI in the Output will be related to REG_X and TB_X.
        #
        TB_X_Vol                = -1
        TB_X_Vol_Prev           = -1
        done_mapping_TB_X       = -1
        duplicated_TB_X         = copy.deepcopy(list_TB_X)
        duplicated_Tile_Sizes   = copy.deepcopy(list_inherited_Tile_Sizes)

        #
        #   Handling the FVI (Default)
        #
        for each_left_idx in list_given_input_tensor_left:
            if each_left_idx == list_TB_X[0]:
                size_FVI = tc_helper.tc_gen_helper_find(list_representative_problem_size, each_left_idx)
                #
                #   Need to Split
                #
                if size_FVI > size_TB_X:
                    duplicated_Tile_Sizes.append([each_left_idx, size_TB_X])
                    TB_X_Vol            = size_TB_X
                    TB_X_Vol_Prev       = size_TB_X
                    done_mapping_TB_X   = 1
                #
                #   No Need to Split (Fitted)
                #
                elif size_FVI == size_TB_X:
                    duplicated_Tile_Sizes.append([each_left_idx, size_TB_X])
                    TB_X_Vol            = size_TB_X
                    TB_X_Vol_Prev       = size_TB_X
                    done_mapping_TB_X   = 1
                #
                #   No Need to Split
                #
                else:
                    duplicated_Tile_Sizes.append([each_left_idx, size_FVI])
                    TB_X_Vol        = size_FVI
                    TB_X_Vol_Prev   = size_FVI

        #
        #
        #
        for each_left_idx in list_given_input_tensor_left:
            #
            #   #1. Internal Index
            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_left_idx) != -1:
                continue
            
            #
            #   #2. Indices Mapped on REG_X
            #
            if tc_helper.tc_gen_helper_find_1d(list_REG_X, each_left_idx) != -1:
                continue

            #
            #   #3. (Just In Case) Indices Mapped on REG_Y
            #
            if tc_helper.tc_gen_helper_find_1d(list_REG_Y, each_left_idx) != -1:
                continue

            #
            #   #4. Indiced Mapped on TB_X (Should Use list_TB_X (passed by))
            #
            if tc_helper.tc_gen_helper_find_1d(list_TB_X, each_left_idx) != -1:
                continue

            #
            #   |TB_X'|
            #
            TB_X_Vol *= tc_helper.tc_gen_helper_find(list_representative_problem_size, each_left_idx)

            #
            #   |TB_X'| >= |TB_X|
            #
            if TB_X_Vol >= size_TB_X:
                #
                #   |TB_X'| > |TB_X|
                #
                if TB_X_Vol > size_TB_X:
                    #
                    #
                    #
                    if done_mapping_TB_X == -1:
                        blocking_tile_size = size_TB_X / TB_X_Vol_Prev
                        duplicated_TB_X.append(each_left_idx)
                        duplicated_Tile_Sizes.append([each_left_idx, int(blocking_tile_size)])
                        done_mapping_TB_X = 1
                    else:
                        duplicated_TB_X.append(each_left_idx)
                        duplicated_Tile_Sizes.append([each_left_idx, 1])
                #
                #   |TB_X'| = |TB_X|
                #
                else:
                    #
                    #
                    #
                    if done_mapping_TB_X == -1:
                        duplicated_TB_X.append(each_left_idx)
                        duplicated_Tile_Sizes.append([each_left_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_left_idx)])
                    else:
                        duplicated_TB_X.append(each_left_idx)
                        duplicated_Tile_Sizes.append([each_left_idx, 1])
            #
            #   |TB_X'| < |TB_X|
            #
            else:
                duplicated_TB_X.append(each_left_idx)
                duplicated_Tile_Sizes.append([each_left_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_left_idx)])
            
            #
            #
            #
            TB_X_Vol_Prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, each_left_idx)

        #
        #
        #
        if done_mapping_TB_X == 1:
            tc_gen_perms_exclusive_TB_Y(list_sizes_TB,
                                                list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                list_internal_indices,
                                                list_representative_problem_size,
                                                list_TB_K, duplicated_TB_X,
                                                list_REG_X, list_REG_Y,
                                                duplicated_Tile_Sizes,
                                                list_CLASS_configuration,
                                                opt_print)
#
#
#
def tc_gen_perms_exclusive_TB_Y(list_sizes_TB,
                                                list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                list_internal_indices,
                                                list_representative_problem_size,
                                                list_TB_K, list_TB_X,
                                                list_REG_X, list_REG_Y,
                                                list_inherited_Tile_Sizes,
                                                list_CLASS_configuration, opt_print):
    #
    #
    #
    num_ext_idx = 0
    num_int_idx = 0
    for each_right_idx in list_given_input_tensor_right:
        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_right_idx) == -1:
            num_ext_idx += 1
        else:
            num_int_idx += 1
    #
    len_tensor_right = len(list_given_input_tensor_right)

    if opt_print == 1:
        print ("========================================== [Enumerations-TB_Y]  ===================================================")        
        print ("========================================== [Exclusive] [START]  ===================================================")
        print ("Tensor (LEFT): ", list_given_input_tensor_right)
        print ("len(LEFT): ", len_tensor_right, ", # of External Indices: ", num_ext_idx, ", # of Internal Indices: ", num_int_idx)
        print ("list_representative_problem_size: ", list_representative_problem_size)
        print ("Given Tile-Sizes: ", list_inherited_Tile_Sizes)
        print ("Given REG_X: ", list_REG_X)
        print ("Given REG_Y: ", list_REG_Y)
        print ("Given TB_X:  ", list_TB_X)
        print ("========================================== [Exclusive]   [END]  ===================================================")

    #
    #
    #
    for size_TB_Y in list_sizes_TB:
        if opt_print == 1:
            print ("|TB_Y| = ", size_TB_Y)

        #
        #   Assumption: This Input Tensor does not have the FVI in the Output Tensor.
        #
        TB_Y_Vol                =  1
        TB_Y_Vol_Prev           =  1
        done_mapping_TB_Y       = -1
        list_TB_Y               = []
        duplicated_Tile_Sizes   = copy.deepcopy(list_inherited_Tile_Sizes)

        #
        #
        #
        for each_right_idx in list_given_input_tensor_right:
            #
            #   #1. Internal Index
            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_right_idx) != -1:
                continue
            
            #
            #   #2. (Just In Case) Indices Mapped on REG_X
            #
            if tc_helper.tc_gen_helper_find_1d(list_REG_X, each_right_idx) != -1:
                continue

            #
            #   #3. Indices Mapped on REG_Y
            #
            if tc_helper.tc_gen_helper_find_1d(list_REG_Y, each_right_idx) != -1:
                continue
            
            #
            #   |TB_Y'|
            #
            TB_Y_Vol *= tc_helper.tc_gen_helper_find(list_representative_problem_size, each_right_idx)

            #
            #   |TB_Y'| >= |TB_XY
            #
            if TB_Y_Vol >= size_TB_Y:
                #
                #   |TB_Y'| > |TB_Y|
                #
                if TB_Y_Vol > size_TB_Y:
                    #
                    #
                    #
                    if done_mapping_TB_Y == -1:
                        blocking_tile_size = size_TB_Y / TB_Y_Vol_Prev
                        list_TB_Y.append(each_right_idx)
                        duplicated_Tile_Sizes.append([each_right_idx, int(blocking_tile_size)])
                        done_mapping_TB_Y = 1
                    else:
                        list_TB_Y.append(each_right_idx)
                        duplicated_Tile_Sizes.append([each_right_idx, 1])
                        
                #
                #   |TB_Y'| = |TB_Y|
                #
                else:
                    #
                    #
                    #
                    if done_mapping_TB_Y == -1:
                        list_TB_Y.append(each_right_idx)
                        duplicated_Tile_Sizes.append([each_right_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_right_idx)])
                    else:
                        list_TB_Y.append(each_right_idx)
                        duplicated_Tile_Sizes.append([each_right_idx, 1])
            #
            #   |TB_Y'| < |TB_Y|
            #
            else:
                list_TB_Y.append(each_right_idx)
                duplicated_Tile_Sizes.append([each_right_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_right_idx)])

            #
            #
            #
            TB_Y_Vol_Prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, each_right_idx)
        #
        #
        #
        #print ("list_TB_X: ", list_TB_X)
        #print ("list_TB_Y: ", list_TB_Y)
        #print ("Tile-Sizes: ", duplicated_Tile_Sizes)
        #
        #   Configuration
        #
        if done_mapping_TB_Y == 1:
            #
            #   #1. Shared-Memory |SMEM_L| = |SMEM_R|
            #
            size_SMEM_Left      = 1
            size_SMEM_Right     = 1

            #
            #
            #
            for each_idx in list_given_input_tensor_left:
                if tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx) != -1:
                    size_SMEM_Left *= tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx)
            
            for each_idx in list_given_input_tensor_right:
                if tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx) != -1:
                    size_SMEM_Right *= tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx)

            #
            #   #1. H/W Constraint---- Shared Memory
            #
            if (size_SMEM_Left * 16) + (size_SMEM_Right * 16) > 4096:
                continue

            #
            #
            #
            if size_SMEM_Left == size_SMEM_Right:
                tmp_config = tc_gen_permutations.Configuration()
                tmp_config.add_tensor_C(list_given_output_tensor)
                tmp_config.add_tensor_A(list_given_input_tensor_left)
                tmp_config.add_tensor_B(list_given_input_tensor_right)
                tmp_config.add_REG_X(list_REG_X)
                tmp_config.add_REG_Y(list_REG_Y)
                tmp_config.add_TB_X(list_TB_X)
                tmp_config.add_TB_Y(list_TB_Y)
                tmp_config.add_TB_K(list_TB_K)

                #
                #   [To-Do] Need to Make it Automatically
                #
                #duplicated_Tile_Sizes.append(["e", 16])
                #duplicated_Tile_Sizes.append(["f", 1])
                #duplicated_Tile_Sizes.append(["g", 16])

                #duplicated_Tile_Sizes.append(["d", 16]) # for 15
                duplicated_Tile_Sizes.append(["f", 16]) #
                #duplicated_Tile_Sizes.append(["e", 16]) # 3

                tmp_config.add_tile_size(duplicated_Tile_Sizes)
                tmp_config.add_representative_problem_size(list_representative_problem_size)

                #
                tmp_config.size_REG_X   = 1
                tmp_config.size_REG_Y   = 1
                tmp_config.size_TB_X    = 1
                tmp_config.size_TB_Y    = 1
                tmp_config.size_TB_K    = 1
                for each_idx in list_REG_X:
                    tmp_config.size_REG_X *= tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx)

                for each_idx in list_REG_Y:
                    tmp_config.size_REG_Y *= tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx)

                for each_idx in list_TB_X:
                    tmp_config.size_TB_X *= tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx)

                for each_idx in list_TB_Y:
                    tmp_config.size_TB_Y *= tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx)

                for each_idx in list_TB_K:
                    tmp_config.size_TB_K *= tc_helper.tc_gen_helper_find(duplicated_Tile_Sizes, each_idx)
                '''
                print (">>>> list_TB_X: ", tmp_config.list_TB_X)
                print (">>>> list_TB_Y: ", tmp_config.list_TB_Y)
                print (">>>> list_TB_K: ", tmp_config.list_TB_K)
                print (">>>> list_REG_X: ", tmp_config.list_REG_X)
                print (">>>> list_REG_Y: ", tmp_config.list_REG_Y)
                print (">>>> list_Tile_Sizes: ", tmp_config.list_tile_sizes)
                print (">>>> # of Elements in SMEM_L: ", size_SMEM_Left * 16)
                print (">>>> # of Elements in SMEM_R: ", size_SMEM_Right * 16)
                '''
                #
                list_CLASS_configuration.append(tmp_config)

