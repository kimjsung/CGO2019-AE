'''
    configuration.py
    : mapping and tile-sizes
'''
import sys
import copy

import src.generators.tc_helper             as tc_helper
import src.algs.bases.class_configuration   as class_config


#
#   [Main-Function]
#
def alg_enumeration_pruning(tensor_contraction, list_info_split, list_representative_problem_size, opt_limited_split, opt_print, opt_data_type):
    #
    #list_pruned_configurations          = [] # TB_X, TB_Y, REG_X, REG_Y, Tile-Sizes
    list_pruned_configurations_class    = []

    #
    if opt_print == 1:
        print ("============================================================================================")
        print (" tensor_contration: ", tensor_contraction)
        print (" list_representative_problem_size: ", list_representative_problem_size)
        print ("============================================================================================")

    #
    if opt_data_type == "DOUBLE":
        list_tiles_TB       = [4, 8, 16]#, 32]
        list_tiles_TB_Y     = [4, 8, 16]
        list_tiles_REG      = [1, 2, 4, 6, 8]
        list_tiles_REG_Y    = [1, 2, 4, 6, 8]
    else:
        list_tiles_TB       = [4, 8, 16]
        list_tiles_TB_Y     = [4, 8, 16]
        list_tiles_REG      = [1, 2, 4, 6, 8]
        list_tiles_REG_Y    = [1, 2, 4, 6, 8]
    #
    #   Pre-Processing from the Given Inputs such as a Tensor Contraction and a Representative Problem 
    #   each_tensor_contraction[0]: output-tensor name
    #   each_tensor_contraction[1]: output-tensor indices
    #   each_tensor_contraction[2]: operator (+= or -=)
    #   each_tensor_contraction[3]: internal indices
    #   each_tensor_contraction[4]: input-tensor name
    #   each_tensor_contraction[5]: input-tensor indices
    #   each_tensor_contraction[6]: input-tensor name
    #   each_tensor_contraction[7]: input-tensor indices
    #
    list_internal_indices   = tensor_contraction[3]
    list_output_tensor      = tensor_contraction[1]
    list_input_tensor_left  = tensor_contraction[5]
    list_input_tensor_right = tensor_contraction[7]

    #
    if opt_print == 1:
        print ("============================== [Enumerations-ALL] ==========================================")
        print (" List of |TB_X|  or |TB_Y|:  ", list_tiles_TB)
        print (" List of |REG_X| or |REG_Y|: ", list_tiles_REG)
        print (" Given Tensor Contraction: ", tensor_contraction)
        print (" > Output Tensor (a.k.a. External Indices): ", list_output_tensor)
        print (" > Input Tensor (LEFT):  ", list_input_tensor_left)
        print (" > Input Tensor (RIGHT): ", list_input_tensor_right)
        print (" > Internal Indices: ", list_internal_indices)
        print ("============================================================================================")

    #
    #   Assumption: Indices in an input tensor will be mapped on one of x-axis and y-axis exclusively.
    #               This is just for "a single tensor contraction."
    #
    #   [0] One of Input Tensors whose one of indices is the FVI in the output tensor will be mapped on x-axis.
    #
    opt_fvi_input = -1  
    for each_idx in list_input_tensor_left:
        if each_idx == list_output_tensor[0]:
            opt_swap = 1

    for each_idx in list_input_tensor_right:
        if each_idx == list_output_tensor[0]:
            opt_swap = 2

    #
    #   options-- prints
    #
    opt_print_K     = -1         # need to be revised when there are multiple internal indices.
    opt_print_E_L   = -1
    opt_print_E_R   = -1

    #
    #   [Assumption]
    #
    if opt_swap == 1:
        print ("[Code Generator][Configurations] L. Tensor has THE FVI in the Output")
    else:
        print ("[Code Generator][Configurations] R. Tensor has THE FVI in the Output")
        list_input_tensor_left  = tensor_contraction[7]
        list_input_tensor_right = tensor_contraction[5]
        if opt_print == 1:
            print (" > Input Tensor (LEFT):  ", list_input_tensor_left)
            print (" > Input Tensor (RIGHT): ", list_input_tensor_right)
    print ("============================================================================")

    #
    #   [Internal Indices]
    #
    list_partial_config_TB_K = alg_config_K(list_internal_indices, list_representative_problem_size, list_tiles_TB, opt_print_K)
    
    #   [Completed][Partial-Configurations][K] --- TB
    print ("[Code Generator][Configurations] # of Configurations--- K: ", len(list_partial_config_TB_K))
    if opt_print == 1:
        print ("============================================================================")
        for each_partial_config in list_partial_config_TB_K:
            print ("each_partial_config_K: ", each_partial_config)
        print ("============================================================================")

    #
    #   [External Indices][LEFT]
    #
    list_partial_config_LEFT_TB_REG = alg_config_E_L(list_input_tensor_left, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tiles_TB, list_tiles_REG, opt_print_E_L)
    print ("[Code Generator][Configurations] # of Configurations--- E (LEFT): ", len(list_partial_config_LEFT_TB_REG))
    if opt_print == 1:
        print ("============================================================================")
        for each_partial_config in list_partial_config_LEFT_TB_REG:
            print ("each_partial_config_E_L: ", each_partial_config)
        print ("============================================================================")

    #
    #   [External Indices][RIGHT]
    #
    list_partial_config_RIHGT_TB_REG = alg_config_E_R(list_input_tensor_right, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tiles_TB_Y, list_tiles_REG_Y, opt_print_E_R)

    #   [Completed][Partial-Configurations][E] --- TB && REG
    print ("[Code Generator][Configurations] # of Configurations--- E (RIGHT): ", len(list_partial_config_RIHGT_TB_REG))
    if opt_print == 1:
        print ("============================================================================")
        for each_partial_config in list_partial_config_RIHGT_TB_REG:
            print ("each_partial_config_E_R: ", each_partial_config)
        print ("============================================================================")

    #
    #   [Total Configurations] = |K| * (|E_L| * |R_L|) * (|E_R| * |R_R|)
    #
    #list_pruned_configurations = [] # TB_X, TB_Y, REG_X, REG_Y, Tile-Sizes
    num_configurations              = 0
    num_configurations_smem_sized   = 0
    for each_config_K in list_partial_config_TB_K:
        tmp_TB_K            = each_config_K[1]
        tmp_TB_K_tile_sizes = each_config_K[2]
        #print ("tmp_TB_K: ", tmp_TB_K, ", tmp_TB_K_tile_sizes: ", tmp_TB_K_tile_sizes)
        for each_config_L in list_partial_config_LEFT_TB_REG:
            tmp_TB_X            = each_config_L[0]
            tmp_REG_X           = each_config_L[1]
            tmp_X_tile_sizes    = each_config_L[2]
            #print ("tmp_TB_X: ", tmp_TB_X, ", REG_X: ", tmp_REG_X, ", tiles: ", tmp_X_tile_sizes)
            for each_config_R in list_partial_config_RIHGT_TB_REG:
                tmp_TB_Y            = each_config_R[0]
                tmp_REG_Y           = each_config_R[1][0]
                tmp_Y_tile_sizes    = each_config_R[1][1]
                #print ("tmp_TB_Y: ", tmp_TB_Y, ", REG_Y: ", tmp_REG_Y, ", tiles: ", tmp_Y_tile_sizes)

                #
                tmp_combined_tile_size = []
                for each_tile in tmp_TB_K_tile_sizes:
                    tmp_combined_tile_size.append(each_tile)
                
                for each_tile in tmp_X_tile_sizes:
                    tmp_combined_tile_size.append(each_tile)
                
                for each_tile in tmp_Y_tile_sizes:
                    tmp_combined_tile_size.append(each_tile)
                #
                #   
                #
                num_configurations += 1

                #
                #   [Constraint #1] |TB_X| * |TB_Y| > 16
                #
                size_TB_X = 1
                size_TB_Y = 1
                for each_idx in tmp_TB_X:
                    size_TB_X *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)

                for each_idx in tmp_TB_Y:
                    size_TB_Y *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)

                #
                #   |TB_X| > 4 or |TB_Y| > 4
                #
                if size_TB_X == 4 or size_TB_Y == 4:
                    continue

                #
                #   [Constraint #2] Arithmetic Intensity: 2.0 // |REG_X|, |REG_Y| 
                #
                size_REG_X = 1
                size_REG_Y = 1
                for each_idx in tmp_REG_X:
                    size_REG_X *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)

                for each_idx in tmp_REG_Y:
                    size_REG_Y *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)
                
                #
                #   # of registers =< 64 + alpha <= 72
                #
                if opt_data_type == "DOUBLE":
                    if (size_REG_X * size_REG_Y) > 36:
                        continue
                else:
                    if (size_REG_X * size_REG_Y) > 36:
                        continue
                #
                #   
                #
                #print ("(size_REG_X * size_REG_Y) / (size_REG_X + size_REG_Y): ", (size_REG_X * size_REG_Y) / (size_REG_X + size_REG_Y))
                if (size_REG_X * size_REG_Y) / (size_REG_X + size_REG_Y) < 2.0: # it allows 4 x 4 register tiling
                    continue


                #
                #   Class: Config()
                #
                #
                #   [Constraint #2]     1st: |SMEM_L| == |SMEM_R| 
                #                       2nd: 2 * |SMEM_L| == |SMEM_R| or |SMEM_L| == 2 * |SMEM_R|
                #
                size_SMEM_L = 1
                size_SMEM_R = 1

                for each_idx in list_input_tensor_left:
                    if tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx) != -1:
                        size_SMEM_L *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)
                
                for each_idx in list_input_tensor_right:
                    if tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx) != -1:
                        size_SMEM_R *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)
            
                #
                #   2nd (changed)
                #
                #if size_SMEM_L == size_SMEM_R or size_SMEM_L*2 == size_SMEM_R or size_SMEM_L == size_SMEM_R*2:
                if size_SMEM_L == size_SMEM_R:
                    tmp_config = class_config.Config()
                    tmp_config.add_tensor_C(list_output_tensor)
                    tmp_config.add_tensor_A(list_input_tensor_left)
                    tmp_config.add_tensor_B(list_input_tensor_right)
                    tmp_config.add_REG_X(tmp_REG_X)
                    tmp_config.add_REG_Y(tmp_REG_Y)
                    tmp_config.add_TB_X(tmp_TB_X)
                    tmp_config.add_TB_Y(tmp_TB_Y)
                    tmp_config.add_TB_K(tmp_TB_K)
                    tmp_config.add_split_index(list_info_split)

                    tmp_config.add_representative_problem_size(list_representative_problem_size)
                    tmp_config.add_tile_size(tmp_combined_tile_size)

                    #
                    for each_idx in tmp_REG_X:
                        tmp_config.size_REG_X *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)                    

                    for each_idx in tmp_REG_Y:
                        tmp_config.size_REG_Y *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)
                    
                    for each_idx in tmp_TB_X:
                        tmp_config.size_TB_X *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)
                    
                    for each_idx in tmp_TB_Y:
                        tmp_config.size_TB_Y *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)

                    for each_idx in tmp_TB_K:
                        tmp_config.size_TB_K *= tc_helper.tc_gen_helper_find(tmp_combined_tile_size, each_idx)

                    #
                    list_pruned_configurations_class.append(tmp_config)
    #
    print ("[Code Generator][Configurations] # of Configurations--- E_L * E_R * K: ", num_configurations)
    print ("[Code Generator][Configurations] # of Configurations--- E_L * E_R * K (pruned by constraints): ", len(list_pruned_configurations_class))
    return list_pruned_configurations_class

#
#   [Configuration][Algorithm][Right][Thread-Block] --- "TB_X" && REG_X
#
def alg_config_E_R(list_input_tensor, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tiles_TB, list_tile_REG, opt_print):
    #
    list_partial_config_E = []

    #
    if opt_print == 1:
        print ("[Algorithm][Configuration][External Index][Right] ", list_input_tensor)
        print ("[Algorithm][Configuration][External Index][Right] ", list_representative_problem_size)

    #
    #   [Double-Check] The Right Tensor does not have the Output Tensor's FVI.
    #
    opt_has_output_fvi = -1
    for each_idx in list_input_tensor:
        if each_idx == list_output_tensor[0]:
            opt_has_output_fvi = 1
            print("[Algorithm][Configuration][External Index][Right] ERROR: Given Right Tensor has the Output's FVI")
            sys.exit()
    
    #
    #   Per Each |TB| Size,
    #
    for each_size_TB in list_tiles_TB:
        if opt_print == 1:
            print ("[Algorithm][Configuration][External Index][Right] |TB| = ", each_size_TB)

        #
        #
        #
        for start_idx in range(0, len(list_input_tensor)):
            if opt_print == 1:
                print (" > start_idx: ", start_idx)
            #
            #   
            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[start_idx]) != -1:
                continue
            
            #
            #
            #
            vol_TB                  = 1
            vol_TB_prev             = 1
            list_TB                 = []
            list_temp_tile_sizes    = []
            opt_done                = -1

            #
            if opt_print == 1:
                print (" > (pruned) start_idx: ", start_idx)
            
            #
            #   "target_idx" from "start_idx" to |T|.length
            #
            for target_idx in range(start_idx, len(list_input_tensor)):
                #
                if opt_print == 1:
                    print (" >> target_idx: ", target_idx)
                #
                #
                #
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[target_idx]) != -1:
                    continue

                #
                if opt_print == 1:
                    print (" >> (pruned) target_idx: ", target_idx)
                
                #
                vol_TB *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])

                #   |TB'| >= |TB|
                if vol_TB >= each_size_TB:
                    #   |TB'| > |TB|
                    if vol_TB > each_size_TB:
                        blocking_tile_size = int(each_size_TB / vol_TB_prev)
                        list_TB.append(list_input_tensor[target_idx])
                        list_temp_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                    #   |TB'| = |TB|
                    else:
                        list_TB.append(list_input_tensor[target_idx])
                        list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                    #
                    opt_done = 1
                    break
                #   |TB'| < |TB|
                else:
                    list_TB.append(list_input_tensor[target_idx])
                    list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                
                #
                vol_TB_prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])
            
            #
            #   "target_idx" from "0" to "start_idx"
            #
            if opt_done == -1:
                for target_idx in range(0, start_idx):
                    #
                    #
                    #
                    if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[target_idx]) != -1:
                        continue

                    #
                    vol_TB *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])

                    #   |TB'| >= |TB|
                    if vol_TB >= each_size_TB:
                        #   |TB'| > |TB|
                        if vol_TB > each_size_TB:
                            blocking_tile_size = int(each_size_TB / vol_TB_prev)
                            list_TB.append(list_input_tensor[target_idx])
                            list_temp_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                        #   |TB'| = |TB|
                        else:
                            list_TB.append(list_input_tensor[target_idx])
                            list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                        #
                        opt_done = 1
                        break
                    #   |TB'| < |TB|
                    else:
                        list_TB.append(list_input_tensor[target_idx])
                        list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                    #
                    vol_TB_prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])

            #
            #
            #         
            if opt_done == 1:
                #
                #   Goes to Mapping REG
                #
                list_partial_config_E_R = alg_config_E_R_R(list_input_tensor, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tile_REG, list_TB, list_temp_tile_sizes, 0)

                #
                #   Checks: unmapped external indices
                #
                if len(list_partial_config_E_R) > 0:
                    #
                    #   Per Each Config_E_R
                    #
                    for each_config_E_R in list_partial_config_E_R:
                        #
                        #   Check if there exists unmapped indices or not.
                        #
                        list_TB_copied          = copy.deepcopy(list_TB)
                        for each_idx in list_input_tensor:
                            if tc_helper.tc_gen_helper_find_1d(list_TB, each_idx) != -1:
                                continue
                            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_idx) != -1:
                                continue
                            if tc_helper.tc_gen_helper_find_1d(each_config_E_R[0], each_idx) != -1:
                                continue

                            list_TB_copied.append(each_idx)
                            each_config_E_R[1].append([each_idx, 1])
                        #
                        list_partial_config_E.append([list_TB_copied, each_config_E_R])
                            




                '''
                print ("TB: ", list_TB)
                for each_config_ER in list_partial_config_E_R:
                    print (" REG: ", each_config_ER)
                
                #
                if len(list_partial_config_E_R) > 0:
                    for each_config in list_partial_config_E_R:
                        list_partial_config_E.append([list_TB, each_config])
                '''
    #
    return list_partial_config_E

#
#   [Configuration][Algorithm][Right][Register] --- TB_X && "REG_X"
#
def alg_config_E_R_R(list_input_tensor, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tile_REG, list_base_TB, list_base_tile_sizezs, opt_print):
    #
    list_partial_config_E_R = []

    #
    if opt_print == 1:
        print ("------------------------------------------------------------------------------------------------------------------------")
        print ("[Algorithm][Configuration][External-Index][Right][REG] T: ", list_input_tensor)
        print ("[Algorithm][Configuration][External-Index][Right][REG] Size: ", list_representative_problem_size)
        print ("[Algorithm][Configuration][External-Index][Right][REG] list_TB: ", list_base_TB, ", ", list_base_tile_sizezs)
        print ("------------------------------------------------------------------------------------------------------------------------")

    #
    for each_size_REG in list_tile_REG:
        #
        opt_fvi = -1

        #
        if opt_print == 1:
            print ("[Algorithm][Configuration][External Index][Right][REG] |REG| = ", each_size_REG)
        #
        #
        #
        for start_idx in range(0, len(list_input_tensor)):
            #
            if opt_print == 1:
                print ("[Algorithm][Configuration][External Index][Right][REG] start_idx: ", start_idx, ", ", list_input_tensor[start_idx])
            
            #
            if tc_helper.tc_gen_helper_find_1d(list_base_TB, list_input_tensor[start_idx]) != -1:
                continue

            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[start_idx]) != -1:
                continue
            
            #
            if start_idx == 0:
                continue

            #
            if opt_print == 1:
                print ("[Algorithm][Configuration][External Index][Right][REG] (pruned) start_idx: ", start_idx, ", ", list_input_tensor[start_idx])

            #
            vol_REG                     = 1
            vol_REG_prev                = 1
            list_REG                    = []
            list_inherited_TB           = copy.deepcopy(list_base_TB)
            list_inherited_tile_sizes   = copy.deepcopy(list_base_tile_sizezs)
            opt_done                    = -1

            #
            #   [1] "target_idx" from "start_idx" to |T|.length
            #
            for target_idx in range(start_idx, len(list_input_tensor)):
                #
                if opt_print == 1:
                    print ("[1] target_idx: ", target_idx)
                
                #
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[target_idx]) != -1:
                    continue
                #
                if tc_helper.tc_gen_helper_find_1d(list_base_TB, list_input_tensor[target_idx]) != -1:
                    continue

                #
                vol_REG *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])

                #   |REG'| >= |REG|
                if vol_REG >= each_size_REG:
                    #   |REG'| > |REG|
                    if vol_REG > each_size_REG:
                        blocking_tile_size = int(each_size_REG / vol_REG_prev)
                        list_REG.append(list_input_tensor[target_idx])
                        list_inherited_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                    #   |REG'| = |REG|
                    else:
                        list_REG.append(list_input_tensor[target_idx])
                        list_inherited_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                    #
                    opt_done = 1
                    break
                #   |REG'| < |REG|
                else:
                    list_REG.append(list_input_tensor[target_idx])
                    list_inherited_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])

            #
            #   [2] "target_idx" from "0" to "start_idx"
            #
            if opt_done == -1:
                for target_idx in range(0, start_idx):
                    #
                    if opt_print == 1:
                        print ("[2] target_idx: ", target_idx)

                    #
                    if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[target_idx]) != -1:
                        continue
                    #
                    if tc_helper.tc_gen_helper_find_1d(list_base_TB, list_input_tensor[target_idx]) != -1:
                        continue
                    #
                    vol_REG *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])

                    #   |REG'| >= |REG|
                    if vol_REG >= each_size_REG:
                        #   |REG'| > |REG|
                        if vol_REG > each_size_REG:
                            blocking_tile_size = int(each_size_REG / vol_REG_prev)
                            list_REG.append(list_input_tensor[target_idx])
                            list_inherited_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                        #   |REG'| = |REG|
                        else:
                            list_REG.append(list_input_tensor[target_idx])
                            list_inherited_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                        #
                        opt_done = 1
                        break
                    #   |REG'| < |REG|
                    else:
                        list_REG.append(list_input_tensor[target_idx])
                        list_inherited_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])

            #
            #   Per Tile-Size, based on a specific "start_idx," there is a mapping for REG.
            #
            if opt_done == 1:
                opt_fvi = 1
                if opt_print == 1:
                    print ("list_REG: ", list_REG)
                    print ("list_tile_sizezs: ", list_inherited_tile_sizes)
                list_partial_config_E_R.append([list_REG, list_inherited_tile_sizes])
        #
        #
        #
        if opt_fvi == -1:
            #
            #   [Double-Check]
            #
            opt_double_check = -1
            for each_idx in list_input_tensor:
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_idx) != -1:
                    continue
                if tc_helper.tc_gen_helper_find_1d(list_base_TB, each_idx) != -1:
                    continue
                #
                opt_double_check = 1
            
            #
            if opt_double_check == 1:
                list_inherited_tile_sizes = copy.deepcopy(list_base_tile_sizezs)
                list_inherited_tile_sizes.append([list_input_tensor[0], each_size_REG])
                list_partial_config_E_R.append([[list_input_tensor[0]], list_inherited_tile_sizes])
    #
    return list_partial_config_E_R

#
#   [Configuration][Algorithm][LEFT][Thread-Block] --- "TB_Y" && REG_Y
#
def alg_config_E_L(list_input_tensor, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tiles_TB, list_tile_REG, opt_print):
    #
    opt_print_E_L_R         = -1
    list_partial_config_E   = []

    #
    if opt_print == 1:
        print ("list_input_tensor: ", list_input_tensor)

    #
    #   [Double-Check] if the LEFT Tensor has the OUTPUT's FVI or not.
    #
    opt_has_output_fvi = -1
    for each_idx in list_input_tensor:
        if each_idx == list_output_tensor[0]:
            opt_has_output_fvi = 1
    #
    if opt_has_output_fvi == -1:
        print ("[Algorithm][Configuration] ERROR!")

    #
    #   Per Each |TB| Size,
    #
    for each_size_TB in list_tiles_TB:
        if opt_print == 1:
            print ("========================================================================================================================")
            print ("each_size_TB: ", each_size_TB)
        #
        default_vol_TB                  = 1
        default_list_TB                 = []
        default_list_temp_tile_sizes    = []
        default_opt_done                = -1
        
        #
        #   [Special Case] 
        #
        default_vol_TB *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_output_tensor[0])
        default_list_TB.append(list_output_tensor[0])

        #   |TB_X'| >= |TB_X|
        if default_vol_TB >= each_size_TB:
            #   |TB_X'| > |TB_X|
            if default_vol_TB > each_size_TB:
                blocking_tile_size = int(each_size_TB)
                #default_list_TB.append(list_output_tensor[0])
                default_list_temp_tile_sizes.append([list_output_tensor[0], blocking_tile_size])
            #   |TB_X'| = |TB_X|
            else:
                #default_list_TB.append(list_output_tensor[0])
                default_list_temp_tile_sizes.append([list_output_tensor[0], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_output_tensor[0])])
            #
            #   opt_done = 1
            #
            default_opt_done = 1
        #   |TB_X'| < |TB_X|
        else:
            #default_list_TB.append(list_output_tensor[0])
            default_list_temp_tile_sizes.append([list_output_tensor[0], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_output_tensor[0])])

        #
        #   [Normal Case]
        #
        if default_opt_done == -1:
            #
            #   "start_idx" from "0" to |T|.length
            #
            for start_idx in range(0, len(list_input_tensor)):
                #
                vol_TB                  = default_vol_TB
                vol_TB_prev             = default_vol_TB
                list_TB                 = copy.deepcopy(default_list_TB)
                list_temp_tile_sizes    = copy.deepcopy(default_list_temp_tile_sizes)
                opt_done                = -1

                #
                #   [Constraint #1]
                #
                if list_input_tensor[start_idx] == list_output_tensor[0]:
                    continue

                #
                #   [Constraint #2]
                #
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[start_idx]) != -1:
                    continue

                #
                #   from "start_idx" to |T|.length
                #
                for target_idx in range(start_idx, len(list_input_tensor)):
                    #
                    #   [Constraint #1] If Input[target_idx] == Output[0],
                    #
                    if list_input_tensor[target_idx] == list_output_tensor[0]:
                        continue
                    #
                    #   [Constraint #2] If Intput[target_idx] == Internal Index
                    #
                    if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[target_idx]) != -1:
                        continue

                    #
                    vol_TB *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])
                    #list_TB.append(list_input_tensor[target_idx])

                    #   |TB_X'| >= |TB_X|
                    if vol_TB >= each_size_TB:
                        #   |TB_X'| > |TB_X|
                        if vol_TB > each_size_TB:
                            blocking_tile_size = int(each_size_TB / vol_TB_prev)
                            list_TB.append(list_input_tensor[target_idx])
                            list_temp_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                        #   |TB_X'| = |TB_X|
                        else:
                            list_TB.append(list_input_tensor[target_idx])
                            list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                        #
                        opt_done = 1
                        break
                    #   |TB_X'| < |TB_X|
                    else:
                        list_TB.append(list_input_tensor[target_idx])
                        list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                    #
                    vol_TB_prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])
                    
                #
                #   [2] from "0" to "start_idx"
                #
                if opt_done == -1:
                    for target_idx in range(0, start_idx):
                        #
                        #   [Constraint #1] If Input[target_idx] == Output[0],
                        #
                        if list_input_tensor[target_idx] == list_output_tensor[0]:
                            continue
                        #
                        #   [Constraint #2] If Intput[target_idx] == Internal Index
                        #
                        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[target_idx]) != -1:
                            continue

                        print ("[2]", start_idx, ", target_idx: ", list_input_tensor[target_idx], ", opt_done: ", opt_done)
                        #
                        if opt_done == -1:
                            vol_TB *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])
                            #   |TB_X'| >= |TB_X|
                            if vol_TB >= each_size_TB:
                                print ("[2] >> ", list_input_tensor[target_idx], " is mapped")
                                #   |TB_X'| > |TB_X|
                                if vol_TB > each_size_TB:
                                    blocking_tile_size = int(each_size_TB / vol_TB_prev)
                                    opt_start_mapped = 1
                                    list_TB.append(list_input_tensor[target_idx])
                                    list_temp_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                                #   |TB_X'| = |TB_X|
                                else:
                                    opt_start_mapped = 1
                                    list_TB.append(list_input_tensor[target_idx])
                                    list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                                #
                                opt_done = 1
                                print ("[2] list_TB: ", list_TB)
                                print ("[2] list_temp_tile_sizes: ", list_temp_tile_sizes)
                                break
                            #   |TB_X'| < |TB_X|
                            else:
                                opt_start_mapped = 1
                                list_TB.append(list_input_tensor[target_idx])
                                list_temp_tile_sizes.append([list_input_tensor[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])])
                            #
                            vol_TB_prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])
                #
                #
                #
                if opt_done == 1:
                    list_partial_config_E_L = alg_config_E_L_R(list_input_tensor, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tile_REG, list_TB, list_temp_tile_sizes, opt_print_E_L_R)
                    for each_config in list_partial_config_E_L:
                        if opt_print == 1:
                            print ("[1] each_config: ", each_config)
                        list_partial_config_E.append(each_config)
                #
                #   "TB" is fully mapped at the begining (special case: the output FVI)
                #
        #
        #   |TB| is fully mapped when |Special Case|
        #
        else:
            #
            if opt_print == 1:
                print ("Inputs: Mapping: ", default_list_TB, ", Tile-Sizes: ", default_list_temp_tile_sizes)

            list_partial_config_E_L = alg_config_E_L_R(list_input_tensor, list_output_tensor, list_internal_indices, list_representative_problem_size, list_tile_REG, default_list_TB, default_list_temp_tile_sizes, opt_print_E_L_R)
            #
            for each_config in list_partial_config_E_L:
                if opt_print == 1:
                    print ("[2] each_config: ", each_config)
                list_partial_config_E.append(each_config)
    #
    return list_partial_config_E

#
#   [Configuration][Algorithm][LEFT][Register] --- TB_Y && "REG_Y"
#
def alg_config_E_L_R(list_input_tensor, input_out_tensor, list_internal_indices, list_representative_problem_size, list_tile_REG, list_TB, list_tile_sizes, opt_print):
    #
    if opt_print == 1:
        print ("========================================================================================================================")
        print ("[alg_config_E_L_R] list_TB: ", list_TB)
        print ("[alg_config_E_L_R] list_tile_sizes: ", list_tile_sizes)

    #
    list_partial_config_R = []

    #
    #   Each Tile-Size
    #
    for each_size_REG in list_tile_REG:
        #
        if opt_print == 1:
            print ("========================================================================================================================")
            print ("|REG| = ", each_size_REG)

        #
        #   "start_idx" from "0" to |T|.length
        #
        for start_idx in range(0, len(list_input_tensor)):
            #
            vol_REG                 = 1
            vol_REG_prev            = 1
            list_REG                = []
            list_inherited_TB       = copy.deepcopy(list_TB)
            list_temp_tile_sizes    = copy.deepcopy(list_tile_sizes)
            opt_done                = -1

            #
            if opt_print == 1:
                print ("start_idx: ", start_idx, ", opt_done: ", opt_done)

            #
            if tc_helper.tc_gen_helper_find_1d(list_TB, list_input_tensor[start_idx]) != -1:
                continue
            
            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[start_idx]) != -1:
                continue

            #
            #if start_idx == 0:
            #    continue

            #
            if opt_print == 1:
                print ("(pruned) start_idx: ", start_idx, ", opt_done: ", opt_done)

            #
            #   [1] Check from "start_idx" to |T|.length
            #
            for target_idx in range(start_idx, len(list_input_tensor)):
                #
                if opt_print == 1:
                    print (">1> target_idx: ", target_idx, ", opt_done: ", opt_done)
                
                #
                #   [Check] This index is already mapped on TB
                #
                if tc_helper.tc_gen_helper_find_1d(list_TB, list_input_tensor[target_idx]) != -1:
                    continue

                #
                #   [Check] This index is one of internal indices
                #
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, list_input_tensor[target_idx]) != -1:
                    continue

                #
                #   [Check] This index is the FVI in the Input
                #
                #if target_idx == 0:
                #    continue
                
                #
                #   >>> 
                #
                target_idx_representative_size = tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])
                vol_REG *= target_idx_representative_size

                #   |REG'| >= |REG|
                if vol_REG >= each_size_REG:
                    #   |REG'| > |REG|
                    if vol_REG > each_size_REG:
                        blocking_tile_size = int(each_size_REG / vol_REG_prev)
                        list_REG.append(list_input_tensor[target_idx])
                        list_temp_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                    #   |REG'| = |REG|
                    else:
                        list_REG.append(list_input_tensor[target_idx])
                        list_temp_tile_sizes.append([list_input_tensor[target_idx], target_idx_representative_size])
                    #
                    #   [Done] "REG" is fully mapped, but need to check if there are unmapped indices or not.
                    #
                    opt_done = 1
                    break   # for "target_idx"
                #   |REG'| < |REG|
                else:
                    list_REG.append(list_input_tensor[target_idx])
                    list_temp_tile_sizes.append([list_input_tensor[target_idx], target_idx_representative_size])            
            
            #
            #   Unchecked indices from "0" to "start_idx"
            #
            if opt_done == -1:
                for target_idx in range(0, start_idx):
                    #
                    if opt_print == 1:
                        print (">2> target_idx: ", target_idx, ", opt_done: ", opt_done)
            
                    #
                    target_idx_representative_size = tc_helper.tc_gen_helper_find(list_representative_problem_size, list_input_tensor[target_idx])
                    vol_REG *= target_idx_representative_size

                    #   |REG'| >= |REG|
                    if vol_REG >= each_size_REG:
                        #   |REG'| > |REG|
                        if vol_REG > each_size_REG:
                            blocking_tile_size = int(each_size_REG / vol_REG_prev)
                            opt_start_mapped = 1
                            list_REG.append(list_input_tensor[target_idx])
                            list_temp_tile_sizes.append([list_input_tensor[target_idx], blocking_tile_size])
                        #   |REG'| = |REG|
                        else:
                            opt_start_mapped = 1
                            list_REG.append(list_input_tensor[target_idx])
                            list_temp_tile_sizes.append([list_input_tensor[target_idx], target_idx_representative_size])
                        #
                        #   [Done] "REG" is fully mapped, but need to check if there are unmapped indices or not.
                        #
                        opt_done = 1
                        break   # for "target_idx"
                    #   |REG'| < |REG|
                    else:
                        opt_start_mapped = 1
                        list_REG.append(list_input_tensor[target_idx])
                        list_temp_tile_sizes.append([list_input_tensor[target_idx], target_idx_representative_size])
            #
            #   []
            #
            if opt_done == 1:
                for each_idx in list_input_tensor:
                    #   "each_idx" is mapped on REG
                    if tc_helper.tc_gen_helper_find_1d(list_REG, each_idx) != -1:
                        continue
                    #   "each_idx" is mapped on TB_E
                    if tc_helper.tc_gen_helper_find_1d(list_TB, each_idx) != -1:
                        continue
                    #   "each_idx" is an internal index
                    if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_idx) != -1:
                        continue
                    #
                    list_inherited_TB.append(each_idx)
                    list_temp_tile_sizes.append([each_idx, 1])
                #
                #   [Configuration][Partial] TB_E and REG
                #
                list_partial_config_R.append([list_inherited_TB, list_REG, list_temp_tile_sizes])    
            #
            if opt_print == 1:
                print ("========================================================================================================================")
        #
        #   [Double-Check] good for 06, not 26
        #
        if opt_done == -1:
            opt_fvi_input   = -1
            list_tmp_fvi    = []
            for each_idx in list_input_tensor:
                #   "each_idx" -> REG
                if tc_helper.tc_gen_helper_find_1d(list_REG, each_idx) != -1:
                    continue
                #   "each_idx" -> TB
                if tc_helper.tc_gen_helper_find_1d(list_TB, each_idx) != -1:
                    continue
                #   "each_idx" -> K
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_idx) != -1:
                    continue
                #   "each_idx" == input's FVI
                if each_idx == list_input_tensor[0]:
                    continue
                #
                opt_fvi_input = 1
            #
            if opt_fvi_input == -1:
                #
                #   >>>
                #      
                if tc_helper.tc_gen_helper_find_1d(list_TB, list_input_tensor[0]) == -1:
                    list_REG.append(list_input_tensor[0])
                    list_temp_tile_sizes.append([list_input_tensor[0], each_size_REG])
                    #
                    list_partial_config_R.append([list_inherited_TB, list_REG, list_temp_tile_sizes])
    #
    return list_partial_config_R

#
#   [Configuration][Algorithm][TB_K]
#   opt_print: -1 (off), 0 (basic info.), 1 (basic info. + debug info.)
#
def alg_config_K(list_internal_indices, list_representative_problem_size, list_tiles_TB, opt_print):
    #
    list_partial_config_K = []

    #
    #   Each Tile-Size
    #
    for each_size_TB in list_tiles_TB:
        # 
        for start_idx in range(0, len(list_internal_indices)):
            #
            vol_TB_K                = 1
            vol_TB_K_prev           = 1
            list_TB_K               = []
            list_temp_tile_sizes    = []
            opt_done                = -1

            #
            for target_idx in range(start_idx, len(list_internal_indices)):
                #
                if opt_done == -1:
                    vol_TB_K *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])
                    
                    #   |TB_K'| >= |TB_K|
                    if vol_TB_K >= each_size_TB:
                        #   |TB_K'| > |TB_K|
                        if vol_TB_K > each_size_TB:
                            blocking_tile_size = int(each_size_TB / vol_TB_K_prev)
                            list_TB_K.append(list_internal_indices[target_idx])
                            list_temp_tile_sizes.append([list_internal_indices[target_idx], blocking_tile_size])
                        #   |TB_K'| = |TB_K|
                        else:
                            list_TB_K.append(list_internal_indices[target_idx])
                            list_temp_tile_sizes.append([list_internal_indices[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])])
                        #
                        opt_done = 1
                    #   |TB_K'| < |TB_K|
                    else:
                        list_TB_K.append(list_internal_indices[target_idx])
                        list_temp_tile_sizes.append([list_internal_indices[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])])
                    #
                    vol_TB_K_prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])
                #
                else:
                    list_TB_K.append(list_internal_indices[target_idx])
                    list_temp_tile_sizes.append([list_internal_indices[target_idx], 1])

            #
            for target_idx in range(0, start_idx):
                #
                if opt_done == -1:
                    vol_TB_K *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])

                    #   |TB_K'| >= |TB_K|
                    if vol_TB_K >= each_size_TB:
                        #   |TB_K'| > |TB_K|
                        if vol_TB_K > each_size_TB:
                            blocking_tile_size = each_size_TB / vol_TB_K_prev
                            list_TB_K.append(list_internal_indices[target_idx])
                            list_temp_tile_sizes.append([list_internal_indices[target_idx], int(blocking_tile_size)])
                        #   |TB_K'| = |TB_K|
                        else:
                            list_TB_K.append(list_internal_indices[target_idx])
                            list_temp_tile_sizes.append([list_internal_indices[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])])
                        #
                        opt_done = 1
                    #   |TB_K'| < |TB_K|
                    else:
                        list_TB_K.append(list_internal_indices[target_idx])
                        list_temp_tile_sizes.append([list_internal_indices[target_idx], tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])])
                    #
                    vol_TB_K_prev *= tc_helper.tc_gen_helper_find(list_representative_problem_size, list_internal_indices[target_idx])
                #
                else:
                    list_TB_K.append(list_internal_indices[target_idx])
                    list_temp_tile_sizes.append([list_internal_indices[target_idx], 1])

            #
            if opt_done == 1:
                list_partial_config_K.append([each_size_TB, list_TB_K, list_temp_tile_sizes])    

            #
            if opt_print == 1:
                print ("|TB_K| = ", each_size_TB, ", opt_done = ", opt_done)
                print ("[final result] list_TB_K: ", list_TB_K)
                print ("[final result] list_temp_tile_sizes: ", list_temp_tile_sizes)

    #
    #
    #
    return list_partial_config_K
