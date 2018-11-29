#
#
#
import copy
import sys

import src.generators.tc_helper                 as tc_helper
import src.algs.bases.enumeration               as alg_configurations
import src.algs.bases.cost_models               as cost_model
import src.algs.bases.tc_gen_models             as tc_gen_models
import src.inputs.tccg_problem_size             as tccg_problem_size
import src.algs.models.predictive_modeling      as prediction_model


def get_configurations(l_outer_groups, list_configurations_outer_group, tmp_count, tmp_config, opt_print, opt_data_type):
    print ("=========================== [Configurations] ===============================")
    #print (" # of Outer-Groups: ", len(l_outer_groups)) # Assumped that there is only a tensor contraction (PACT 2018)
                                                        # 1 Outer-Group has 1 Tensor Contraction which will be grouped 
                                                        # by 1 Inner-Group
    #
    #   Initial Value (Default)
    #
    list_representative_problem_size = list()
    
    #
    #   Assumption: There is only one Outer-Group and there is only one Tensor Contraction in the Outer-Group.
    #
    list_indices = l_outer_groups[0][2]
    
    
    #
    #   For TTCG Benchmark: representative problem sizes: all-16
    #
    #list_tccg_representative_problem_size = tccg_problem_size.get_tccg_representative_problem_sizes(tmp_count)
    list_tccg_representative_problem_size = l_outer_groups[0][1][0][9]

    #
    #
    #
    if len(list_indices) != len(list_tccg_representative_problem_size):
        print ("list_representative_problem_size from TCCG Benchmark: ", list_tccg_representative_problem_size)
        print ("len(list_indices): ", len(list_indices), " vs len(list_tccg_representative_problem_size): ", len(list_tccg_representative_problem_size))
        for idx_count in range(0, len(list_indices)):
            list_representative_problem_size.append([list_indices[idx_count], 16])
        #print ("[ERROR] src.generators.configurations.get_configurations()")
        #sys.exit()
    else:
        for idx_count in range(0, len(list_indices)):
            list_representative_problem_size.append([list_indices[idx_count], list_tccg_representative_problem_size[idx_count]])
            
    print (" (TCCG) Representative Problem Size--- ", tmp_count)
    print (" : ", list_representative_problem_size)

    #
    #   [Result] List of Configurations
    #
    list_configurations_temp        = list()
    
    #
    #   Per Each-Outer-Group
    #
    idx_outer_count = 1
    for each_outer_group in l_outer_groups:
        print (" > Outer-Group #. ", idx_outer_count)
        
        #
        base_outer_group    = each_outer_group[0]
        list_tc             = each_outer_group[1]
        all_indices         = each_outer_group[2]
        list_info_split     = list()

        #
        #   For Each Tensor Contraction
        #
        idx_tc_count = 1
        for each_tc in list_tc:
            print (" >> Tensor-Contraction [", idx_tc_count, "] ")
            print (" :", each_tc) 

            #
            #   Default:    Input(Left) ---> TB_X and REG_X
            #               But, if Input(Right) has the FVI in Output, Input(Right) ---> TB_X and REG_X
            #
            list_output_tensor      = each_tc[1]
            list_internal_indices   = each_tc[3]
            list_input_tensor_left  = each_tc[5]
            list_input_tensor_right = each_tc[7]
            list_info_idx_split     = []

            #
            #   
            #
            num_ext_left    = len(list_input_tensor_left) - len(list_internal_indices)
            num_ext_right   = len(list_input_tensor_right) - len(list_internal_indices)

            #
            #   opt_limited_split (0: free, 1: limited)
            #
            opt_limited_split = 1
            if num_ext_left == 1 or num_ext_right == 1:
                print ("[Code Generator][Configurations] One of Input Tensors has only one external index, resulting in splitting freely.")
                #
                #   Tensor (Left)
                #
                if num_ext_left == 1:
                    print ("(L) To Split First: ", list_input_tensor_left)
                    #
                    #   To Find a Target Index in the Tensor
                    #
                    idx_count = 0
                    prev_idx = ""
                    for each_idx in list_input_tensor_left:
                        #
                        #   Extenel Indices (|External Indices| == 1)
                        #
                        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_idx) == -1:
                            prev_idx = each_idx
                            each_tc[5].insert(idx_count,        each_idx + "1")
                            each_tc[5].insert(idx_count + 1,    each_idx + "2")
                            each_tc[5].pop(idx_count + 2)
                            break
                        #
                        idx_count += 1
                    #
                    #   To Modify the Output Tensor
                    #
                    idx_count = 0
                    for each_idx in list_output_tensor:
                        if each_idx == prev_idx:
                            each_tc[1].insert(idx_count,        each_idx + "1")
                            each_tc[1].insert(idx_count + 1,    each_idx + "2")
                            each_tc[1].pop(idx_count + 2)
                            list_info_split.append([each_idx, each_idx + "1", each_idx + "2"])
                            list_info_idx_split.append([each_idx, each_idx + "1", each_idx + "2"])
                            break
                        #
                        idx_count += 1
                    #
                    #   To Modify the Representative Problem Size
                    #
                    idx_count = 0
                    for each_element in list_representative_problem_size:
                        #
                        if each_element[0] == prev_idx:
                            list_representative_problem_size.insert(idx_count,      [each_element[0] + "1", each_element[1]])
                            list_representative_problem_size.insert(idx_count + 1,  [each_element[0] + "2", each_element[1]])
                            break
                        #
                        idx_count += 1
                    #
                    #   [Outer-Group] Assumption: Only One Tensor Contraction
                    #
                    idx_count = 0
                    for each_idx in all_indices:
                        #
                        if each_idx == prev_idx:
                            all_indices.insert(idx_count,       each_idx + "1")
                            all_indices.insert(idx_count + 1,   each_idx + "2")
                            all_indices.pop(idx_count + 2)
                            break
                        #
                        idx_count += 1

                #
                #   Tensor (Right)
                #
                if num_ext_right == 1:
                    print ("(R) To Split First: ", list_input_tensor_right)
                    #
                    #   To Find a Target Index in the Tensor
                    #
                    idx_count   = 0
                    prev_idx    = ""
                    for each_idx in list_input_tensor_right:
                        #
                        #   External Indices (|External Indices| == 1)
                        #
                        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_idx) == -1:
                            prev_idx = each_idx
                            each_tc[7].insert(idx_count,        each_idx + "1")
                            each_tc[7].insert(idx_count + 1,    each_idx + "2")
                            each_tc[7].pop(idx_count + 2)
                            break
                        #
                        idx_count += 1
                    
                    #
                    #   To Modify the Output Tensor
                    #
                    idx_count = 0
                    for each_idx in list_output_tensor:
                        if each_idx == prev_idx:
                            each_tc[1].insert(idx_count,        each_idx + "1")
                            each_tc[1].insert(idx_count + 1,    each_idx + "2")
                            each_tc[1].pop(idx_count + 2)
                            list_info_split.append([each_idx, each_idx + "1", each_idx + "2"])
                            list_info_idx_split.append([each_idx, each_idx + "1", each_idx + "2"])
                            break
                        #
                        idx_count += 1
                    #
                    #   To Modify the Representative Problem Size
                    #
                    idx_count = 0
                    for each_element in list_representative_problem_size:
                        #
                        if each_element[0] == prev_idx:
                            list_representative_problem_size.insert(idx_count,      [each_element[0] + "1", each_element[1]])
                            list_representative_problem_size.insert(idx_count + 1,  [each_element[0] + "2", each_element[1]])
                            break
                        #
                        idx_count += 1
                    #
                    #   [Outer-Group] Assumption: Only One Tensor Contraction
                    #
                    idx_count = 0
                    for each_idx in all_indices:
                        #
                        if each_idx == prev_idx:
                            all_indices.insert(idx_count,       each_idx + "1")
                            all_indices.insert(idx_count + 1,   each_idx + "2")
                            all_indices.pop(idx_count + 2)
                            break
                        #
                        idx_count += 1
                    
                #
                #   This Option (opt_limited_split) is related to "Interface"
                #
                opt_limited_split = 0
            else:
                print ("[Code Generator][Configurations] Both Input Tensors have at lease two external indices, resulting in splitting exclusively.")

            #
            #   Input:  an Equation for a Tensor Contraction
            #           a Representative Problem Size
            #   Output: a List of Configurations 
            #
            
            #
            #   New Mapping Algorithms
            #
            list_temp = alg_configurations.alg_enumeration_pruning(each_tc, list_info_idx_split, list_representative_problem_size, opt_limited_split, 0, opt_data_type)
            print ("[Code Generator][Configurations] configurations: # of Configurations--- Total: ", len(list_temp))

            #
            if len(list_temp) < 1:
                print ("[Code Generator][Configurations] ERROR: Problem(s) in Enumerating Configurations")
                sys.exit()

            #
            #   Models: each configuration has its own cost.
            #
            cost_model.cost_model_total(list_temp, 0)
            prediction_model.model_predictive_modeling(list_temp)

            #
            #list_temp[0].print_configuration(0)
            #
            list_temp.sort(key = lambda x: x.cost_total)

            #
            #   All Configurationss
            #
            idx_configuration   = 0
            min_cost            = 1000000000000
            min_steps           = 1000000000000
            idx_count = 0
            for each_config in list_temp:
                if min_cost > each_config.cost_total:
                    min_cost    = each_config.cost_total
                    min_steps   = each_config.steps_main_loops
                    idx_configuration = idx_count
                
                if min_cost == each_config.cost_total:
                    if min_steps > each_config.steps_main_loops:
                        min_steps = each_config.steps_main_loops
                        idx_configuration = idx_count
                #
                idx_count += 1
            #
            print ("[Code Generator][Configurations] # ", idx_configuration, " in ", len(list_temp))
            
            #
            if tmp_config < len(list_temp) and tmp_config != -1:
                print ("[Code Generator][Configurations] manually picked # ", tmp_config)
                list_configurations_outer_group.append(list_temp[tmp_config])
            else:
                list_configurations_outer_group.append(list_temp[idx_configuration])

            #
            #
            #
            #for rank in range(0, len(list_temp)):
            #    list_temp[rank].print_configuration(0,str(rank))
            
        #
        each_outer_group.append(list_info_split)
        idx_outer_count = idx_outer_count + 1
    #
    #    
    #
    print ("============================================================================")
