import src.generators.tc_helper     as tc_helper

#
#
#
def tc_gen_models_Total_Cost(list_configurations,
                            opt_print):
    #
    #
    #
    if opt_print == 1:
        print ("========================================= [Model][Total Cost] =========================================")

    #
    if opt_print == 1:
        #
        #   All Configurationss
        #   
        for each_config in list_configurations:
            each_config.print_configuration()

    #
    #   [1]
    #   
    for each_configuration in list_configurations:
        #
        #
        #
        tc_gen_models_GMEM(each_configuration, opt_print)

    if opt_print == 1:
        print ("========================================= [Model][Total Cost] =========================================")


#
#   Model based on DRAM Data-Movements
#
def tc_gen_models_GMEM(each_configuration, opt_print):
    #
    #
    #
    if opt_print == 1:
        print ("========================================= [Model][GMEM Load Inputs] =========================================")

    #
    numElements_Double  = int(128 / 8)
    size_TB_K           = 1
    #
    #   For Internal Indicies,
    #
    if len(each_configuration.list_TB_K) == 1:
        size_TB_K = tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_configuration.list_TB_K[0])
    else:
        size_TB_K = 1

    #
    #   Check Types of Input such as [E_K, ...] or [E_A, ...]
    #
    opt_load_A_ext = -1     # -1: FVI = internal
    opt_load_B_ext = -1     #  1: FVI = external
    if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_B, each_configuration.list_tensor_A[0]) == -1:
        opt_load_A_ext = 1
    
    if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_A, each_configuration.list_tensor_B[0]) == -1:
        opt_load_B_ext = 1

    #
    #   Initial Values
    #
    size_continuous_elements_A = 1
    size_continuous_elements_B = 1
    size_continuous_elements_C = 1

    #
    #   Based on the Representative Problem Size
    #
    size_tiles_A = 1
    size_tiles_B = 1
    size_tiles_C = 1

    #
    #   Input: A (Continuous)
    #
    is_continuous = 1
    #each_configuration.print_tensor_A()
    for each_idx in each_configuration.list_tensor_A:
        #
        #   Starts from External Index
        #
        if opt_load_A_ext == 1:
            #
            #   Internal
            #
            if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_B, each_idx) != -1:
                break
            #
            #   External
            #
            else:
                #
                #   Need to Check if This Index is Continuous Or NOT.
                #
                if is_continuous == 1:
                    size_continuous_elements_A *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
                    #
                    #
                    #
                    if tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx) != tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_idx):
                        is_continuous = -1
                else:
                    break            
        #
        #   Starts from Internal Index
        #
        else:
            #
            #   External
            #
            if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_B, each_idx) == -1:
                break
            #
            #   Internal
            #
            else:
                #
                #
                #
                if is_continuous == 1:
                    size_continuous_elements_A *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
                    #
                    #
                    #
                    if tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx) != tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_idx):
                        is_continuous = -1
                else:
                    break
        #
        #
        #

    #
    #   Input: A (TB and REG)
    #
    size_A_E_TB     = 1
    size_A_K_TB     = 1
    size_A_E_REG    = 1
    for each_idx in each_configuration.list_tensor_A:
        #   External Index
        if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_B, each_idx) == -1:
            #   TB
            if tc_helper.tc_gen_helper_find_1d(each_configuration.list_REG_X, each_idx) == -1 and tc_helper.tc_gen_helper_find_1d(each_configuration.list_REG_Y, each_idx) == -1:
                size_A_E_TB *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
            #   REG
            else:
                size_A_E_REG *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
        else:
            size_A_K_TB *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
    
    #
    #   TB_X -> External Index && TB_Y -> Internal Index
    #
    if opt_load_A_ext == 1:
        times_inner_TB_X    = size_A_E_TB / each_configuration.size_TB_X
        times_inner_TB_Y    = size_A_K_TB / each_configuration.size_TB_Y
    #
    #   TB_X -> Internal Index && TB_Y -> External Index
    #
    else:
        times_inner_TB_X    = size_A_K_TB / each_configuration.size_TB_X
        times_inner_TB_Y    = size_A_E_TB / each_configuration.size_TB_Y

    #
    #   To Calculate The Cost of Loading Input Tensor per a Thread Block
    #
    trans_Row_TB_X = min(size_continuous_elements_A, each_configuration.size_TB_X) / each_configuration.size_TB_X
    cost_TB_load_A = trans_Row_TB_X * each_configuration.size_TB_Y * size_A_E_REG * times_inner_TB_X * times_inner_TB_Y
    #print (">>> # of Transactions per Step: ", cost_TB_load_A)

    #
    #   Input: B
    #
    is_continuous = 1
    #each_configuration.print_tensor_B()
    for each_idx in each_configuration.list_tensor_B:
        #
        #
        #
        if opt_load_B_ext == 1:
            #
            #   Internal
            #
            if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_A, each_idx) != -1:
                break
            #
            #   External
            #
            else:
                #
                #
                #
                if is_continuous == 1:
                    size_continuous_elements_B *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
                    #
                    #
                    #
                    if tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx) != tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_idx):
                        is_continuous = -1
                else:
                    break
        else:
            #
            #   External
            #
            if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_A, each_idx) == -1:
                break
            #
            #   Internal
            #
            else:
                #
                #
                #
                if is_continuous == 1:
                    size_continuous_elements_B *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
                    #
                    #
                    #
                    if tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx) != tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_idx):
                        is_continuous = -1
                else:
                    break
        #
        #
        #
    
    #
    #   Input: B (TB and REG)
    #
    size_B_E_TB     = 1
    size_B_K_TB     = 1
    size_B_E_REG    = 1
    for each_idx in each_configuration.list_tensor_B:
        #
        if tc_helper.tc_gen_helper_find_1d(each_configuration.list_tensor_A, each_idx) == -1:
            #   TB
            if tc_helper.tc_gen_helper_find_1d(each_configuration.list_REG_X, each_idx) == -1 and tc_helper.tc_gen_helper_find_1d(each_configuration.list_REG_Y, each_idx) == -1:
                size_B_E_TB *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
            #   REG
            else:
                size_B_E_REG *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
        else:
            size_B_K_TB *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)

    #
    #   TB_X -> External Index && TB_Y -> Internal Index
    #
    if opt_load_B_ext == 1:
        times_inner_TB_X    = size_B_E_TB / each_configuration.size_TB_X
        times_inner_TB_Y    = size_B_K_TB / each_configuration.size_TB_Y
    #
    #   TB_X -> Internal Index && TB_Y -> External Index
    #
    else:
        times_inner_TB_X    = size_B_K_TB / each_configuration.size_TB_X
        times_inner_TB_Y    = size_B_E_TB / each_configuration.size_TB_Y
    
    #
    #
    #
    trans_Row_TB_X = min(size_continuous_elements_B, each_configuration.size_TB_X) / each_configuration.size_TB_X
    cost_TB_load_B = trans_Row_TB_X * each_configuration.size_TB_Y * size_A_E_REG * times_inner_TB_X * times_inner_TB_Y
    #print (">>> # of Transactions per Step: ", cost_TB_load_B)

    #
    #   Output: C
    #
    is_continuous = 1
    #each_configuration.print_tensor_C()
    for each_idx in each_configuration.list_tensor_C:
        #
        #   Index mapped on TB
        #
        if tc_helper.tc_gen_helper_find_1d(each_configuration.list_REG_X, each_idx) == -1 and tc_helper.tc_gen_helper_find_1d(each_configuration.list_REG_Y, each_idx) == -1:
            if is_continuous == 1:
                size_continuous_elements_C *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
                if tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx) != tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_idx):
                    is_continuous = -1
            else:
                break
        #
        #   Index mapped on REG
        #
        else:
            break

    #
    #
    #
    size_continuous_elements_C_based_TB_X = 1
    for idx_count in range(0, len(each_configuration.list_TB_X)):
        if each_configuration.list_TB_X[idx_count] == each_configuration.list_tensor_C[idx_count]:
            size_continuous_elements_C_based_TB_X *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_configuration.list_TB_X[idx_count])
        else:
            break
    
    size_Output_TB = 1
    for each_idx in each_configuration.list_tensor_C:
        size_Output_TB *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)
    
    cost_TB_store_C = size_Output_TB / size_continuous_elements_C_based_TB_X
    #print (">>> # of Transactions for Output: ", cost_TB_store_C)

    #
    #   The # of Thread Blocks
    #
    num_TBs = 1
    for each_idx in each_configuration.list_tensor_C:
        num_TBs *= tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_idx) / tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)

    #print (">>> # of TBs: ", num_TBs)

    '''
    self.cost_total         = 0
    self.cost_load_input    = 0
    self.cost_load_output   = 0
    self.cost_store_output  = 0
    '''
    each_configuration.cost_load_input      = (cost_TB_load_A + cost_TB_load_B) * num_TBs
    each_configuration.cost_store_output    = cost_TB_store_C * num_TBs
    each_configuration.cost_load_output     = cost_TB_store_C * num_TBs
    each_configuration.cost_total           = each_configuration.cost_load_input + each_configuration.cost_store_output + each_configuration.cost_load_output

    #
    #print ("Total Cost: ", each_configuration.cost_total)

    #
    #
    #
    if opt_print == 1:
        print ("=============================================================================================================")

    #
    return 1
