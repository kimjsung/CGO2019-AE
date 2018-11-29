#
#
#
import math
import src.generators.tc_helper     as tc_helper
import src.generators.helper_base   as helper_base

#
#
#
def cost_model_total(list_configurations, opt_print):
    #
    #
    #
    if opt_print == 1:
        print ("========================================= [Model][Total Cost] =========================================")

    #
    #   Print All Configurationss
    #   
    if opt_print == 2:
        for each_config in list_configurations:
            each_config.print_configuration()

    #
    #   [1]
    #   
    idx_count = 0
    for each_configuration in list_configurations:
        #
        #
        #
        list_comb = tc_gen_models_TBs(each_configuration, idx_count, 0)
        tc_gen_models_GMEM(each_configuration, list_comb, idx_count, 0)
        tc_gen_models_Kernels(each_configuration, idx_count, 0)
        tc_gen_models_Computes(each_configuration, idx_count, 0)

        #
        idx_count += 1
    #
    if opt_print == 1:
        print ("========================================= [Model][Total Cost] =========================================")

#
#
#
def tc_gen_models_TBs(each_configuration, idx_count, opt_print=0):
    #
    #
    #
    if opt_print == 1:
        print ("===[", idx_count, "]====================================== [Model][GMEM Load Inputs] =========================================")
        print (" tile-sizes: ", each_configuration.list_tile_sizes)
        print (" repr-sizes: ", each_configuration.list_representative_problem_size)
        print (" split-info: ", each_configuration.list_splits)

    #
    list_possible_comb_splits = []

    #
    for each_split in each_configuration.list_splits:
        #
        idx_base            = each_split[0]
        idx_first           = each_split[1]
        idx_second          = each_split[2]
        
        idx_base_repre_size = tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, idx_base)
        list_possible_cases = []

        #
        if opt_print == 2:
            print (idx_base, " --> both ", idx_first, " &", idx_second)
        
        #
        for denominiator in range(1, idx_base_repre_size + 1):
            if idx_base_repre_size % denominiator == 0:
                #
                list_possible_cases.append([[idx_base, idx_base_repre_size], [idx_first, int(idx_base_repre_size / denominiator)], [idx_second, denominiator]])
        #
        list_possible_comb_splits.append(list_possible_cases)
    
    #
    if opt_print == 2:
        for each_split in list_possible_comb_splits:
            print ("len(each_split): ", len(each_split))
            for each_comb in each_split:
                print (" >> ", each_comb)
    
    #
    #   External Indices: related to # of TBs, and Full-Tiles for External Indices
    #
    opt_full_ext                                = True
    opt_full_int                                = True
    list_possible_representative_problem_sizes  = []

    #
    #   [Assumption] len(list_possible_comb_splits) == 1 or 2.
    #
    if len(list_possible_comb_splits) == 1:
        for each_comb in list_possible_comb_splits[0]:
            tmp_list    = []
            tmp_num_TBs = 1

            #
            tmp_list.append(each_comb[1])
            tmp_num_TBs *= math.ceil(each_comb[1][1] / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_comb[1][0]))

            tmp_list.append(each_comb[2])
            tmp_num_TBs *= math.ceil(each_comb[2][1] / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_comb[2][0]))

            #
            for each_ext_idx in each_configuration.list_tensor_C:
                if helper_base.helper_base_find_list_2D(each_comb, each_ext_idx) == -1:
                    tmp_list.append([each_ext_idx, helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_ext_idx)])
                    tmp_num_TBs *= math.ceil(helper_base.helper_base_find_list_2D(each_configuration.list_representative_problem_size, each_ext_idx) / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_ext_idx))
                    
            list_possible_representative_problem_sizes.append([tmp_list, tmp_num_TBs])
        #
        tmp_min_num_TBs = 10000000000000000
        min_idx         = 0
        idx_count       = 0
        for each_comb in list_possible_representative_problem_sizes:
            if tmp_min_num_TBs > each_comb[1]:
                tmp_min_num_TBs = each_comb[1]
                min_idx         = idx_count
            #
            idx_count += 1
        #
        each_configuration.add_split_representative_problem_size(list_possible_representative_problem_sizes[min_idx][0])
        each_configuration.num_TBs = list_possible_representative_problem_sizes[min_idx][1]
        return list_possible_representative_problem_sizes[min_idx]
    #
    elif len(list_possible_comb_splits) == 2:
        for each_comb_out in list_possible_comb_splits[0]:
            for each_comb_in in list_possible_comb_splits[1]:
                tmp_list    = []
                tmp_num_TBs = 1

                #
                tmp_list.append(each_comb_out[1])
                tmp_num_TBs *= math.ceil(each_comb_out[1][1] / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_comb_out[1][0]))

                tmp_list.append(each_comb_out[2])
                tmp_num_TBs *= math.ceil(each_comb_out[2][1] / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_comb_out[2][0]))

                #
                tmp_list.append(each_comb_in[1])
                tmp_num_TBs *= math.ceil(each_comb_in[1][1] / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_comb_in[1][0]))

                tmp_list.append(each_comb_in[2])
                tmp_num_TBs *= math.ceil(each_comb_in[2][1] / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_comb_in[2][0]))

                #
                for each_ext_idx in each_configuration.list_tensor_C:
                    if helper_base.helper_base_find_list_2D(each_comb_out, each_ext_idx) == -1 and helper_base.helper_base_find_list_2D(each_comb_in, each_ext_idx) == -1:
                        tmp_list.append([each_ext_idx, helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_ext_idx)])
                        tmp_num_TBs *= math.ceil(helper_base.helper_base_find_list_2D(each_configuration.list_representative_problem_size, each_ext_idx) / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_ext_idx))
                        
                #
                list_possible_representative_problem_sizes.append([tmp_list, tmp_num_TBs])
        #
        tmp_min_num_TBs = 1000000000000000000
        min_idx         = 0
        idx_count       = 0
        for each_comb in list_possible_representative_problem_sizes:
            if tmp_min_num_TBs > each_comb[1]:
                tmp_min_num_TBs = each_comb[1]
                min_idx         = idx_count
            #
            idx_count += 1
        #print ("[2] min. comb: ", list_possible_representative_problem_sizes[min_idx])
        each_configuration.add_split_representative_problem_size(list_possible_representative_problem_sizes[min_idx][0])
        each_configuration.num_TBs = list_possible_representative_problem_sizes[min_idx][1]
        return list_possible_representative_problem_sizes[min_idx]
    #
    else:
        list_possible_representative_problem_sizes = each_configuration.list_representative_problem_size

        tmp_num_TBs = 1
        for each_ext_idx in each_configuration.list_tensor_C:
            tmp_num_TBs *= math.ceil(helper_base.helper_base_find_list_2D(each_configuration.list_representative_problem_size, each_ext_idx) / helper_base.helper_base_find_list_2D(each_configuration.list_tile_sizes, each_ext_idx))
        
        #print ("[3] min. comb: ", tmp_num_TBs)
        each_configuration.add_split_representative_problem_size(each_configuration.list_representative_problem_size)
        each_configuration.num_TBs = tmp_num_TBs
        return [each_configuration.list_representative_problem_size, tmp_num_TBs]

#
#   Model based on DRAM Data-Movements
#
def tc_gen_models_GMEM(each_configuration, list_comb, idx_count, opt_print=0):
    #
    #
    #
    if opt_print == 1:
        print ("===[", idx_count, "]====================================== [Model][GMEM Load Inputs] =========================================")
        print (" mappings: TB_X <- ", each_configuration.list_TB_X, ", TB_Y <- ", each_configuration.list_TB_Y)
        print ("         : TB_K <- ", each_configuration.list_TB_K)
        print ("         : REG_X <- ", each_configuration.list_REG_X, ", REG_Y <-", each_configuration.list_REG_Y)
        print (" list_comb: ", list_comb)
        print (" tile-sizes: ", each_configuration.list_tile_sizes)

    #
    numElements_Double  = int(128 / 8)
    #print ("numElements_Double: ", numElements_Double)
    
    #
    #   For Internal Indicies,
    #
    size_TB_K   = 1
    size_N_K    = 1
    for each_int_idx in each_configuration.list_TB_K:
        size_TB_K   *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_int_idx)
        size_N_K    *= tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, each_int_idx)
        
    #
    #   # of "main" loop (calculated by N_K / T_K)
    #
    steps_main_loops = size_N_K / size_TB_K

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
    if opt_print == 1:
        print ("-1: FVI = internal, 1: FVI = external")
        print ("opt_load_A_ext: ", opt_load_A_ext, ", opt_load_B_ext: ", opt_load_B_ext)

    #
    #   Input: A (Continuous)
    #
    is_continuous = 1
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
    #print ("[A] is_continuous: ", is_continuous, ", size_continuous_elements_A: ", size_continuous_elements_A)


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
    if opt_print == 1:
        print ("|SMEM_A| = ", size_A_E_TB * size_A_E_REG * size_A_K_TB, ", (", size_A_E_REG, " * ", size_A_E_TB, " * ", size_A_K_TB, ")")
        print ("|TB_X| = ", each_configuration.size_TB_X, ", |TB_Y| = ", each_configuration.size_TB_Y)
    #
    #   TB_X -> External Index && TB_Y -> Internal Index
    #
    if opt_load_A_ext == 1:
        times_inner_TB_X    = max(size_A_E_TB / each_configuration.size_TB_X, 1.0)
        times_inner_TB_Y    = max(size_A_K_TB / each_configuration.size_TB_Y, 1.0)
    #
    #   TB_X -> Internal Index && TB_Y -> External Index
    #
    else:
        times_inner_TB_X    = max(size_A_K_TB / each_configuration.size_TB_X, 1.0)
        times_inner_TB_Y    = max(size_A_E_TB / each_configuration.size_TB_Y, 1.0)
    #
    if opt_print == 1:
        print ("A: :times_inner_TB_X: ", times_inner_TB_X, ", times_inner_TB_Y: ", times_inner_TB_Y)

    #
    #   Based on a row along TB_X, can the number of continuous elements be loaded concurrently?
    #
    if opt_load_A_ext == -1:    # TB_X -> K
        size_TB_X = min(size_A_K_TB, each_configuration.size_TB_X)
        size_TB_Y = min(size_A_E_TB, each_configuration.size_TB_Y)
    else:                       # TB_X -> E
        size_TB_X = min(size_A_E_TB, each_configuration.size_TB_X)
        size_TB_Y = min(size_A_K_TB, each_configuration.size_TB_Y)

    #
    #
    #
    estimated_DRAM_transaction_per_TB_X                     = size_TB_X / min(size_continuous_elements_A, size_TB_X)
    estimated_DRAM_transaction_per_TB                       = estimated_DRAM_transaction_per_TB_X * size_TB_Y
    estimated_DRAM_transaction_per_TB_inner_loops           = estimated_DRAM_transaction_per_TB * times_inner_TB_X * times_inner_TB_Y
    estimated_DRAM_transaction_per_TB_inner_loops_reg       = estimated_DRAM_transaction_per_TB_inner_loops * size_A_E_REG
    estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K   = estimated_DRAM_transaction_per_TB_inner_loops_reg * steps_main_loops

    if opt_print == 1:
        print ("size_TB_X: ", size_TB_X, ", size_TB_Y: ", size_TB_Y, ", size_TB_K: ", size_TB_K)
        print ("estimated_DRAM_transactions_per_TB_X (should be fixed): ", estimated_DRAM_transaction_per_TB_X)
        print ("estimated_DRAM_transactions_per_TB: ", estimated_DRAM_transaction_per_TB)
        print ("estimated_DRAM_transaction_per_TB_inner_loops: ", estimated_DRAM_transaction_per_TB_inner_loops)
        print ("estimated_DRAM_transaction_per_TB_inner_loops_reg: ", estimated_DRAM_transaction_per_TB_inner_loops_reg)
        print ("estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K: ", estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K)

    #
    #   To Calculate The Cost of Loading Input Tensor per a Thread Block
    #
    cost_TB_load_A = estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K
    
    #
    #   Input: B
    #
    is_continuous = 1
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
    if opt_print == 1:
        print ("size_continuous_elements_B: ", size_continuous_elements_B)
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
        times_inner_TB_X    = max(size_B_E_TB / each_configuration.size_TB_X, 1.0)
        times_inner_TB_Y    = max(size_B_K_TB / each_configuration.size_TB_Y, 1.0)
    #
    #   TB_X -> Internal Index && TB_Y -> External Index
    #
    else:
        times_inner_TB_X    = max(size_B_K_TB / each_configuration.size_TB_X, 1.0)
        times_inner_TB_Y    = max(size_B_E_TB / each_configuration.size_TB_Y, 1.0)
    
    #
    if opt_print == 1:
        print ("B: :times_inner_TB_X: ", times_inner_TB_X, ", times_inner_TB_Y: ", times_inner_TB_Y)

    #
    #   Based on a row along TB_X, can the number of continuous elements be loaded concurrently?
    #
    if opt_load_A_ext == -1:    # TB_X -> K
        size_TB_X = min(size_B_K_TB, each_configuration.size_TB_X)
        size_TB_Y = min(size_B_E_TB, each_configuration.size_TB_Y)
    else:                       # TB_X -> E
        size_TB_X = min(size_B_E_TB, each_configuration.size_TB_X)
        size_TB_Y = min(size_B_K_TB, each_configuration.size_TB_Y)
   
    #
    #
    #
    estimated_DRAM_transaction_per_TB_X                     = size_TB_X / min(size_continuous_elements_B, size_TB_X)
    estimated_DRAM_transaction_per_TB                       = estimated_DRAM_transaction_per_TB_X * size_TB_Y
    estimated_DRAM_transaction_per_TB_inner_loops           = estimated_DRAM_transaction_per_TB * times_inner_TB_X * times_inner_TB_Y
    #estimated_DRAM_transaction_per_TB_inner_loops_reg       = estimated_DRAM_transaction_per_TB_inner_loops * size_A_E_REG
    estimated_DRAM_transaction_per_TB_inner_loops_reg       = estimated_DRAM_transaction_per_TB_inner_loops * size_B_E_REG
    estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K   = estimated_DRAM_transaction_per_TB_inner_loops_reg * steps_main_loops

    if opt_print == 1:
        print ("size_TB_X: ", size_TB_X, ", size_TB_Y: ", size_TB_Y, ", size_TB_K: ", size_TB_K)
        print ("estimated_DRAM_transactions_per_TB_X (should be fixed): ", estimated_DRAM_transaction_per_TB_X)
        print ("estimated_DRAM_transactions_per_TB: ", estimated_DRAM_transaction_per_TB)
        print ("estimated_DRAM_transaction_per_TB_inner_loops: ", estimated_DRAM_transaction_per_TB_inner_loops)
        print ("estimated_DRAM_transaction_per_TB_inner_loops_reg: ", estimated_DRAM_transaction_per_TB_inner_loops_reg)
        print ("estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K: ", estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K)

    #
    #
    #
    cost_TB_load_B = estimated_DRAM_transaction_per_TB_inner_loops_reg_N_K
    
    #
    #   Output: C
    #
    is_continuous = 1
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

    #
    cost_TB_store_C = size_Output_TB / size_continuous_elements_C_based_TB_X

    #
    #   The # of Thread Blocks
    #
    num_TBs = list_comb[1]

    if opt_print == 1:
        print (">>> # of TBs: ", num_TBs)

    #
    each_configuration.cost_load_input      = (cost_TB_load_A + cost_TB_load_B) * num_TBs
    each_configuration.cost_store_output    = cost_TB_store_C * num_TBs

    #each_configuration.cost_load_output     = 0
    each_configuration.cost_load_output     = cost_TB_store_C * num_TBs
    each_configuration.cost_total           = each_configuration.cost_load_input + each_configuration.cost_store_output + each_configuration.cost_load_output
    each_configuration.steps_main_loops     = steps_main_loops

    #
    if opt_print == 1:
        print ("Cost Input (Load): ",   each_configuration.cost_load_input)
        print ("Cost Output (Store): ", each_configuration.cost_store_output)
        print ("Cost Output (Load): ",  each_configuration.cost_load_output)
        print ("Total Cost: ",          each_configuration.cost_total, ", # of steps for main-loop: ", each_configuration.steps_main_loops)

    #
    #
    #
    if opt_print == 1:
        print ("=============================================================================================================")

    #
    return 1

#
#
#
def tc_gen_models_Kernels(each_configuration, idx_count, opt_print=0):
    #
    #
    #
    #
    if opt_print == 1:
        print ("each_config.representative_problem_size: ", each_configuration.list_representative_problem_size)
        print ("each_config.tile_sizes: ", each_configuration.list_tile_sizes)
    
    #
    opt_full_ext = True
    opt_full_int = True

    for each_idx_tile in each_configuration.list_tile_sizes:
        #
        idx_name = each_idx_tile[0]
        idx_tile = each_idx_tile[1]

        #
        if tc_helper.tc_gen_helper_find_1d(each_configuration.list_TB_K, idx_name) != -1:
            if tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, idx_name) % idx_tile != 0:
                opt_full_int = False               
        else:
            if tc_helper.tc_gen_helper_find(each_configuration.list_representative_problem_size, idx_name) % idx_tile != 0:
                opt_full_ext = False

    #
    if opt_print == 1:
        print (">>> opt_full_int: ", opt_full_int, ", opt_full_ext: ", opt_full_ext)
    #
    each_configuration.kernel_full_ext = opt_full_ext
    each_configuration.kernel_full_int = opt_full_int

#
#
#
def tc_gen_models_Computes(each_configuration, idx_count, opt_print=0):
    #
    #
    #
    if opt_print == 1:
        print ("[Model][Computes]")

    size_REG_X = 1
    size_REG_Y = 1
    for each_idx in each_configuration.list_REG_X:
        size_REG_X *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)

    for each_idx in each_configuration.list_REG_Y:
        size_REG_Y *= tc_helper.tc_gen_helper_find(each_configuration.list_tile_sizes, each_idx)

    #
    each_configuration.kernel_arithmetic_intensity = (size_REG_X * size_REG_Y) / (size_REG_X + size_REG_Y)
