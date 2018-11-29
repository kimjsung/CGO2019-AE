import copy
import sys
#
import src.algs.bases.tc_gen_models             as tc_gen_models
import src.generators.tc_helper                 as tc_helper
#
import src.algs.bases.tc_gen_perms_exclusive    as tc_gen_perm_exclusive
import src.algs.bases.enumeration               as alg_configurations

class Configuration:
    #
    #
    #
    def __init__(self):
        #
        self.list_representative_problem_size   = []
        self.list_tile_sizes                    = []

        #
        self.list_tensor_C  = []
        self.list_tensor_A  = []
        self.list_tensor_B  = []
        self.list_TB_X      = []
        self.list_TB_Y      = []
        self.list_TB_K      = []
        self.list_REG_X     = []
        self.list_REG_Y     = []
        self.list_GRID_X    = []
        self.list_splits    = []

        #
        self.size_TB_X      = 0
        self.size_TB_Y      = 0
        self.size_TB_K      = 0
        self.size_REG_X     = 0
        self.size_REG_Y     = 0

        #
        self.cost_total         = 0
        self.cost_load_input    = 0
        self.cost_load_output   = 0
        self.cost_store_output  = 0

    #
    def add_representative_problem_size(self, representative_problem_size):
        for each_pair in representative_problem_size:
            self.list_representative_problem_size.append(each_pair)

    #
    def add_tile_size(self, list_input_tile_sizes):
        for each_pair in list_input_tile_sizes:
            self.list_tile_sizes.append(each_pair)

    #
    def add_split_index(self, list_split_info):
        for each_idx in list_split_info:
            self.list_splits.append([each_idx[0], each_idx[1], each_idx[2]])

    #
    def add_tensor_C(self, tensor_C):
        for each_info in tensor_C:
            self.list_tensor_C.append(each_info)

    #
    def add_tensor_A(self, tensor_A):
        for each_info in tensor_A:
            self.list_tensor_A.append(each_info)
    
    #
    def del_idx_tensor_A(self, str_target_idx):
        idx_count = 0
        for each_idx in self.list_tensor_A:
            if each_idx == str_target_idx:
                self.list_tensor_A.pop(idx_count)

            idx_count = idx_count + 1
            
    #
    def offset_tensor_A(self, str_target_idx):
        idx_count = 0;
        for each_idx in self.list_tensor_A:
            if each_idx == str_target_idx:
                return idx_count
            
            idx_count = idx_count + 1 

        return -1

    #
    def add_tensor_B(self, tensor_B):
        for each_info in tensor_B:
            self.list_tensor_B.append(each_info)

    #
    def del_idx_tensor_B(self, str_target_idx):
        idx_count = 0
        for each_idx in self.list_tensor_B:
            if each_idx == str_target_idx:
                self.list_tensor_B.pop(idx_count)
            
            idx_count = idx_count + 1

    #
    def offset_tensor_B(self, str_target_idx):
        idx_count = 0
        for each_idx in self.list_tensor_B:
            if each_idx == str_target_idx:
                return idx_count
            
            idx_count = idx_count + 1
        
        return -1

    #
    def add_GRID_X(self, GRID_X):
        for each_info in GRID_X:
            self.list_GRID_X.append(each_info)

    #
    def add_TB_X(self, TB_X):
        for each_info in TB_X:
            self.list_TB_X.append(each_info)

    #
    def add_TB_Y(self, TB_Y):
        for each_info in TB_Y:
            self.list_TB_Y.append(each_info)

    #
    def add_TB_K(self, TB_K):
        for each_info in TB_K:
            self.list_TB_K.append(each_info)

    #
    def add_REG_X(self, REG_X):
        for each_info in REG_X:
            self.list_REG_X.append(each_info)

    #
    def add_REG_Y(self, REG_Y):
        for each_info in REG_Y:
            self.list_REG_Y.append(each_info)

    #
    def print_representative_problem_size(self):
        print ("Representative Problem Size: ", self.list_representative_problem_size)

    #
    def print_splits(self):
        print ("Split Indices: ", self.list_splits)

    #
    def print_tensor_C(self):
        print ("Tensor C: ", self.list_tensor_C)

    #
    def print_tensor_A(self):
        print ("Tensor A: ", self.list_tensor_A)
    
    #
    def print_tensor_B(self):
        print ("Tensor B: ", self.list_tensor_B)

    #
    def print_REG_X(self):
        print ("REG_X: ", self.list_REG_X)

    #
    def print_REG_Y(self):
        print ("REG_Y: ", self.list_REG_Y)

    #
    def print_TB_X(self):
        print ("TB_X: ", self.list_TB_X)

    #
    def print_TB_Y(self):
        print ("TB_Y: ", self.list_TB_Y)
    
    #
    def print_TB_K(self):
        print ("TB_K: ", self.list_TB_K)

    #
    def print_GRID_X(self):
        print ("BX_X: ", self.list_GRID_X)

    #
    def print_tile_sizes(self):
        print ("Tile-Sizes: ", self.list_tile_sizes)

    #
    def print_configuration(self):
        print ("===========================================================================")
        self.print_representative_problem_size()
        self.print_tile_sizes()
        self.print_tensor_C()
        self.print_tensor_A()
        self.print_tensor_B()
        self.print_REG_X()
        self.print_REG_Y()
        self.print_TB_X()
        self.print_TB_Y()
        self.print_TB_K()
        self.print_GRID_X()
        self.print_splits()
        print ("|TB| = ", self.size_TB_X, ", ", self.size_TB_Y, ", |REG| = ", self.size_REG_X, ", ", self.size_REG_Y)
        print ("===========================================================================")

#
#   Enumeration with Pruning (PACT 2018)
#
#   Constraint #1:  The FVI of C should be present in A or B.
#                   This FVI index should be mapped on TB.
#   Constraint #2:  The FVI of A and B should be mapped on TB.
#   Constraint #3:  If multiple indices are mapped on REG, 
#                   Then, they should be continuous.
#                   In terms of actual problem size, they should be continuous, 
#                   because they are continuous on global memory.
#   
#   The problem of Splitting an Index: It is hard to deal with actual Problem Size.
#
#
#
def tc_gen_permutations(l_outer_groups, list_configurations_outer_group, opt_data_type):
    print ("========================================== [Permutations] ===================================================")
    print (" # of Outer-Groups: ", len(l_outer_groups)) # Assumped that there is only a tensor contraction (PACT 2018)
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
    #
    #
    for each_idx in list_indices:
        list_representative_problem_size.append([each_idx, 16])

    print (" (Default) Representative Problem Size: ", list_representative_problem_size)

    #
    #   [Result] List of Configurations
    #
    list_configurations_temp        = list()
    
    #
    #   Per Each-Outer-Group
    #
    idx_outer_count = 1
    for each_outer_group in l_outer_groups:
        print (" > Outer-Group [", idx_outer_count, "] ")
        
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
            print (" >> Tensor-Contraction [", idx_tc_count, "] ", each_tc) 

            #
            #   Default:    Input(Left) ---> TB_X and REG_X
            #               But, if Input(Right) has the FVI in Output, Input(Right) ---> TB_X and REG_X
            #
            list_internal_indices   = each_tc[3]
            list_input_tensor_left  = each_tc[5]
            list_input_tensor_right = each_tc[7]

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
                print ("[Code Generator][tc_gen_permutations] One of Input Tensors has only one external index, resulting in splitting freely.")
                #
                #   Tensor (Left)
                #
                if num_ext_left == 1:
                    print ("(L) To Split First: ", list_input_tensor_left)

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
                    for each_idx in each_tc[1]:
                        if each_idx == prev_idx:
                            each_tc[1].insert(idx_count,        each_idx + "1")
                            each_tc[1].insert(idx_count + 1,    each_idx + "2")
                            each_tc[1].pop(idx_count + 2)
                            list_info_split.append([each_idx, each_idx + "1", each_idx + "2"])
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
                            #list_representative_problem_size.pop(idx_count + 2)
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
                print ("[Code Generator][tc_gen_permutations] Both Input Tensors have at lease two external indices, resulting in splitting exclusively.")

            #
            #   Input:  an Equation for a Tensor Contraction
            #           a Representative Problem Size
            #   Output: a List of Configurations 
            #
            tc_gen_permutations_enumerating_all(each_tc, 
                                                list_representative_problem_size, 
                                                list_configurations_temp,
                                                opt_limited_split,  # opt_limited_split (0: free, 1: limited)
                                                1)  # opt_print

            #
            #   New Mapping Algorithms
            #
            list_temp = alg_configurations.alg_enumeration_pruning(each_tc, list_representative_problem_size, opt_limited_split, 0, opt_data_type)
            print ("[Code Generator][src][algs][bases] configurations: # of Configurations--- Total: ", len(list_temp))

            #for each_pruned_config in list_temp:
            #    each_pruned_config.print_configuration()
        
            print ("[Code Generator][tc_gen_permutations] # of Configurations: ", len(list_configurations_temp))
            if len(list_temp) < 1:
                print ("[Code Generator][tc_gen_permutations] ERROR: Problem(s) in Enumerating Configurations")
                sys.exit()

            if len(list_configurations_temp) < 1:
                print ("[Code Generator][tc_gen_permutations] ERROR: Problem(s) in Enumerating Configurations")
                sys.exit()

            #
            #   Models: each configuration has its own cost.
            #
            #tc_gen_models.tc_gen_models_Total_Cost(list_configurations_temp, 0)
            tc_gen_models.tc_gen_models_Total_Cost(list_temp, 0)

            #
            #   All Configurationss
            #
            idx_configuration   = 0
            min_cost            = 1000000000000
            idx_count = 0
            #for each_config in list_configurations_temp:
            for each_config in list_temp:
                if min_cost > each_config.cost_total:
                    min_cost = each_config.cost_total
                    idx_configuration = idx_count
                #
                idx_count += 1
            #
            #print ("min. Cost: ", min_cost)
            #print ("idx_configuration: ", idx_configuration)

            #
            #list_configurations_outer_group.append(list_configurations_temp[idx_configuration])
            list_configurations_outer_group.append(list_temp[idx_configuration])

        #
        each_outer_group.append(list_info_split)
        #print (" [After] >>> ", each_outer_group)
        idx_outer_count = idx_outer_count + 1

    list_configurations_outer_group[0].print_configuration()
    #
    #    
    #
    print ("=============================================================================================================")


#
#   To Enumerate All Configuration
#
def tc_gen_permutations_enumerating_all(each_tensor_contraction, list_representative_problem_size, 
                                        list_configurations_temp,
                                        opt_limited_split,
                                        opt_print):
    #
    #
    #
    list_tiles_TB   = [4, 8, 16]        # In the Future,    list_tiles_TB  = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,...]
    list_tiles_REG  = [1, 2, 4, 6, 8]   #                   list_tiles_REG = [1,2,3,4,5,6,7,8,...]

    #
    #   Related to "opt_limited_split"
    #
    if opt_limited_split == 0:
        print ("[Code Generator][tc_gen_permutations_enumerating_all] Splitting Freely")
    else:
        print ("[Code Generator][tc_gen_permutations_enumerating_all] Splitting Exclusively")

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
    list_internal_indices   = each_tensor_contraction[3]
    list_output_tensor      = each_tensor_contraction[1]
    list_input_tensor_left  = each_tensor_contraction[5]
    list_input_tensor_right = each_tensor_contraction[7]

    if opt_print == 1:
        print ("========================================== [Enumerations-ALL] ===================================================")
        print (" List of |TB_X|  or |TB_Y|:  ", list_tiles_TB)
        print (" List of |REG_X| or |REG_Y|: ", list_tiles_REG)
        print (" Given Tensor Contraction: ", each_tensor_contraction)
        print (" > Output Tensor (a.k.a. External Indices): ", list_output_tensor)
        print (" > Input Tensor (LEFT):  ", list_input_tensor_left)
        print (" > Input Tensor (RIGHT): ", list_input_tensor_right)
        print (" > Internal Indices: ", list_internal_indices)
        print ("=================================================================================================================")

    #
    #   Assumption: Indices in an input tensor will be mapped on one of x-axis and y-axis exclusively.
    #               This is just for "a single tensor contraction."
    #
    #   [0] One of Input Tensors whose one of indices is the FVI in the output tensor will be mapped on x-axis.
    opt_fvi_input = -1  
    for each_idx in list_input_tensor_left:
        if each_idx == list_output_tensor[0]:
            opt_swap = 1

    for each_idx in list_input_tensor_right:
        if each_idx == list_output_tensor[0]:
            opt_swap = 2

    #
    #
    #
    if opt_swap == 1:
        print ("[Code Generator][tc_gen_permutations_enumerating_all] L. Tensor has THE FVI in the Output")
    else:
        print ("[Code Generator][tc_gen_permutations_enumerating_all] R. Tensor has THE FVI in the Output")
        list_input_tensor_left  = each_tensor_contraction[7]
        list_input_tensor_right = each_tensor_contraction[5]
        if opt_print == 1:
            print (" > Input Tensor (LEFT):  ", list_input_tensor_left)
            print (" > Input Tensor (RIGHT): ", list_input_tensor_right)
    print ("=================================================================================================================")

    #
    #   Lists--- TB_X and TB_K
    #
    list_TB_X = []  # initial
    list_TB_K = []
    tc_gen_permutations_enumerating_TB_K_wo_split(list_tiles_TB, 
                                                    list_internal_indices, 
                                                    list_representative_problem_size,
                                                    list_TB_K, 
                                                    1)
    
    #
    #   [Default] TB_X
    #
    list_TB_X.append(list_output_tensor[0])
    print ("[Code Generator][tc_gen_permutations_enumerating_all] (Default) list_TB_X: ", list_TB_X, " (the FVI in the output tensor)")

    #
    #
    #
    tc_gen_perm_exclusive.tc_gen_perms_exclusive_REG_X(list_tiles_REG, list_tiles_TB, 
                                            list_output_tensor,
                                            list_input_tensor_left, list_input_tensor_right, 
                                            list_internal_indices, 
                                            list_representative_problem_size,
                                            list_TB_K, list_TB_X,
                                            list_configurations_temp,   # result
                                            0)


    #
    #   [1] REG (X -> Y): This makes REG_X and REG_Y
    #   [2] TB  (X -> Y)
    #
    '''
    tc_gen_permutations_enumerating_REG_X_wo_split(list_tiles_REG, list_tiles_TB, 
                                            list_output_tensor,
                                            list_input_tensor_left, list_input_tensor_right, 
                                            list_internal_indices, 
                                            list_representative_problem_size,
                                            list_TB_K,
                                            list_configurations_temp,   # result
                                            1)
    '''

    print ("[Code Generator][tc_gen_permutations_enumerating_all] # of Configurations): ", len(list_configurations_temp))  # DEBUG

    #
    if opt_print == 1:
        print ("=================================================================================================================")
#
#   This is for the base implementation
#
def tc_gen_permutations_enumerating_REG_X_wo_split(list_sizes_REG, list_sizes_TB,
                                                    list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                    list_internal_indices,
                                                    list_representative_problem_size,
                                                    list_TB_K,
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
            #
            #
            str_start_index = list_given_input_tensor_left[start_index]

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
            #   |REG_X'|
            #
            REG_X_Vol = tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)

            #
            #   |REG_X| == |REG_X'|
            #
            if REG_X_Vol == size_REG_X:
                #
                #   str_start_index, REG_X_Vol
                #
                tc_gen_permutations_enumerating_REG_Y_wo_split(list_sizes_REG, list_sizes_TB,
                                                                list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                                list_internal_indices,
                                                                list_representative_problem_size,
                                                                str_start_index, REG_X_Vol,
                                                                list_TB_K,
                                                                list_CLASS_configuration,
                                                                opt_print)

#
#   Enumerating Internal Indices (TB_K)
#
def tc_gen_permutations_enumerating_TB_K_wo_split(list_sizes_TB, 
                                                list_internal_indices,
                                                list_representative_problem_size,
                                                list_TB_K,      # result
                                                opt_print):
    #
    #
    #
    if len(list_internal_indices) == 1:
        list_TB_K.append(list_internal_indices[0])
    else:
        for each_idx in list_internal_indices:
            list_TB_K.append(each_idx)

    #
    if opt_print == 1:
        print ("=================================================================================================================")
        print ("[Code Generator][tc_gen_permutations_enumerating_TB_K_wo_split] # of Internal Indices = ", len(list_internal_indices))
        print ("[Code Generator][tc_gen_permutations_enumerating_TB_K_wo_split] list_TB_K: ", list_TB_K)
        print ("=================================================================================================================")

#
#
#
def tc_gen_permutations_enumerating_REG_Y_wo_split(list_sizes_REG, list_sizes_TB,
                                                    list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                    list_internal_indices,
                                                    list_representative_problem_size,
                                                    REG_X_idx, size_REG_X,
                                                    list_TB_K,
                                                    list_CLASS_configuration,   # result
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
        print ("Tensor (LEFT): ", list_given_input_tensor_right)
        print ("len(LEFT): ", len_tensor_right, ", # of External Indices: ", num_ext_idx, ", # of Internal Indices: ", num_int_idx)
        print ("list_representative_problem_size: ", list_representative_problem_size)

    #
    #   For Each Tile-Size for REG_Y
    #
    for size_REG_Y in list_sizes_REG:
        if opt_print == 1:
            print ("|REG_Y| = ", size_REG_Y)
        #
        #
        #
        for start_index in range(0, len_tensor_right):
            #
            #
            #
            str_start_index = list_given_input_tensor_right[start_index]

            #
            #   #1. Internal Index
            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, str_start_index) != -1:
                continue
            
            #
            #   #2. The FVI in the Output
            #   Due to our assumption, There is no index corresponding to the FVI in the output.
            #
            if str_start_index == list_given_output_tensor[0]:
                continue

            #
            #   |REG_Y'|
            #
            REG_Y_Vol = tc_helper.tc_gen_helper_find(list_representative_problem_size, str_start_index)

            #
            #   |REG_Y| == |REG_Y'|
            #
            if REG_Y_Vol == size_REG_Y:
                #
                #
                #
                tc_gen_permutations_enumerating_TB_X_wo_split(list_sizes_TB, 
                                        list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                        list_internal_indices, list_representative_problem_size,
                                        REG_X_idx, size_REG_X,
                                        str_start_index, REG_Y_Vol,
                                        list_TB_K,
                                        list_CLASS_configuration,
                                        opt_print)

#
#
#
def tc_gen_permutations_enumerating_TB_X_wo_split(list_sizes_TB,
                                                    list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                    list_internal_indices,
                                                    list_representative_problem_size,
                                                    REG_X_idx, size_REG_X,
                                                    REG_Y_idx, size_REG_Y,
                                                    list_TB_K,
                                                    list_CLASS_configuration,
                                                    opt_print):
    #
    if opt_print == 1:
        print ("=====================================================================================")
        print ("|REG_X| = ", size_REG_X, ", |REG_Y| = ", size_REG_Y)
        print ("REG_X <-- ", REG_X_idx, ", REG_Y <-- ", REG_Y_idx)                                            
    
    #
    #
    #
    for size_TB_X in list_sizes_TB:
        #
        #
        #
        TB_X_Vol        = 1
        Mapping_TB_X    = []
        Tile_Sizes      = []
        check_find_it   = -1

        #
        #
        #
        for each_idx in list_given_input_tensor_left:
            #print ("each_idx: ", each_idx)
            #
            #
            #
            if each_idx == REG_X_idx or each_idx == REG_Y_idx:
                continue

            #
            #
            #
            checked_int_idx = -1
            for each_int_idx in list_internal_indices:
                if each_idx == each_int_idx:
                    checked_int_idx = 1

            #
            if checked_int_idx == 1:
                continue

            #
            #
            #
            TB_X_Vol *= tc_helper.tc_gen_helper_find(list_representative_problem_size, each_idx)

            #
            #
            #
            if TB_X_Vol == size_TB_X:
                #print ("TB_X_Vol = size_TB_X // ", TB_X_Vol, "=", size_TB_X)
                Mapping_TB_X.append(each_idx)
                Tile_Sizes.append([each_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_idx)])
                Tile_Sizes.append([REG_X_idx, size_REG_X])
                Tile_Sizes.append([REG_Y_idx, size_REG_Y])

                check_find_it = 1
                #
                #   the remaining indices will be mapped on BX
                #
            elif TB_X_Vol > size_TB_X:
                #
                #
                #
                if check_find_it == 1:
                    Tile_Sizes.append([each_idx, 1])
                    Mapping_TB_X.append(each_idx)
                else:
                    break
            else:
                Mapping_TB_X.append(each_idx)
                Tile_Sizes.append([each_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_idx)])
                
        #
        #
        #
        if check_find_it == 1:
            if opt_print == 1:
                print ("=====================================================================================")
                print ("|TB_X|: ", size_TB_X)
                print ("Mapping_TB_X: ", Mapping_TB_X)
                print ("Tile-Sizes: ", Tile_Sizes)
                print ("=====================================================================================")
            tc_gen_permutations_enumerating_TB_Y_wo_split(list_sizes_TB,
                                                    list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                    list_internal_indices,
                                                    list_representative_problem_size,
                                                    REG_X_idx, size_REG_X,
                                                    REG_Y_idx, size_REG_Y,
                                                    Mapping_TB_X, size_TB_X,
                                                    Tile_Sizes,
                                                    list_TB_K,
                                                    list_CLASS_configuration,
                                                    opt_print)
        else:
            del Mapping_TB_X
            del Tile_Sizes
    #
    #
    #
    if opt_print == 1:
        print ("=====================================================================================")

#
#
#
def tc_gen_permutations_enumerating_TB_Y_wo_split(list_sizes_TB,
                                                    list_given_output_tensor, list_given_input_tensor_left, list_given_input_tensor_right,
                                                    list_internal_indices,
                                                    list_representative_problem_size,
                                                    REG_X_idx, size_REG_X,
                                                    REG_Y_idx, size_REG_Y,
                                                    Mapping_TB_X, size_TB_X,
                                                    Tile_Sizes,
                                                    list_TB_K,
                                                    list_CLASS_configuration,
                                                    opt_print):
    #
    #
    if opt_print == 1:
        print ("=====================================================================================")
        print ("TB_X: ", Mapping_TB_X)

    for size_TB_Y in list_sizes_TB:
        #
        #
        #
        TB_Y_Vol                = 1
        Mapping_TB_Y            = []
        check_find_it           = -1
        duplicatd_Tile_Sizes    = copy.deepcopy(Tile_Sizes)

        #
        #
        #
        for each_idx in list_given_input_tensor_right:
            #
            #
            #
            if each_idx == REG_X_idx or each_idx == REG_Y_idx:
                continue
            
            #
            #
            #
            checked_int_idx = -1
            for each_int_idx in list_internal_indices:
                if each_idx == each_int_idx:
                    checked_int_idx = 1
            
            #
            if checked_int_idx == 1:
                continue

            #
            #
            #
            TB_Y_Vol *= tc_helper.tc_gen_helper_find(list_representative_problem_size, each_idx)

            #
            #
            #
            if TB_Y_Vol == size_TB_Y:
                Mapping_TB_Y.append(each_idx)
                duplicatd_Tile_Sizes.append([each_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_idx)])
                check_find_it = 1
            elif TB_Y_Vol > size_TB_Y:
                if check_find_it == 1:
                    Mapping_TB_Y.append(each_idx)
                    duplicatd_Tile_Sizes.append([each_idx, 1])
                else:
                    break
            else:
                Mapping_TB_Y.append(each_idx)
                duplicatd_Tile_Sizes.append([each_idx, tc_helper.tc_gen_helper_find(list_representative_problem_size, each_idx)])
                
        #
        #
        #
        if check_find_it == 1:
            if opt_print == 1:
                print ("=====================================================================================")
                print ("|TB_X|: ", size_TB_X, ", |TB_Y|: ", size_TB_Y)
                print ("Mapping_TB_X: ", Mapping_TB_X)
                print ("Mapping_TB_Y: ", Mapping_TB_Y)
                print ("Tile-Sizes: ", duplicatd_Tile_Sizes)
                print ("Input-L: ", list_given_input_tensor_left)
                print ("Input-R: ", list_given_input_tensor_right)
                print ("=====================================================================================")
            #
            #   Before Creating a Configuration,
            #   We need to check Constraints such as H/W and Performance.
            #
            size_SMEM_Left      = 1
            size_SMEM_Right     = 1

            #
            #
            #
            for each_idx in list_given_input_tensor_left:
                if tc_helper.tc_gen_helper_find(duplicatd_Tile_Sizes, each_idx) != -1:
                    size_SMEM_Left *= tc_helper.tc_gen_helper_find(duplicatd_Tile_Sizes, each_idx)

            for each_idx in list_given_input_tensor_right:
                if tc_helper.tc_gen_helper_find(duplicatd_Tile_Sizes, each_idx) != -1:
                    size_SMEM_Right *= tc_helper.tc_gen_helper_find(duplicatd_Tile_Sizes, each_idx)

            #
            #
            #
            if opt_print == 1:
                print ("|SMEM_L|: ", size_SMEM_Left)
                print ("|SMEM_R|: ", size_SMEM_Right)

            #
            #
            #
            if size_SMEM_Left == size_SMEM_Right:
                tmp_config = Configuration()
                tmp_config.add_tensor_C(list_given_output_tensor)
                tmp_config.add_tensor_A(list_given_input_tensor_left)
                tmp_config.add_tensor_B(list_given_input_tensor_right)
                tmp_config.add_REG_X([REG_X_idx])
                tmp_config.add_REG_Y([REG_Y_idx])
                tmp_config.add_TB_X(Mapping_TB_X)
                tmp_config.add_TB_Y(Mapping_TB_Y)
                tmp_config.add_TB_K(list_TB_K)

                #
                #   Temporally
                #   
                duplicatd_Tile_Sizes.append(["e", 16])
                duplicatd_Tile_Sizes.append(["f", 1])

                tmp_config.add_tile_size(duplicatd_Tile_Sizes)
                tmp_config.add_representative_problem_size(list_representative_problem_size)
                
                tmp_config.size_REG_X = size_REG_X
                tmp_config.size_REG_Y = size_REG_Y
                tmp_config.size_TB_X = size_TB_X
                tmp_config.size_TB_Y = size_TB_Y
                tmp_config.size_TB_K = 16

                list_CLASS_configuration.append(tmp_config)

    if opt_print == 1:
        print ("=====================================================================================")
    

#
#   >> Enumerating with Constraints <<
#   [1] REG_X
#
def tc_gen_permutations_enumerating_REG_X(list_tiles_REG, list_tiles_TB,
                                            list_output_tensor, list_input_tensor_left, list_input_tensor_right, 
                                            list_internal_indices,
                                            list_representative_problem_size,
                                            list_configurations_temp,
                                            opt_limited_split,
                                            opt_print):
    #
    #
    #
    num_ext_idx = 0
    num_int_idx = 0
    for each_left_idx in list_input_tensor_left:
        if tc_helper.tc_gen_helper_find_1d(list_internal_indices, each_left_idx) == -1:
            num_ext_idx += 1
        else:
            num_int_idx += 1
    #
    len_tensor_left = len(list_input_tensor_left)
    
    if opt_print == 1:
        print ("========================================== [Enumerations-REG_X] ===================================================")
        print ("Tensor (LEFT): ", list_input_tensor_left)
        print ("len(LEFT): ", len_tensor_left, ", # of External Indices: ", num_ext_idx, ", # of Internal Indices: ", num_int_idx)
        print ("list_representative_problem_size: ", list_representative_problem_size)

    #
    if opt_limited_split == 1: 
        if num_ext_idx < 2:
            opt_limited_split = 0

    if opt_limited_split == 0:
        print ("THERE IS NOT LIMITED-SPLIT-TECHNIQUE")
    else:
        print ("THERE IS LIMITED-SPLIT-TECHNIQUE, BUT NOT YET IMPLEMENTED")

    #
    #   For Each Tile-Size for REG_X
    #
    for tile_size_REG_X in list_tiles_REG:
        #
        #
        #
        if opt_print == 1:
            print ("========================================== [Enumerations-REG_X-START] ===================================================")
            print ("|REG_X| = ", tile_size_REG_X)
            print ("=========================================================================================================================")
        #
        #   For each "star_index," there might be a configuration for an input tensor.
        #
        for start_index in range(0, len_tensor_left):
            #
            #
            str_start_index = list_input_tensor_left[start_index]

            #
            #   We should skip. If not, the results might be duplicated.
            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, str_start_index) != -1:
                continue
    
            #
            #   To Skip the FVI in Output Tensor 
            #
            if str_start_index == list_output_tensor[0]:
                continue

            #
            #
            REG_X_idx                               = list()
            list_tile_size                          = list()
            REG_X_vol_prev                          = 1
            REG_X_vol                               = 1
            blocking_size                           = 0
            duplicated_tensor_left                  = copy.deepcopy(list_input_tensor_left)
            duplicated_tensor_output                = copy.deepcopy(list_output_tensor)
            duplicated_representative_problem_size  = copy.deepcopy(list_representative_problem_size)

            #
            #   For an Input Tensor (LEFT or RIGHT),
            #   >>> a Single Configuration  
            # 
            for tensor_index in range(start_index, len_tensor_left):
                #
                str_current_index   = list_input_tensor_left[tensor_index]
                info_split_left     = list()

                #
                #   Internal Indices cannot be mapped to REG_X
                #   This is for multiple internal indices
                #  
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, str_current_index) != -1:
                    continue
                #
                #   [0] This is a Start Point.
                #
                if opt_print == 1:
                    print ("=========================================================================================================================")
                    print (" Start from [", start_index, "] external indices: ", str_current_index)
                #
                #   External Indices can be mapped to REG_X
                #
                REG_X_vol = REG_X_vol * tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, str_current_index)

                #
                #   [REG_X] is fully mapped.
                #
                if REG_X_vol >= tile_size_REG_X:
                    #
                    #   [1] REG_X_vol > tile_size_REG_X
                    #       : "the blocking dimension"
                    #
                    if REG_X_vol > tile_size_REG_X:
                        if opt_print == 1:
                            print ("[1] REG_X_vol > tile_size_REG_X: ", REG_X_vol, " > ", tile_size_REG_X)

                        #
                        #   Last index is the blocking dimension
                        #
                        blocking_size = tile_size_REG_X / REG_X_vol_prev

                        #
                        #
                        #
                        if blocking_size - int(blocking_size) > 0:
                            print ("Blocking_size: Floating-Point")
                            print ("Discard... and Need to Decide |REG_X|")
                        else:
                            #   [To-Do]
                            #   Split an index of size S into S1,S2 such that S = S1*S2 in an "input".
                            #
                            duplicated_tensor_left.insert(tensor_index + 1, str_current_index + "_2")
                            duplicated_tensor_left[tensor_index] = str_current_index + "_1" 
                            info_split_left.append([str_current_index, str_current_index + "_1", str_current_index + "_2"])

                            duplicated_representative_problem_size.append([str_current_index + "_1", int(blocking_size)])
                            duplicated_representative_problem_size.append([str_current_index + "_2", int(tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, str_current_index) / blocking_size)])
                            
                            #
                            if opt_print == 1:
                                print ("[after]  copied tensor_left: ", duplicated_tensor_left, ", info_split_left: ", info_split_left)
                            #
                            #
                            #
                            REG_X_idx.append(str_current_index + "_1")
                            list_tile_size.append([str_current_index + "_1", int(blocking_size)])

                            #   [To-Do]
                            #   Since we split an external index in an input,
                            #   we have to split this index in the "output" as well.
                            #
                            idx_count = 0
                            for idx_output in duplicated_tensor_output:
                                if idx_output == str_current_index:
                                    duplicated_tensor_output.insert(idx_count + 1, str_current_index + "_2")
                                    duplicated_tensor_output[idx_count] = str_current_index + "_1"
                                idx_count = idx_count + 1
                    else:
                        #
                        #   REG_X <--- Index
                        #
                        REG_X_idx.append(str_current_index)
                        list_tile_size.append([str_current_index, tile_size_REG_X])
                    #
                    #
                    #
                    if opt_print == 1:
                        print ("REG_X: ", REG_X_idx)
                        print ("Tile-Sizes: ", list_tile_size)
                    #
                    #   [2] REG_X_vol == tile_size_REG_X 
                    #   [1] REG_X_vol  > tile_size_REG_X (After Splitting)
                    #   It returns All possible configurations related to an input Tensor (Right) and the corresponding output Tensor.
                    #   
                    
                    tc_gen_permutations_enumerating_REG_Y(list_tiles_REG, list_tiles_TB,
                                                            duplicated_tensor_output,   # result
                                                            duplicated_tensor_left,     # result
                                                            info_split_left,            # result
                                                            list_input_tensor_right,
                                                            list_internal_indices,
                                                            duplicated_representative_problem_size,
                                                            REG_X_idx,                  # result
                                                            list_tile_size,             # result
                                                            list_configurations_temp,   # Based on Class
                                                            tile_size_REG_X,            # |REG_X|
                                                            opt_limited_split,          # opt_limited_split
                                                            1)                          # opt_print
                    
                    #
                    break
                else:
                    #
                    #   REG_X <--- Index
                    #
                    REG_X_idx.append(str_current_index)
                    list_tile_size.append([str_current_index, tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, str_current_index)])

                #
                #
                #
                REG_X_vol_prev = REG_X_vol_prev * tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, str_current_index)

                #
                #   End-For-Statement--- Candidates
                #
            #
            #   After creating all possible configurations,
            #   then, we need to prune some of them.
            #   Finally, we return "Configurations" after prunning.
            #
            
            #
            #   End: For-Statement--- start_index
            #
            del REG_X_idx
            
#
#   Assumption: REG_X is determined before REG_Y
#
def tc_gen_permutations_enumerating_REG_Y(list_tiles_REG, list_tiles_TB,
                                        list_output_tensor,
                                        list_input_tensor_left, 
                                        info_split_left,
                                        list_input_tensor_right,
                                        list_internal_indices,
                                        list_representative_problem_size,
                                        REG_X_idx,
                                        list_tile_size,
                                        list_configurations_temp,
                                        size_REG_X,
                                        opt_limited_split,
                                        opt_print):
    if opt_print == 1:
        print ("========================================== [Enumerations-REG_Y] ===================================================")
    #
    #   TENSOR (LEFT) is Determined.
    #
    len_tensor_right = len(list_input_tensor_right)
    
    #
    #   For each tile-size,
    #
    for tile_size_REG_Y in list_tiles_REG:
        if opt_print == 1:
            print ("========================================== [Enumerations-REG_Y-START] ===================================================")
            print ("|REG_X| = ", size_REG_X)
            print ("|REG_Y| = ", tile_size_REG_Y)
            print ("=========================================================================================================================")

        #
        #   There are multiple ways to pick indices mapped on REG_Y based on "start_index"
        #
        for start_index in range(0, len_tensor_right):
            #
            #
            str_start_index = list_input_tensor_right[start_index]

            #
            #
            #
            if tc_helper.tc_gen_helper_find_1d(list_internal_indices, str_start_index) != -1:
                continue

            #
            #   To Skip the FVI in Output Tensor 
            #
            if str_start_index == list_output_tensor[0]:
                continue

            #
            #
            #
            REG_Y_idx                               = list()
            REG_Y_vol                               = 1
            REG_Y_vol_prev                          = 1
            blocking_size                           = 0
            duplicated_tensor_right                 = copy.deepcopy(list_input_tensor_right)
            duplicated_tensor_output                = copy.deepcopy(list_output_tensor)
            duplicated_list_tile_size               = copy.deepcopy(list_tile_size)
            duplicated_representative_problem_size  = copy.deepcopy(list_representative_problem_size)

            #
            #   For an input Tensor (RIGHT),
            #   per a start_index, there will be a single configuration.
            #
            for tensor_index in range(start_index, len_tensor_right):
                #
                str_current_index   = list_input_tensor_right[tensor_index]
                info_split_right    = list()

                #
                #   Internl Indices cannot be mapped to REG_Y
                #   This is for multiple internal indices
                #
                if tc_helper.tc_gen_helper_find_1d(list_internal_indices, str_current_index) != -1:
                    continue

                #
                #   
                #
                if opt_print == 1:
                    print ("=========================================================================================================================")
                    print (" Start from [", start_index, "] external indices: ", str_current_index)
                REG_Y_vol = REG_Y_vol * tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, str_current_index)

                #
                #
                #
                if REG_Y_vol >= tile_size_REG_Y:
                    #
                    if REG_Y_vol > tile_size_REG_Y:
                        if opt_print == 1:
                            print ("[1] REG_Y_vol > tile_size_REG_Y: ", REG_Y_vol, " > ", tile_size_REG_Y)
                        #
                        #   Last index is the blocking dimension
                        #
                        blocking_size = tile_size_REG_Y / REG_Y_vol_prev
                        
                        #
                        #   Need to Consider "Blocking_Size" is not Integer!!!
                        #
                        if blocking_size - int(blocking_size) > 0:
                            print ("blocking_size: floating-point")
                            #print ("discard....")
                            #print ("Need to Device |REG_Y|")
                        else:
                            #
                            #
                            #
                            duplicated_tensor_right.insert(tensor_index + 1, duplicated_tensor_right[tensor_index] + "_2")
                            duplicated_tensor_right[tensor_index] = duplicated_tensor_right[tensor_index] + "_1"
                            info_split_right.append([str_current_index, str_current_index + "_1", str_current_index + "_2"])
                            
                            duplicated_representative_problem_size.append([str_current_index + "_1", int(blocking_size)])
                            duplicated_representative_problem_size.append([str_current_index + "_2", int(tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, str_current_index) / blocking_size)])

                            #
                            if opt_print == 1:
                                print ("[after]  copied tensor_right: ", duplicated_tensor_right, ", info_split_right: ", info_split_right, ", blocking-size: ", blocking_size)

                            #
                            #   REG_Y <--- "idx_1" (Split)
                            #
                            REG_Y_idx.append(str_current_index + "_1") 
                            duplicated_list_tile_size.append([str_current_index + "_1", int(blocking_size)])

                            #
                            #
                            #
                            idx_count = 0
                            for idx_output in duplicated_tensor_output:
                                if idx_output == str_current_index:
                                    duplicated_tensor_output.insert(idx_count + 1, str_current_index + "_2")
                                    duplicated_tensor_output[idx_count] = str_current_index + "_1"
                                idx_count = idx_count + 1

                            if opt_print == 1:
                                print ("[after]  copied tensor_output: ", duplicated_tensor_output)
                    else:
                        #
                        #   REG_Y <--- Given Index
                        #
                        REG_Y_idx.append(str_current_index)
                        duplicated_list_tile_size.append([str_current_index, tile_size_REG_Y])
                        #
                    #
                    #
                    #
                    if opt_print == 1:
                            print ("REG_Y: ", REG_Y_idx)
                            print ("Tile-Sizes: ", duplicated_list_tile_size)

                    #
                    #   Call TB_X and then Call TB_Y
                    #
                    '''
                    tc_gen_permutations_enumerating_TB_X_new(list_tiles_TB, 
                                                             duplicated_tensor_output, 
                                                             list_input_tensor_left, duplicated_tensor_right, 
                                                             info_split_left, info_split_right,
                                                             REG_X_idx, REG_Y_idx,
                                                             duplicated_representative_problem_size,
                                                             duplicated_list_tile_size,
                                                             size_REG_X, tile_size_REG_Y,
                                                             list_internal_indices, 
                                                             list_configurations_temp,
                                                             1)
                    '''

                    #
                    #   Done: a configuration for a specific "start_index"
                    #   Data Structure of a Configuration: [output, input-left, info-split-left, input-right, info-split-right, info-BX, info-TX, info-RX]
                    #   
                    #break
                else:
                    #
                    #
                    #
                    if opt_print == 1:
                        print ("[3] REG_Y_vol < tile_size_REG_Y")
                    #
                    #   Case 1: Need to add the next index to REG_Y.                        >>> Loop is not done.
                    #   Case 2: Product of Representative Problem Size is less than REG_Y.  >>> Loop is done.
                    #
                    #   REG_Y <--- Given Index.
                    REG_Y_idx.append(str_current_index)
                    duplicated_list_tile_size.append([str_current_index, tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, str_current_index)])

                    #
                    if opt_print == 1:
                            print ("REG_Y: ", REG_Y_idx)
                            print ("Tile-Sizes: ", duplicated_list_tile_size)

                #
                #
                #
                REG_Y_vol_prev = REG_Y_vol_prev * tc_helper.tc_gen_helper_find(list_representative_problem_size, str_current_index)
            #
            #
            #
            del REG_Y_idx

#
#
#
def tc_gen_permutations_enumerating_TB(list_tiles_TB, list_internal_indices, list_configurations, list_CLASS_configuration):
    #
    #print ("========================================================= [Enumerations-TB] ========================================================")
    #
    #
    #   For each Configuration,
    #
    idx_count = 0
    for each_configuration in list_configurations:
        #
        #
        #
        #list_idx_REG_X = each_configuration[5]
        #list_idx_REG_Y = each_configuration[6]
        #print (">> REG_X: ", list_idx_REG_X)
        #print (">> REG_Y: ", list_idx_REG_Y)

        #
        #   TB_X -> TB_Y
        #
        tc_gen_permutations_enumerating_TB_X(list_tiles_TB, list_internal_indices, each_configuration, list_CLASS_configuration, 0)

        #
        idx_count = idx_count + 1

#
#   [Enumerate][TB_X]
#
def tc_gen_permutations_enumerating_TB_X(list_tiles_TB, 
                                                list_internal_indices, each_configuration, 
                                            list_CLASS_configuration,
                                            opt_print):
    #
    #
    #
    if opt_print == 1:
        print ("========================================================= [Enumerations-TB_X] ========================================================")

    #
    #   
    #
    list_given_tensor_C         = each_configuration[0]
    list_given_tensor_A         = each_configuration[1]
    list_given_tensor_B         = each_configuration[3]

    list_given_info_split_A     = each_configuration[2]
    list_given_info_split_B     = each_configuration[4]

    list_given_REG_X            = each_configuration[5]
    list_given_REG_Y            = each_configuration[6]

    list_given_representative_problem_size = each_configuration[7]

    size_given_REG_X            = each_configuration[8]
    size_given_REG_Y            = each_configuration[9]

    if opt_print == 1:
        print ("================================================================================================")
        print ("list_given_tensor_C: ", list_given_tensor_C)
        print ("list_given_tensor_A: ", list_given_tensor_A)
        print ("list_given_tensor_B: ", list_given_tensor_B)
        print ("list_given_info_split_A: ", list_given_info_split_A)
        print ("list_given_info_split_B: ", list_given_info_split_B)
        print ("list_given_REG_X: ", list_given_REG_X)
        print ("list_given_REG_Y: ", list_given_REG_Y)
        print ("list_given_representative_problem_size: ", list_given_representative_problem_size)
        print ("|REG|: ", size_given_REG_X, ", ", size_given_REG_Y)
        print ("================================================================================================")

    #
    #
    #
    for tile_size_TB_X in list_tiles_TB:
        #
        #
        #
        if opt_print == 1:
            print ("========================================== [Enumerations-TB_X-START] ===================================================")
            print (" >>> |TB_X| = ", tile_size_TB_X)
            print ("=========================================================================================================================")

        #
        #
        #
        TB_X_vol_prev   = 1
        TB_X_vol        = 1
        blocking_size   = 0
        list_TB_X       = []
        list_BX_X       = []
        
        # deepcopy a configuration??
        duplicated_tensor_C                     = copy.deepcopy(list_given_tensor_C)
        duplicated_tensor_A                     = copy.deepcopy(list_given_tensor_A)
        duplicated_representative_problem_size  = copy.deepcopy(list_given_representative_problem_size)
        duplicated_info_split_A                 = copy.deepcopy(list_given_info_split_A)

        #
        #
        #
        check_split = -1
        for each_idx in list_given_tensor_A:
            #print (">>> ", each_idx)
            #
            #   Currently, Indices in LEFT TENSOR Should be Mapped on REG_X. (Not REG_Y)
            #
            mapped_REG = -1
            for each_idx_reg_x in list_given_REG_X:
                if each_idx == each_idx_reg_x:
                    mapped_REG = 1
            
            #
            if mapped_REG == 1:
                continue
            
            #
            #   SKIP, The Internal Indicies.
            #
            checked_int_idx = -1
            for each_int_idx in list_internal_indices:
                if each_idx == each_int_idx:
                    checked_int_idx = 1

            #
            if checked_int_idx == 1:
                continue

            #
            #   
            #
            TB_X_vol = TB_X_vol * tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx)

            #
            #   
            #
            if TB_X_vol >= tile_size_TB_X:
                #
                #   [1] Split
                #
                if TB_X_vol > tile_size_TB_X:
                    #
                    #   blocking-size
                    #                    
                    blocking_size = tile_size_TB_X / TB_X_vol_prev

                    #
                    #   FIRST INDEX OVER |TB_X|
                    #
                    if check_split == -1:
                        #
                        #   Input-LEFT
                        #
                        offset_target_idx = tc_helper.tc_gen_helper_list_offset_str(duplicated_tensor_A, each_idx)
                        tc_helper.tc_gen_helper_list_pop_str(duplicated_tensor_A, each_idx)
                        duplicated_tensor_A.insert(offset_target_idx,       each_idx + "_1")
                        duplicated_tensor_A.insert(offset_target_idx + 1,   each_idx + "_2")
                        
                        #
                        #   Output
                        #
                        offset_target_idx = tc_helper.tc_gen_helper_list_offset_str(duplicated_tensor_C, each_idx)
                        tc_helper.tc_gen_helper_list_pop_str(duplicated_tensor_C, each_idx)
                        duplicated_tensor_C.insert(offset_target_idx,       each_idx + "_1")
                        duplicated_tensor_C.insert(offset_target_idx + 1,   each_idx + "_2")
                        
                        #
                        #   Representative Problem-Size
                        #
                        duplicated_representative_problem_size.append([each_idx + "_1", int(blocking_size)])
                        duplicated_representative_problem_size.append([each_idx + "_2", int(tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx) / blocking_size)])
                        tc_helper.tc_gen_helper_list_pair_pop_str(duplicated_representative_problem_size, each_idx, 0)
                        
                        #
                        #   Info-Splited Indices
                        #   
                        duplicated_info_split_A.append([each_idx, each_idx + "_1", each_idx + "_2"])

                        #
                        #   TB_X
                        #
                        list_TB_X.append(each_idx + "_1")

                        #
                        #   BX_X
                        #
                        list_BX_X.append(each_idx + "_2")

                        #
                        check_split = 1
                    else:
                        #print (">>> Need to map to BX")
                        #   BX_X
                        list_BX_X.append(each_idx)
                        #
                        #   [To-Do]
                        #   Rule: Need to Map It on TB_X with Tile-Size == 1?
                        #
                #
                #   [2] Non-Split
                #
                else:
                    list_TB_X.append(each_idx)

            else:
                #
                #
                #
                #print ("[else]", each_idx, " is mapped on TB_X: ", TB_X_vol)
                list_TB_X.append(each_idx)

            #
            #
            #
            TB_X_vol_prev = TB_X_vol_prev * tc_helper.tc_gen_helper_find(list_given_representative_problem_size, each_idx)

            #print ("list_TB_X: ", list_TB_X)
            #print ("list_BX_X: ", list_BX_X)
            #
            #print ("not mapped on REG: ", each_idx, ", ", tc_helper.tc_gen_helper_find(duplicated_configuration.list_representative_problem_size, each_idx), " |TB_X| = ", TB_X_vol)

        #
        #
        #print ("list_CLASS_configuration between TB_X and TB_Y: ", len(list_CLASS_configuration))   
        #
        tc_gen_permutations_enumerating_TB_Y(list_tiles_TB, 
                                                list_internal_indices,
                                                duplicated_tensor_C,                    #   FROM HERE
                                                duplicated_tensor_A,                    #   FROM HERE
                                                duplicated_info_split_A,
                                                list_given_tensor_B,
                                                list_given_info_split_B,
                                                list_given_REG_X,
                                                list_given_REG_Y,
                                                list_TB_X,                              #   FROM HERE
                                                list_BX_X,                              #   FROM HERE
                                                duplicated_representative_problem_size, #   FROM HERE
                                                list_CLASS_configuration,
                                                size_given_REG_X,
                                                size_given_REG_Y,
                                                tile_size_TB_X,
                                                0)

#
#   [Enumerate][TB_X]
#
def tc_gen_permutations_enumerating_TB_X_new(list_tiles_TB, 
                                            list_given_tensor_C,
                                            list_given_tensor_A,        list_given_tensor_B,
                                            list_given_info_split_A,    list_given_info_split_B,
                                            list_given_REG_X,           list_given_REG_Y,
                                            list_given_representative_problem_size,
                                            list_given_tile_sizes,
                                            size_given_REG_X,           size_given_REG_Y,
                                            list_internal_indices, 
                                            list_CLASS_configuration,
                                            opt_print):
    #
    #
    #
    if opt_print == 1:
        print ("========================================================= [Enumerations-TB_X] ========================================================")

    #
    #   
    #
    if opt_print == 1:
        print ("================================================================================================")
        print ("list_given_tensor_C: ", list_given_tensor_C)
        print ("list_given_tensor_A: ", list_given_tensor_A)
        print ("list_given_tensor_B: ", list_given_tensor_B)
        print ("list_given_info_split_A: ", list_given_info_split_A)
        print ("list_given_info_split_B: ", list_given_info_split_B)
        print ("list_given_REG_X: ", list_given_REG_X)
        print ("list_given_REG_Y: ", list_given_REG_Y)
        print ("list_given_representative_problem_size: ", list_given_representative_problem_size)
        print ("list_given_tile_sizes: ", list_given_tile_sizes)
        print ("|REG|: ", size_given_REG_X, ", ", size_given_REG_Y)
        print ("================================================================================================")

    #
    #
    #
    for tile_size_TB_X in list_tiles_TB:
        #
        #
        #
        if opt_print == 1:
            print ("========================================== [Enumerations-TB_X-START] ===================================================")
            print (" >>> |TB_X| = ", tile_size_TB_X)
            print ("=========================================================================================================================")

        #
        #
        #
        TB_X_vol_prev   = 1
        TB_X_vol        = 1
        blocking_size   = 0
        list_TB_X       = []
        list_BX_X       = []
        
        # deepcopy a configuration??
        duplicated_tensor_C                     = copy.deepcopy(list_given_tensor_C)
        duplicated_tensor_A                     = copy.deepcopy(list_given_tensor_A)
        duplicated_representative_problem_size  = copy.deepcopy(list_given_representative_problem_size)
        duplicated_info_split_A                 = copy.deepcopy(list_given_info_split_A)
        duplicated_tile_size                    = copy.deepcopy(list_given_tile_sizes)

        #
        #
        #
        check_split = -1
        for each_idx in list_given_tensor_A:
            #
            #   Currently, Indices in LEFT TENSOR Should be Mapped on REG_X. (Not REG_Y)
            #
            mapped_REG = -1
            for each_idx_reg_x in list_given_REG_X:
                if each_idx == each_idx_reg_x:
                    mapped_REG = 1
            
            #
            if mapped_REG == 1:
                continue
            
            #
            #   SKIP, The Internal Indicies.
            #
            checked_int_idx = -1
            for each_int_idx in list_internal_indices:
                if each_idx == each_int_idx:
                    checked_int_idx = 1

            #
            if checked_int_idx == 1:
                continue

            #
            #   |TB_X'|
            #
            TB_X_vol = TB_X_vol * tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx)

            #
            #   |TB_X'| >= |TB_X|
            #
            if TB_X_vol >= tile_size_TB_X:
                #
                #   [1] Split
                #
                if TB_X_vol > tile_size_TB_X:
                    #
                    #   blocking-size
                    #                    
                    blocking_size = tile_size_TB_X / TB_X_vol_prev
                    '''
                    if blocking_size - int(blocking_size) > 0:
                        print (">>> blocking_size: floating point <-- ", tile_size_TB_X, " / ", TB_X_vol_prev, ", blocking_size: ", blocking_size)
                    else:
                        print (">>> blocking_size: integer <-- ", tile_size_TB_X, " / ", TB_X_vol_prev, ", blocking_size: ", blocking_size)
                    '''
                    #
                    #   FIRST INDEX OVER |TB_X|
                    #
                    if check_split == -1:
                        #
                        #   Input-LEFT
                        #
                        offset_target_idx = tc_helper.tc_gen_helper_list_offset_str(duplicated_tensor_A, each_idx)
                        tc_helper.tc_gen_helper_list_pop_str(duplicated_tensor_A, each_idx)
                        duplicated_tensor_A.insert(offset_target_idx,       each_idx + "_1")
                        duplicated_tensor_A.insert(offset_target_idx + 1,   each_idx + "_2")
                        
                        #
                        #   Output
                        #
                        offset_target_idx = tc_helper.tc_gen_helper_list_offset_str(duplicated_tensor_C, each_idx)
                        tc_helper.tc_gen_helper_list_pop_str(duplicated_tensor_C, each_idx)
                        duplicated_tensor_C.insert(offset_target_idx,       each_idx + "_1")
                        duplicated_tensor_C.insert(offset_target_idx + 1,   each_idx + "_2")
                        
                        #
                        #   Representative Problem-Size
                        #
                        duplicated_representative_problem_size.append([each_idx + "_1", int(blocking_size)])
                        duplicated_representative_problem_size.append([each_idx + "_2", int(tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx) / blocking_size)])
                        tc_helper.tc_gen_helper_list_pair_pop_str(duplicated_representative_problem_size, each_idx, 0)
                        
                        #
                        #   Info-Splited Indices
                        #   
                        duplicated_info_split_A.append([each_idx, each_idx + "_1", each_idx + "_2"])

                        #
                        #   Tile-Sizes (Split)
                        #
                        duplicated_tile_size.append([each_idx + "_1", int(blocking_size)])
                        #
                        #   TB_X
                        #
                        list_TB_X.append(each_idx + "_1")

                        #
                        #   BX_X
                        #
                        list_BX_X.append(each_idx + "_2")

                        #
                        check_split = 1
                    else:
                        #print (">>> Need to map to BX")
                        #   BX_X
                        list_BX_X.append(each_idx)
                        #
                        #   [To-Do]
                        #   Rule: Need to Map It on TB_X with Tile-Size == 1?
                        #
                #
                #   [2] Non-Split
                #
                else:
                    list_TB_X.append(each_idx)
                    #
                    #   Tile-Size (Non-Split)
                    #
                    duplicated_tile_size.append([each_idx, tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx)])

            else:
                #
                #
                #
                list_TB_X.append(each_idx)
                duplicated_tile_size.append([each_idx, tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx)])

            #
            #
            #
            TB_X_vol_prev = TB_X_vol_prev * tc_helper.tc_gen_helper_find(list_given_representative_problem_size, each_idx)

            #print ("list_TB_X: ", list_TB_X)
            #print ("list_BX_X: ", list_BX_X)
            #
            #print ("not mapped on REG: ", each_idx, ", ", tc_helper.tc_gen_helper_find(duplicated_configuration.list_representative_problem_size, each_idx), " |TB_X| = ", TB_X_vol)

        #
        #
        #print ("list_CLASS_configuration between TB_X and TB_Y: ", len(list_CLASS_configuration))   
        #
        tc_gen_permutations_enumerating_TB_Y(list_tiles_TB, 
                                                list_internal_indices,
                                                duplicated_tensor_C,                    #   FROM HERE
                                                duplicated_tensor_A,                    #   FROM HERE
                                                duplicated_info_split_A,
                                                list_given_tensor_B,
                                                list_given_info_split_B,
                                                list_given_REG_X,
                                                list_given_REG_Y,
                                                list_TB_X,                              #   FROM HERE
                                                list_BX_X,                              #   FROM HERE
                                                duplicated_representative_problem_size, #   FROM HERE
                                                duplicated_tile_size,
                                                list_CLASS_configuration,
                                                size_given_REG_X,
                                                size_given_REG_Y,
                                                tile_size_TB_X,
                                                0)


#
#   [Enumerate][TB_Y]
#
def tc_gen_permutations_enumerating_TB_Y(list_tiles_TB, 
                                            list_internal_indices,
                                            list_given_tensor_C,
                                            list_given_tensor_A,
                                            list_given_info_split_A,
                                            list_given_tensor_B,
                                            list_given_info_split_B,
                                            list_given_REG_X,
                                            list_given_REG_Y,
                                            list_given_TB_X,
                                            list_given_BX_X,
                                            list_given_representative_problem_size,
                                            list_given_tile_sizes,
                                            list_CLASS_configuration,
                                            size_REG_X, size_REG_Y,
                                            size_TB_X,
                                            opt_print):
    #
    #
    #
    if opt_print == 1:
        print ("========================================================= [Enumerations-TB_Y-START] ========================================================")

    
    #
    #
    #
    for tile_size_TB_Y in list_tiles_TB:
        #
        #
        #
        if opt_print == 1:
            print ("========================================== [Enumerations-TB_Y-EACH] ===================================================")
            print (" >>> |TB_Y| = ", tile_size_TB_Y)
            print ("=========================================================================================================================")

        #
        #
        #
        TB_Y_vol_prev   = 1
        TB_Y_vol        = 1
        blocking_size   = 0
        list_TB_Y       = []
        
        #   deepcopy
        duplicated_tensor_C                     = copy.deepcopy(list_given_tensor_C)
        duplicated_tensor_B                     = copy.deepcopy(list_given_tensor_B)
        duplicated_list_BX_X                    = copy.deepcopy(list_given_BX_X)
        duplicated_representative_problem_size  = copy.deepcopy(list_given_representative_problem_size) 
        duplicated_tiles_size                   = copy.deepcopy(list_given_tile_sizes)
        duplicated_info_split_B                 = copy.deepcopy(list_given_info_split_B)

        #
        #
        #
        check_split = -1
        for each_idx in list_given_tensor_B:
            #
            #   Indices Mapped on REG_Y
            #
            mapped_REG = -1
            for each_idx_reg_y in list_given_REG_Y:
                if each_idx == each_idx_reg_y:
                    mapped_REG = 1

            #
            if mapped_REG == 1:
                continue
            
            #
            #   Internal Indices
            #
            check_int_idx = -1
            for each_int_idx in list_internal_indices:
                if each_idx == each_int_idx:
                    check_int_idx = 1

            #
            if check_int_idx == 1:
                continue
            
            #
            #
            #
            TB_Y_vol = TB_Y_vol * tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx)

            #
            #
            #
            if TB_Y_vol >= tile_size_TB_Y:
                #
                #   [1] Split
                #
                if TB_Y_vol > tile_size_TB_Y:
                    #
                    #   Blocking_size
                    #
                    blocking_size = tile_size_TB_Y / TB_Y_vol_prev

                    #
                    #   FIRST INDEX OVER |TB_Y|
                    #
                    if check_split == -1:
                        #
                        #   Input-LEFT
                        #
                        offset_target_idx = tc_helper.tc_gen_helper_list_offset_str(duplicated_tensor_B, each_idx)
                        tc_helper.tc_gen_helper_list_pop_str(duplicated_tensor_B, each_idx)
                        duplicated_tensor_B.insert(offset_target_idx,       each_idx + "_1")
                        duplicated_tensor_B.insert(offset_target_idx + 1,   each_idx + "_2")
                        
                        #
                        #   Output
                        #
                        offset_target_idx = tc_helper.tc_gen_helper_list_offset_str(duplicated_tensor_C, each_idx)
                        tc_helper.tc_gen_helper_list_pop_str(duplicated_tensor_C, each_idx)
                        duplicated_tensor_C.insert(offset_target_idx,       each_idx + "_1")
                        duplicated_tensor_C.insert(offset_target_idx + 1,   each_idx + "_2")
                        
                        #
                        #   Representative Problem-Size
                        #
                        duplicated_representative_problem_size.append([each_idx + "_1", int(blocking_size)])
                        duplicated_representative_problem_size.append([each_idx + "_2", int(tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx) / blocking_size)])
                        tc_helper.tc_gen_helper_list_pair_pop_str(duplicated_representative_problem_size, each_idx, 0)
                        
                        #
                        #   Info-Splited Indices
                        #   
                        duplicated_info_split_B.append([each_idx, each_idx + "_1", each_idx + "_2"])

                        #
                        #   Tile-Sizes (Split)
                        #
                        duplicated_tiles_size.append([each_idx + "_1", int(blocking_size)])

                        #
                        #   TB_Y
                        #   
                        list_TB_Y.append(each_idx + "_1")

                        #
                        #   BX_X
                        #
                        duplicated_list_BX_X.append(each_idx + "_2")

                        #
                        check_split = 1
                    else:
                        #print (">>> Neet to Map on BX")
                        duplicated_list_BX_X.append(each_idx)
                #
                #   [2] Non-Split
                #
                else:
                    duplicated_list_BX_X.append(each_idx)
            else:
                #
                #   
                #
                list_TB_Y.append(each_idx)
                duplicated_tiles_size.append([each_idx, tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx)])
            #
            #
            #
            TB_Y_vol_prev = TB_Y_vol_prev * tc_helper.tc_gen_helper_find(duplicated_representative_problem_size, each_idx)
        #
        #   Class--- Configuration
        #
        a = Configuration()
        a.add_tensor_C(duplicated_tensor_C)
        a.add_tensor_A(list_given_tensor_A)
        a.add_tensor_B(duplicated_tensor_B)
        a.add_split_index(list_given_info_split_A)
        a.add_split_index(duplicated_info_split_B)
        
        a.add_REG_X(list_given_REG_X)
        a.add_REG_Y(list_given_REG_Y)
        a.add_TB_X(list_given_TB_X)
        a.add_TB_Y(list_TB_Y)
        a.add_GRID_X(duplicated_list_BX_X)

        a.size_TB_X     = size_TB_X
        a.size_TB_Y     = tile_size_TB_Y
        a.size_REG_X    = size_REG_X
        a.size_REG_Y    = size_REG_Y
        a.add_representative_problem_size(duplicated_representative_problem_size)
        a.add_tile_size(duplicated_tiles_size)

        #
        list_CLASS_configuration.append(a)

            
    #
    #
    #
    if opt_print == 1:
        print ("========================================================= [Enumerations-TB_Y-END] ========================================================")
