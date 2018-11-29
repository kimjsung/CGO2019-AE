import sys
import src.generators.tc_helper             as tc_helper
import src.algs.bases.tc_gen_permutations   as tc_gen_permutations

import src.generators.configurations        as Configuration
import src.generators.helper_inputs         as Helper_Inputs

#
#   Inner-Goups depend on "Mapping" and "Tile-Sizes".
#   There might be different ways to fuse tensor contractions within a group.
#   For example, for sd2, we have two different approachs such as (1) 3 different kernels and (2) a single kernel with register transpose.
#   This can be determined in this function.
#   For IPDPS 2018, we manually determine it.
#   For CC 2018, we should use models to pick somewhat optimal way to fuse them.
#
def tc_gen_Inner_Group(l_outer_groups, tmp_count, tmp_config, opt_print, opt_data_type):
    #
    if opt_print == 1:
        print ("[Code Generator][Inner-Group] Working...")
        print ("====================== Step 2: Creating Inner-Groups =======================")
        print (" Only Support the First Outer-Group")

    #
    #   Permutations ([Assumption] Input: an arbitrary tensor contraction)
    #
    

    #
    #   To Handle 2D Tensors is in "tc_gen_permutations()"*****
    #

    #
    #   Cost-Model ([Verion 1.0] DRAM Data Movement, [Version 2.0] Bank Conflicts ...)
    #
    list_configurations_outer_group = list()
    Configuration.get_configurations(l_outer_groups, list_configurations_outer_group, tmp_count, tmp_config, 0, opt_data_type)

    #
    #
    #
    for each_config_outer_group in list_configurations_outer_group:
        each_config_outer_group.print_configuration(1)
    
    #
    #   (Temporary, Mapping and Tile-Sizes should be Determined by Models in the future.)
    #
    info_each_inner_group = Helper_Inputs.transform_config_innergroup(list_configurations_outer_group[0])

    #
    #   Every Outer-Groups:
    #
    print ("[Code Generator][Inner-Group] # of Outer-Groups: ", len(l_outer_groups))
    for each_outer_group in l_outer_groups:
        #
        print ("[Code Generator][Inner-Group] # of Tensor Contractions (Candidates) within an Outer-Group: ", len(each_outer_group[1]))

        #
        #   Within an Outer-Group, there might be several Inner-Groups.
        #
        l_inner_groups              = list()
        l_each_group_mapping_tb     = list()
        l_each_group_mapping_2D     = list()
        l_each_group_mapping_reg    = list()
        l_t3_slices_size            = list()
        l_t3_interface_info         = list()
        l_t3_temp_inputs            = list()
        l_t3_temp_conditions        = list()

        #
        #   To Create "Interface"
        #
        idx_count           = 0
        str_common_output   = ""
        for each_tc in each_outer_group[1]:
            l_t3_temp_inputs.append([each_tc[4], each_tc[6]])
            l_t3_temp_conditions.append("cond_kernel_" + str(idx_count + 1))
            if idx_count == 0:
                str_common_output = each_tc[0]
            idx_count = idx_count + 1
        #
        #   Information: Split Indices
        #
        l_each_group_split_info = each_outer_group[3]        

        #
        #   l_interface_info: [0] All Index, [1] Output, [2] Inputs, [3] Conditions, [4] Options
        #
        l_t3_interface_info.append([each_outer_group[2], str_common_output, l_t3_temp_inputs, l_t3_temp_conditions, "opt_register_transpose", l_each_group_split_info])

        #
        #   (Temporary)
        #                           [0]          [1]          [2]        [3]            [4]
        #   each_manual_group: Mapping_TB, Mapping_TB_2D, Mapping_Reg, Slices, Split-Info(Repre-size)
        #
        for each_manual_group in info_each_inner_group:
            #
            l_each_group_mapping_tb     = each_manual_group[0]
            l_each_group_mapping_2D     = each_manual_group[1]
            l_each_group_mapping_reg    = each_manual_group[2]
            l_t3_slices_size            = each_manual_group[3]
            l_info_split_ext            = each_manual_group[4]
            l_each_group_mapping_TB_K   = each_manual_group[5]
            l_tensor_contractions       = list()

            #
            print ("[Code Generator][Inner-Groups] Picked Tiles: ", l_t3_slices_size)

            #
            if opt_print == 1:
                print ("Target Outer-Group: ", each_outer_group[0])

            #
            #   Fusion-Constraint #2:
            #
            promissing_left     = 1
            promissing_right    = 1
            all_x_axis          = l_each_group_mapping_2D[0] + [l_each_group_mapping_reg[0]]
            all_y_axis          = l_each_group_mapping_2D[1] + [l_each_group_mapping_reg[1]]

            #
            #   X-Axis (Assumption: (Hypothetically) Left Input) including Output[0]
            #
            for each_idx in all_x_axis:
                promissing_left = promissing_left * tc_helper.tc_gen_helper_find(l_t3_slices_size, each_idx)

            #
            #   Y-Axis (Assumption: (Hypothetically) Right Input)
            #
            for each_idx in all_y_axis:
                promissing_right = promissing_right * tc_helper.tc_gen_helper_find(l_t3_slices_size, each_idx)

            print ("[Code Generator][Inner-Groups] Supposed Shared Memeory Lenghts: Left >>>", promissing_left, ", Right >>>", promissing_right)

            #
            #   Checking if All Tensor Contractions can be Fused or not.
            #
            l_picked_tc             = list()
            idx_tensor_contraction  = 0

            #   Checking if |info_each_inner_group| > |each_outer_group|,
            checking_used = 1
            if len(each_outer_group[1]) == 0:
                checking_used = -1

            #
            for each_tc in each_outer_group[1]:
                #
                #print (">> each_tc: ", each_tc)
                l_input_left = each_tc[5]
                l_input_right = each_tc[7]

                #
                #   Fusion-Constraint #1: Two indices for Register Tile should be on two different inputs.
                #
                #   LEFT
                idx_check_reg_left  = 0
                size_left           = 1
                for each_left_idx in l_input_left: #each_tc[7]:
                    #
                    #   Fusion-Constraint #1: Two indices for Register Tile should be on two different inputs.
                    #
                    if each_left_idx == l_each_group_mapping_reg[0]:
                        idx_check_reg_left = idx_check_reg_left + 1
                    if each_left_idx == l_each_group_mapping_reg[1]:
                        idx_check_reg_left = idx_check_reg_left + 1

                    #
                    #   Fusion-Constraint #2: The Size of Shared Memeory
                    #
                    #print ("[l] each_outer_group[0]: ", each_outer_group[0])
                    if tc_helper.tc_gen_helper_find_1d(each_outer_group[0], each_left_idx) != -1:
                        #print (">> ", each_left_idx)
                        size_left = size_left * tc_helper.tc_gen_helper_find(l_t3_slices_size, each_left_idx)

                #   RIGHT
                idx_check_reg_right = 0
                size_right          = 1
                for each_right_idx in l_input_right: #each_tc[5]:
                    #
                    #   Fusion-Constraint #1: Two indices for Register Tile should be on two different inputs.
                    #
                    if each_right_idx == l_each_group_mapping_reg[0]:
                        idx_check_reg_right = idx_check_reg_right + 1
                    if each_right_idx == l_each_group_mapping_reg[1]:
                        idx_check_reg_right = idx_check_reg_right + 1

                    #
                    #   Fusion-Constraint #2: The Size of Shared Memeory
                    #
                    #print ("[r] each_outer_group[0]: ", each_outer_group[0])
                    if tc_helper.tc_gen_helper_find_1d(each_outer_group[0], each_right_idx) != -1:
                        #print (">> ", each_right_idx)
                        size_right = size_right * tc_helper.tc_gen_helper_find(l_t3_slices_size, each_right_idx)

                #
                #   [Should be Fixed] 
                #
                if idx_check_reg_right != 2 and idx_check_reg_left != 2 and promissing_left == size_left and promissing_right == size_right:
                    l_picked_tc.append(idx_tensor_contraction)
                    l_tensor_contractions.append(each_tc)
                else:
                    print ("[DEBUG] promissing_left: ", promissing_left, ", promissing_right: ", promissing_right)
                    print ("[DEBUG] idx_check_reg_right: ", idx_check_reg_right, ", idx_check_reg_left: ", idx_check_reg_left, ", size_left: ", size_left, ", size_right: ", size_right)
                    sys.exit()

                #
                idx_tensor_contraction = idx_tensor_contraction + 1
                #
                #   End of For-Statement
                #
            
            #
            if checking_used == 1:
                #print ("added a tensor contraction to an inner-group")
                #print ("l_info_split_ext: ", l_info_split_ext)
                l_inner_groups.append([l_each_group_mapping_tb, l_each_group_mapping_2D, l_each_group_mapping_reg, l_tensor_contractions, l_t3_slices_size, l_info_split_ext, l_each_group_mapping_TB_K])

            #   To-Do: Should be checked in detail
            for each_tc in list(reversed(l_picked_tc)):
                each_outer_group[1].pop(each_tc)
                
        #
        #   For-Outer-Group
        #
        break   # To-Do: (Currently) Only 1 Outer-Group.
    #
    #
    #
    if opt_print == -1:
        print ("============================================================================")
        #
        #
        #
        print ("===================== Step 2: [Output] Inner-Groups ========================")
        print (" Does not Support Register-Transpose")
        print (" These Tensor Contractions will be fused.")
        print (" # of Inner-Groups: ", len(l_inner_groups))

        # [l_each_group_mapping_tb, l_each_group_mapping_2D, l_each_group_mapping_reg, l_tensor_contractions, l_t3_slices_size, l_info_split_ext, l_each_group_mapping_TB_K]
        for each_inner_group in l_inner_groups:
            print ("Mapping All: ",     each_inner_group[0])
            print ("Mapping TB : ",     each_inner_group[1])
            print ("Mapping Reg: ",     each_inner_group[2])
            print ("Mapping TB_K: ",    each_inner_group[6])
            print ("Slices : ",         each_inner_group[4])
            print ("Split-Slices : ",   each_inner_group[5])
            print ("# of Tensor Contractions: ", len(each_inner_group[3]))
            for each_tc in each_inner_group[3]:
                print ("Each Tensor Contraction: ", each_tc)

        print ("============================================================================")
    
    #
    #   Return Output
    #
    return l_inner_groups, l_t3_interface_info
