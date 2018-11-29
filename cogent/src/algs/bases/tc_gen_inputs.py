#
#   imports
#
import src.generators.tc_helper             as tc_helper
import src.algs.bases.tc_gen_permutations   as tc_gen_permutations
import src.generators.configurations        as Configuration
import src.generators.helper_inputs         as Helper_Inputs

#
#   Outer-Groups depend on Output's Indices.
#   To-Do:  All tensor contraction within an outer-group, Internal indices might be different.
#           It should be supported.
#   (Current Ver.) Within an outer-group, all tensor contractions has identical internal indices.
#
def tc_gen_Outer_Group(l_str_input_tensor_contractions, opt_print):
    #
    if opt_print == 1:
        print ("===================== Step 1: Creating Outer-Groups ========================")
        print ("[Code Generator][Outer-Group] Working...")

    #
    l_outer_groups  = list()
    idx_line        = 1
    for an_input in l_str_input_tensor_contractions:
        #
        if opt_print == 1:
            print ("[" + str(idx_line) + "]", an_input)

        # Step: Getting Rid of "[", "]", ";"
        target_input                = an_input.replace("[","")
        target_input                = target_input.replace(";","")
        target_input                = target_input.replace("]","")

        # Step: Create List From for Input String
        split_input                 = list(target_input.split())

        # Step: Split
        temp_output_name            = ""
        temp_output_index           = ""
        temp_operator               = ""
        temp_interanl_index         = ""
        temp_operand_left_name      = ""
        temp_operand_left_index     = ""
        temp_operand_right_name     = ""
        temp_operand_right_index    = ""

        #
        temp_output_name            = split_input[0]
        temp_output_index           = split_input[1]
        temp_operator               = split_input[2]
        temp_internal_index         = split_input[3]
        temp_operand_left_name      = split_input[5]
        temp_operand_left_index     = split_input[6]
        temp_operand_right_name     = split_input[8]
        temp_operand_right_index    = split_input[9]

        # Step: Representative-Problem-Size // list_tccg_representative_problem_sizes.append([312, 312, 24, 312])
        temp_processing_output                      = list(temp_output_index.split(","))
        temp_output_index                           = ""
        list_given_representative_problem_size_1    = list()
        list_given_representative_problem_size_2    = list()
        #
        if len(temp_processing_output) % 2 != 0:
            print ("ERROR: In the expression, the output tensor should have indices and their representative problem sizes")
        else:
            # some_list[start:stop:step]
            for each_idx in range(0, len(temp_processing_output), 2):
                list_given_representative_problem_size_1.append([temp_processing_output[each_idx], int(temp_processing_output[each_idx + 1])])
                list_given_representative_problem_size_2.append(int(temp_processing_output[each_idx + 1]))
                if each_idx == 0:
                    temp_output_index = temp_processing_output[each_idx]
                else:
                    temp_output_index = temp_output_index + "," + temp_processing_output[each_idx]
                
        # Step: Getting rid of "Sum()"
        temp_processing_internal    = temp_internal_index.replace("sum(","")
        temp_internal_index         = temp_processing_internal.replace(")","")
        temp_processing_internal    = list(temp_internal_index.split(","))
        temp_internal_index         = ""

        #
        if len(temp_processing_internal) % 2 != 0:
            print ("ERROR: In the expression, the internal indices should have indices and their representative problem sizes")
        else:
            #
            for each_idx in range(0, len(temp_processing_internal), 2):
                list_given_representative_problem_size_1.append([temp_processing_internal[each_idx], int(temp_processing_internal[each_idx + 1])])
                list_given_representative_problem_size_2.append(int(temp_processing_internal[each_idx + 1]))
                if each_idx == 0:
                    temp_internal_index = temp_processing_internal[each_idx]
                else:
                    temp_internal_index = temp_internal_index + "," + temp_processing_internal[each_idx]
        '''
        print ("1) ", list_given_representative_problem_size_1)
        print ("2) ", list_given_representative_problem_size_2)
        print (" >>> temp_output_index: ", temp_output_index)
        print (" >>> temp_internal_index: ", temp_internal_index)
        '''
        # Step: Creating List Form for Output Tensor's Indices, Input Tensors' Indices and Internal Indices.
        temp_list_left_index        = list(temp_operand_left_index.split(","))
        temp_list_right_index       = list(temp_operand_right_index.split(","))
        temp_list_output_index      = list(temp_output_index.split(","))
        temp_list_internal_index    = list(temp_internal_index.split(","))
        temp_list_all_index         = temp_list_output_index + temp_list_internal_index;

        # Pre-Processed Expression for an Input Tensor Contraction
        # Output-Name, Output-Index, Operator, Internal-Index, Left-Name, Left-Index, Right-Name, Right-Index
        temp_target = [temp_output_name, temp_list_output_index, temp_operator, temp_list_internal_index, temp_operand_left_name, temp_list_left_index, temp_operand_right_name, temp_list_right_index, list_given_representative_problem_size_1, list_given_representative_problem_size_2]

        #
        #   Step: Checking Outer-Groups
        #
        if len(l_outer_groups) == 0:
            l_outer_groups.append([temp_list_output_index, [temp_target,], temp_list_all_index])
        else:
            check_exist = -1
            for each_outer_group in l_outer_groups:
                if each_outer_group[0] == temp_list_output_index:
                    each_outer_group[1].append(temp_target)
                    check_exist = 1

            if check_exist == -1:
                l_outer_groups.append([temp_list_output_index, [temp_target,], temp_list_all_index])

        idx_line = idx_line + 1

    #
    if opt_print == 1:
        print ("============================================================================")
        #
        #
        #
        print ("============================================================================")
        print ("==================== Step 1: [Output] Outer-Groups =========================")
        print ("# of Outer-Groups: ", len(l_outer_groups))
        for each_outer_group in l_outer_groups:
            print ("> Outer-Group's Index: ", each_outer_group[0])
            tc_count = 1
            for each_tensor_constraction in each_outer_group[1]:
                print (" >> [" + str(tc_count) + "] Tensor Contraction: ", each_tensor_constraction)
                tc_count = tc_count + 1
        print ("============================================================================")

    #
    #   Return Output
    #
    return l_outer_groups

#
#   Tuning Data of Inner-Groups for "tc_gen_code"
#
def tc_gen_Processing_Inner_Group(l_inner_groups, tmp_count, opt_register_transpose, opt_print):
    #
    if opt_print == 1:
        print ("=================== Step 3: Processing Inner-Groups ========================")
        print (" Creates Data Structures used to create a Kernel based on a given inner group.")
    print ("[Code Generator][Processing Inner-Groups] # of Inner-Groups: ", len(l_inner_groups))

    #
    l_temp_inner_output     = list()

    #
    #   Processing Each Inner-Group
    #
    for each_inner_group in l_inner_groups:
        #
        l_temp_input_tensors    = list()
        l_temp_input_addrs      = list()
        l_temp_external_indices = list()
        l_temp_internal_indices = list()
        l_temp_all_indices      = list()

        #
        #if opt_print == 1:
        print ("[Code Generator][Configuration] Mapping All : ", each_inner_group[0])
        print ("[Code Generator][Configuration] Mapping TB  : ", each_inner_group[1])
        print ("[Code Generator][Configuration] Mapping TB_K: ", each_inner_group[6])
        print ("[Code Generator][Configuration] Mapping REG : ", each_inner_group[2])

        # Assumption: Everything related to the output is identical among all tensor contraction within an Inner-Group.
        # For Example, l_idx_size, l_t3_idx, l_external_idx
        # However, l_internal_idx might be different.
        # In this version, it supports only the case which l_intenral_idx is also identical among all tensor contractions within an Inner-Group.
        l_temp_external_indices = each_inner_group[3][0][1]
        l_temp_internal_indices = each_inner_group[3][0][3]
        
        #
        for each_ext_idx in l_temp_external_indices:
            l_temp_all_indices.append([each_ext_idx, 16])

        for each_int_idx in l_temp_internal_indices:
            l_temp_all_indices.append([each_int_idx, 16])

        #
        if opt_print == 1:
            print ("All Indices:",          l_temp_all_indices)
            print ("External Indices: ",    l_temp_external_indices)
            print ("Internal Indices: ",    l_temp_internal_indices)

        print ("[Code Generator][Processing Inner-Groups] Each Inner-Group Has # of Tensor Contractions: ", len(each_inner_group[3]))

        #
        for each_tc in each_inner_group[3]:   
            #
            if opt_print == 1:
                print ("Each Tensor Contraction: ", each_tc[0])

            #
            str_left_mapping = ""
            for left_idx in each_tc[5]:
                if left_idx == each_inner_group[2][0]:
                    str_left_mapping = "x"
                if left_idx == each_inner_group[2][1]:
                    str_left_mapping = "y"

            #
            str_right_mapping = ""
            for right_idx in each_tc[7]:
                if right_idx == each_inner_group[2][0]:
                    str_right_mapping = "x"
                if right_idx == each_inner_group[2][1]:
                    str_right_mapping = "y"

            #
            #   Create lists called input_tensors and input_addrs (which can be combined in the future)
            #
            l_temp_input_tensors.append([[each_tc[4], each_tc[5]], [each_tc[6], each_tc[7]], each_tc[2]])
            l_temp_input_addrs.append([ [16, "STR_SD2_" + each_tc[4].capitalize() + "_H7", str_left_mapping,  each_tc[4], each_tc[5]],
                                        [16, "STR_SD2_" + each_tc[6].capitalize() + "_H7", str_right_mapping, each_tc[6], each_tc[7]], each_tc[2]])
        #
        #
        #
        l_temp_inner_output.append( [each_inner_group[0],   each_inner_group[1],        each_inner_group[2],
                                    l_temp_all_indices,     l_temp_external_indices,    l_temp_internal_indices,
                                    l_temp_input_tensors,   l_temp_input_addrs,         each_inner_group[4],        each_inner_group[5],
                                    each_inner_group[6]])

        #
        for each_tc in l_temp_input_addrs:
            print ("[Code Generator][Processing Inner-Groups] Target Tensor Contraction: ", each_tc)

    #
    if opt_print == 1:
        print ("============================================================================")

    #
    #   After Processing Inner-Groups, if Register Transpose is ON, then....
    #
    if opt_register_transpose == 1:
        print ("[Code Generator][Processing Inner-Groups] Register Transpose: ON")
        l_ok_combined_inner_groups = tc_gen_Q_Fusion_Register_Transpose(l_temp_inner_output)

        if len(l_ok_combined_inner_groups) != 1:
            print ("[Code Generator][Processing Inner-Groups] Register Transpose is Impossible for a given Input. (", l_ok_combined_inner_groups, ")")
        else:
            print ("[Code Generator][Processing Inner-Groups] Register Transpose is Possible with", l_ok_combined_inner_groups)

            #
            #   Actuall Processing Inner-Groups for Register Transpose
            #


    #
    #   Return Output
    #
    return l_temp_inner_output
