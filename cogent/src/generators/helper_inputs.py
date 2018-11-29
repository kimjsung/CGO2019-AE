
#
#
#
def transform_config_innergroup(given_configuration):
    #
    info_each_inner_group   = []
    
    #   Manually Generated Mappings for SD1 and SD2
    temp_mapping_TB_s_3     = []
    temp_mapping_2D_s_3     = []
    temp_mapping_Reg_s_3    = []
    temp_slices_s_3         = []
    temp_split_info         = []
    temp_mapping_TB_K       = []

    #    
    temp_mapping_2D_s_3.append(given_configuration.list_TB_X)
    temp_mapping_2D_s_3.append(given_configuration.list_TB_Y)

    for each_axis in temp_mapping_2D_s_3:
        for each_idx in each_axis:
            temp_mapping_TB_s_3.append(each_idx)

    temp_mapping_Reg_s_3.append(given_configuration.list_REG_X[0]) # REG_X
    temp_mapping_Reg_s_3.append(given_configuration.list_REG_Y[0]) # REG_Y

    for each_pair in given_configuration.list_tile_sizes:
        temp_slices_s_3.append(each_pair)

    for each_idx in given_configuration.list_TB_K:
        temp_mapping_TB_K.append(each_idx)

    #print ("give_configuration: ", given_configuration.list_split_representative_problem_size)
    #print ("temp_mapping_TB_K: ", temp_mapping_TB_K)

    #
    #
    #
    info_each_inner_group.append([temp_mapping_TB_s_3, temp_mapping_2D_s_3, temp_mapping_Reg_s_3, temp_slices_s_3, given_configuration.list_split_representative_problem_size, temp_mapping_TB_K])

    #
    return info_each_inner_group