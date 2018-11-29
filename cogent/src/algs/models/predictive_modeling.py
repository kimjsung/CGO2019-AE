'''
    Predictive Modeling
    : to forecast performances such as GFLOPS.
'''
import src.generators.tc_helper     as tc_helper
#
#
#
def model_predictive_modeling(list_configurations):
    #
    #   architectures: p100 (sm_60), v100 (sm_70)
    #
    cuda_arch = "sm_70"

    #
    #   0. initialize
    #       0.1. init. indices
    #       0.2. init. reg. tiles
    #
    #   ----- main-loop -----
    #   1. load inputs
    #       1.1. load A
    #       1.2. load B
    #
    #   2. compute 
    #       2.1. 
    #   ---------------------
    #
    #   3. store output
    #
    print ("=========[model_predictive_modeling]=======================================================================")
    for each_config in list_configurations:
        #
        #   (1) m, n, k from a given representative problem size (This is based on the given equation not a configuration)
        #
        tmp_m = 1
        for each_idx in each_config.list_tensor_A:
            if tc_helper.tc_gen_helper_find_1d(each_config.list_TB_K, each_idx) == -1:
                tmp_m *= tc_helper.tc_gen_helper_find(each_config.list_representative_problem_size, each_idx)
        #
        each_config.m = tmp_m

        #
        tmp_n = 1
        for each_idx in each_config.list_tensor_B:
            if tc_helper.tc_gen_helper_find_1d(each_config.list_TB_K, each_idx) == -1:
                tmp_n *= tc_helper.tc_gen_helper_find(each_config.list_representative_problem_size, each_idx)
        #
        each_config.n = tmp_n
        
        #
        tmp_k = 1
        for each_idx in each_config.list_TB_K:
            tmp_k *= tc_helper.tc_gen_helper_find(each_config.list_representative_problem_size, each_idx)
        #
        each_config.k = tmp_k

        #
        #   (2) # of Thread Blocks (calculated in cost-model)
        #

        #
        #   (3) Estimated Number of Registers in a Thread Block
        #
        num_base = 20
        #
        #                       71, 70, 74, 71
        #   example: tccg 48th, 4x4: 16         > 32
        #                       4x1 + 1x1 = 5   > 10    : 42 ~ 29, 28, 32, 29
        #                                       
        #                       128, 121, 126, 120
        #                       6x6: 36         > 72
        #                       6x1 + 1x1 = 7   > 14    : 86 ~ 42, 35, 40, 34
        #
        #                       120, 122, 112, 113
        #                       4x8: 32         > 64
        #                       8x1 + 1x1 = 9   > 18    : 82 ~ 38, 40, 30, 31
        #
        #
        size_register_x = 1
        size_register_y = 1

        for each_idx in each_config.list_REG_X:
            size_register_x *= tc_helper.tc_gen_helper_find(each_config.list_tile_sizes, each_idx)

        for each_idx in each_config.list_REG_Y:
            size_register_y *= tc_helper.tc_gen_helper_find(each_config.list_tile_sizes, each_idx)
        
        each_config.num_Estimated_Registers = size_register_x * size_register_y * 2

        #
        #   (4) Kernel Efficiencies
        #

        #break
    print ("=========[model_predictive_modeling]=======================================================================")

    return 1000