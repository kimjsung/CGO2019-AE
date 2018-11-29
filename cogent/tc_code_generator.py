#!/usr/bin/python
import sys
import os
import time

#
import src.generators.tc_gen    as tc_gen
import src.codes.tc_gen_code    as tc_gen_code

#
#   
#
if len(sys.argv) != 6:
    print ("[Code Generator] Please check arguments...")
    sys.exit()

#
#   Inputs
#
target_number               = int(sys.argv[1])
str_target_number           = sys.argv[1]
path_inputs                 = sys.argv[2]
target_number_config        = int(sys.argv[3])
str_target_number_config    = sys.argv[3]
opt_pre_computed            = int(sys.argv[4])
opt_data_type               = sys.argv[5]

#
#
#
print ("============================================================================")
print ("[Code Generator] target_number              = ", target_number)
print ("[Code Generator] str_target_number          = ", str_target_number)
print ("[Code Generator] path_inputs                = ", path_inputs)
print ("[Code Generator] target_number_config       = ", target_number_config)
print ("[Code Generator] str_target_number_config   = ", str_target_number_config)
print ("[Code Generator] opt_pre_computed           = ", opt_pre_computed)
print ("[Code Generator] opt_data_type              = ", opt_data_type)
print ("============================================================================")


time_overall_start = time.time()

#
#   [Step 1] Processing Inputs
#   : "tc_gen_input" should give several Outer-Groups which create their own cuda-files.
#   : An outer-group corresponds to a cuda-file.
#   : An inner-group corresponds to a kernel.
#   : An inner-group might have a tensor contraction or several tensor contractions which will be fused in a kernel.
#
list_inner_group, list_interface_info = tc_gen.tc_gen_input(target_number, target_number_config, path_inputs, 0, opt_data_type)

#
#   [Step 2] Create Kernel(s) based on groups.
#
#                                                                                              "1"                 "2"
tc_gen_code.tc_gen_code_new(target_number, str_target_number, str_target_number_config, list_inner_group, list_interface_info, opt_pre_computed, opt_data_type)

time_overall_end = time.time()
print ("[Code Generator][Time] Overall: ", time_overall_end - time_overall_start)