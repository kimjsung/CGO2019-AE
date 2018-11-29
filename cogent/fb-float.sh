#!/bin/bash!

#
#   n^{th} equation in TCCG benchmark
#
if [ "$#" -eq 0 ]
then
    #   i.e., [0] tccg.sh 
    target_num=21
    target_num_config=-1
else
    if [ "$#" -eq 1 ]
    then
        #   i.e., [1] tccg.sh [target_number]
        target_num=$1
        target_num_config=-1
    else
        #   i.e., [2+] tccg.sh [target_number] [target_number_config] ....
        target_num=$1
        target_num_config=$2
    fi
fi

#
#
#
echo "============================================================================"
echo " >>> Creating CUDA code for TCCG's ${target_num}th Equation with ${target_num_config}th Configuration"
echo "============================================================================"

#
#
#
source_main=tc_code_generator.py
target_tc_list=input_strings/tccg/input_tccg_$1.in
opt_pre_computed=-1                                 # 1: on, -1: off
#opt_type="DOUBLE"
opt_type="FLOAT"

#
#
#
python ${source_main} ${target_num} ${target_tc_list} ${target_num_config} ${opt_pre_computed} ${opt_type}
