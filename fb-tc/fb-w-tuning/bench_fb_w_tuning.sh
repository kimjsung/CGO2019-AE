#!/bin/bash

for each_tccg in `seq 1 9`;
do
	python input_tccg_${each_tccg}.py |& tee output_${each_tccg}.txt
done

python data.py |& tee fb_tccg_w_tuning.txt

python time.py |& tee fb_tccg_tuning_time_sd2_1.txt
