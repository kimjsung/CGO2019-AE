#!/usr/bin/env python3
import re

#
#
for tccg_number in range(1, 10):
    #
    #
    #
    file_name = "output_"
    f = open(file_name + str(tccg_number) + ".txt")

    # use readline() to read the first line 
    line = f.readline()

    #
    #
    #
    int_ops             = 0
    float_time          = 0.0
    int_line_number     = 0
    str_current_gen     = ""
    str_temp_gen        = "0"
    list_min_gen        = []

    int_tmp_min_total   = 1000000000
    int_tmp_min_gen     = 1000000000

    #
    #
    #
    idx_count = 0
    while line:
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        line        = ansi_escape.sub('', line)
        line        = line.replace("\n", "") 

        #print (">>> ", line)
        #
        if "Operations" in line:
            tmp_line = line.split()
            int_ops = int(tmp_line[3])
        #
        #
        # 
        tmp_line = line.split()
        if tmp_line != []:
            if len(tmp_line) > 7:
                if tmp_line[0] == 'Generation':
                    str_current_gen = tmp_line[1]
                    #
                    #   among the same generation
                    #
                    if str_current_gen == str_temp_gen:
                        tmp_times = tmp_line[7].split("/")
                        #print (str_current_gen, ": ", tmp_line[7], ", ", tmp_times[0])
                       
                        #
                        if int(tmp_times[0]) < int_tmp_min_gen:
                            int_tmp_min_gen = int(tmp_times[0])
                            #print ("tmp_min_gen: ", int_tmp_min_gen)
                        
                    #
                    #   the first new generation
                    #
                    else:
                        #print (">>> changed generations from", str_temp_gen, " to",  str_current_gen)
                        list_min_gen.append([str_temp_gen, int_tmp_min_gen])
                        str_temp_gen = str_current_gen
                        int_tmp_min_gen = 100000000
                        
        #
        line = f.readline()
        int_line_number += 1
    #
    #   for the last generation
    #
    list_min_gen.append([str_current_gen, int_tmp_min_gen])
    
    #
    for each_gen in list_min_gen:
        if int_tmp_min_total > each_gen[1]:
            int_tmp_min_total = each_gen[1]

    #
    float_time= int_tmp_min_total

    #
    #
    #
    if int_ops == 0 or float_time == 0.0:
        print ("[ERROR]")
    else:
        print ("tccg-" + '{:2d}'.format(tccg_number) + " >> ops: ", int_ops, "\ttime(ms): ", '{:.8f}'.format(float_time), "\tGFLOPS: ", '{:.8f}'.format(int_ops / (float_time * 1000)))

    #
    f.close()
