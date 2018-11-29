#!/usr/bin/env python3
import re

#
#
for tccg_number in range(1, 2):
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
    best_time_so_far    = 1000000 

    #
    #
    #
    idx_count = 0
    num_iteration = 0
    while line:
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        line        = ansi_escape.sub('', line)
        line        = line.replace("\n", "") 

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
                    #
                    #   among the same generation
                    #
                    tmp_times = tmp_line[7].split("/")

                    #
                    if len(tmp_times) == 3:
                        if best_time_so_far > int(tmp_times[0]):
                            best_time_so_far = int(tmp_times[0])
                        #print ("Iteration: ", str(num_iteration), " >> # of Operations: ", int_ops, ", Best-Time(us): ", tmp_times[0], " >> GFLOPS: ", '{:.4f}'.format(int_ops / (float(tmp_times[0]) * 1000)))
                        print ("Iteration: ", str(num_iteration), " >> # of Operations: ", int_ops, ", Best-Time(us): ", best_time_so_far, " >> GFLOPS: ", '{:.4f}'.format(int_ops / (best_time_so_far * 1000)))
                        
                    #
                    num_iteration += 1
                        
        #
        line = f.readline()
    #
    f.close()
