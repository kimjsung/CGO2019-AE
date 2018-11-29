#
#
#
for tccg_number in range(1, 49):
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
    int_ops     = 0
    float_time  = 0.0
    while line:
        #

        #
        #   "# of Operations"
        #
        if "of Operations" in line:
        #    print ("# of Operations >>> ", line)    
            tmp_line = line.split()
            int_ops = int(tmp_line[4])
 
        #
        #   "Kernel-Time"
        #
        if "kernel" in line:
            tmp_line = line.split()
            float_time = float(tmp_line[1])

        #
        line = f.readline()
    #
    #
    #
    #
    if int_ops == 0 or float_time == 0.0:
        print ("[ERROR]")
    else:
        print ("tccg-" + str(tccg_number) + " >> ops: ", int_ops, "\ttime(ms): ", "{0:.4f}".format(float_time), "\tGFLOPS: ", "{0:.4f}".format(int_ops / (float_time * 1000*1000)))

    #
    f.close()
