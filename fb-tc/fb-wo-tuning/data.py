#
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
            #print (tmp_line)
            int_ops = int(tmp_line[3])
 
        #
        #   "Kernel-Time"
        #
        if "time" in line:
            tmp_line = line.split()
            #print (tmp_line)
            float_time = float(tmp_line[2])

        #
        line = f.readline()
    #
    #
    #
    #
    if int_ops == 0 or float_time == 0.0:
        print ("[ERROR]")
    else:
        print ("tccg-" + '{:2d}'.format(tccg_number + 39) + " >> ops: ", int_ops, "\ttime(s): ", "{0:.8f}".format(float_time), "\tGFLOPS: ", "{0:.8f}".format(int_ops / (float_time * 1000*1000*1000)) )

    #
    f.close()
