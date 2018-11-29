#
#
#
def tc_gen_helper_GettingInputs(filename):#, l_str_input_tensor_contractions):
    #
    l_str_input_tensor_contractions = list()

    #
    with open(filename) as f:
        for line in f:
            if line[0] == "#":
                print ("Skipped: ", line.rstrip('\n'))
            else:
                print ("Given Eq.: ", line)
                l_str_input_tensor_contractions.append(line.rstrip('\n'))

    print ("[Code Generator][tc_gen_helper_GettingInputs] Processing Input.....")
    return l_str_input_tensor_contractions
