#
#   [LOAD INPUT-LEFT] Boundary Cases such as External and Internal Indices
#
def code_kernel_load_input_left_boundary_case(f, opt_gen_full_special_case, 
                                            cond_boundary_ext, cond_boundary_int, 
                                            str_cond_gen_external,  str_cond_gen_internal):
    #
    #
    #print (">>>>", str_cond_gen_external, ", ", str_cond_gen_internal)
    #
    f.write("\t\tif (")

    if opt_gen_full_special_case == -1:
        f.write(str_cond_gen_external)

        if cond_boundary_ext == 1 and cond_boundary_int == 1:
            f.write(" && ")
    
    f.write(str_cond_gen_internal)

    f.write(") // <<----\n")

#
#   [LOAD INPUT-RIGHT] Boundary Cases such as External and Internal Indices
#
def code_kernel_load_input_right_boundary_case(f, 
                                            cond_boundary_ext, cond_boundary_tbx, cond_boundary_int, 
                                            str_cond_gen_external,  str_cond_gen_tb_x, str_cond_gen_internal):
    f.write("\t\tif (")

    if cond_boundary_ext == 1:
        f.write(str_cond_gen_external)
    
    if cond_boundary_ext == 1 and cond_boundary_tbx == 1:
        f.write(" && ")
    
    if cond_boundary_tbx == 1:
        f.write(str_cond_gen_tb_x)

    if (cond_boundary_ext == 1 or cond_boundary_tbx == 1) and cond_boundary_int == 1:
        f.write(" && ")

    if cond_boundary_int == 1:
        f.write(str_cond_gen_internal)

    f.write(")\n")

#
#   [LOAD INPUT-LEFT] For-Statement
#
def code_kernel_load_input_left_for_statement(f, opt_gen_full, reg_mapped_axis, opt_gen_full_special_case,
                                            size_reg_tile, len_covered_reg, reg_mapped_indices_2D):
    #
    #
    #
    f.write("\t\tfor (int ll = 0; ll < ")

    #
    #   [1] Partial-Tiles (External)
    #
    if opt_gen_full == 1:
        #
        #   One of External Indices in an Input is mapped on REG_X
        #
        if reg_mapped_axis == "x":
            #
            if opt_gen_full_special_case > 0:
                #
                print ("1) special-case: on")
                f.write(str(int(size_reg_tile / len_covered_reg)))
            else:
                #
                print ("1) special-case: off")
                f.write("rng_" + reg_mapped_indices_2D[0])
            
        #
        #   One of External Indices in an Input is mapped on REG_Y
        #    
        else:
            #
            if opt_gen_full_special_case > 0:
                #
                print ("2) special-case: on >> ", size_reg_tile, ", ", len_covered_reg)
                f.write(str(int(size_reg_tile / len_covered_reg)))
            else:
                #
                print ("2) special-case: off")
                f.write("rng_" + reg_mapped_indices_2D[1])    
            
    #
    #   [2] Full-Tiles (External)
    #
    else:
        #
        if opt_gen_full_special_case > 0:
            #
            f.write(str(int(size_reg_tile / len_covered_reg)))
        else:
            f.write(str(size_reg_tile))
    #
    #   
    #
    f.write("; ll++)\n")
    f.write("\t\t{\n")

#
#   [LOAD INPUT-RIGHT] For-Statement
#
def code_kernel_load_input_right_for_statement(f, opt_gen_full, reg_mapped_axis, opt_gen_full_special_case,
                                                size_len_reg_tiles_right,
                                                size_reg_tile, len_covered_reg, reg_mapped_indices_2D):
    #
    #
    #
    f.write("\t\tfor (int ll = 0; ll < ")

    #
    #   Partial-Tile
    #
    if opt_gen_full == 1:
        if reg_mapped_axis == "x":
            #if opt_gen_full_special_case == 1:
            #    f.write(str(int(size_reg_tile / len_covered_reg)))
            #else:
            #    f.write("rng_" + reg_mapped_indices_2D[0])
            f.write("rng_" + reg_mapped_indices_2D[0])
        else:
            #if opt_gen_full_special_case == 1:
            #    f.write(str(int(size_reg_tile / len_covered_reg)))
            #else:
            #    f.write("rng_" + reg_mapped_indices_2D[1])
            f.write("rng_" + reg_mapped_indices_2D[1])
    #
    #   Full-Tile
    #
    else:
        if reg_mapped_axis == "x":
            f.write(str(max(1, int(size_len_reg_tiles_right / len_covered_reg))))    # "x"
        else:
            f.write(str(max(1, int(size_len_reg_tiles_right / len_covered_reg))))    # "y"

    f.write("; ll++)\n")
    f.write("\t\t{\n")