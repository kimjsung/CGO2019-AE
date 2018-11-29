import src.codes.tc_code_helper                 as tc_code_helper
import src.codes.others.tc_pre_CUDA_Malloc      as tc_pre_CUDA_Malloc
import src.codes.others.tc_post_HostDevice_Free as tc_post_HostDevice_Free

#
def tc_gen_code_write_line(f, numTab, str_contents):
    
    for idx_tab in range(0, numTab):
        f.write("\t")
    
    f.write(str_contents)
    f.write("\n")

#
#
#
def tc_gen_code_fusedKernels(f, kernel_name,    l_t3_parameters,    l_t2_parameters,    l_v2_parameters,
                                                l_t3_parameters_nf, l_t2_parameters_nf, l_v2_parameters_nf,
                                                l_t3_parameters_f,  l_t2_parameters_f,  l_v2_parameters_f,  
                                                possible_diff,      len_inner_groups):
    f.write("\n")
    f.write("// created by tc_gen_code_fusedKernels()\n")
    f.write("void fusedKernels()\n")
    f.write("{\n")

    # related to tc_pre_CUDA_Malloc()
    f.write("\tpre_CUDA_Malloc();\n")

    f.write("\n")
    f.write("\tprintf (\" >>> %s <<<\\n\", __func__);\n")

    for idx_kernel in range(1, len_inner_groups + 1):
        f.write("\n")
        f.write("\tprintf (\"========================================= Kernel Info. ==============================================\\n\");\n")
        if possible_diff[0] == -1:
            f.write("\tprintf (\"\t\tGrid Size  : %6d (1D)\\n\", n_blks_" + str(idx_kernel) + ");\n")
        else:
            f.write("\tprintf (\"\t\tGrid Size for     Full Tiles: %6d (1D)\\n\", num_blk_full_" + str(idx_kernel) + ");\n")
            f.write("\tprintf (\"\t\tGrid Size for Non-Full Tiles: %6d (1D)\\n\", num_blk_non_full_" + str(idx_kernel) + ");\n")
        f.write("\tprintf (\"\t\tBlock-size : %2d, %2d (2D)\\n\", SIZE_TB_" + str(idx_kernel) + "_X, SIZE_TB_" + str(idx_kernel) + "_Y);\n")
        f.write("\tprintf (\"\t\tA thread deals with (%d x %d) elements (basically)\\n\", 4 * 16, 4 * 16);\n")
        f.write("\tprintf (\"====================================================================================================\\n\");\n")

    # streams
    if possible_diff[0] != -1:
        f.write("\n")
        f.write("\t// 2 Streams for (1) Full Tiles and (2) Non-Full Tiles\n")
        f.write("\tcudaStream_t streams[2];\n")
        f.write("\tcudaStreamCreate(&streams[0]);\n")
        f.write("\tcudaStreamCreate(&streams[1]);\n")

    # set Grid and Thread-Block
    f.write("\n")
    f.write("\t// Depends on # of Fused Kernel\n")

    #
    for idx_kernel in range(1, len_inner_groups + 1):
        # gridsize
        if possible_diff[0] == -1:
            f.write("\tdim3 gridsize_" + str(idx_kernel) + "(n_blks_" + str(idx_kernel) + ");\n")
        else:
            f.write("\tdim3 gridsize_full_" + str(idx_kernel) + "(num_blk_full_" + str(idx_kernel) + ");\n")
            f.write("\tdim3 gridsize_non_full_" + str(idx_kernel) + "(num_blk_non_full_" + str(idx_kernel) + ");\n")

        # blocksize
        f.write("\tdim3 blocksize_" + str(idx_kernel) + "(SIZE_TB_" + str(idx_kernel) + "_X, SIZE_TB_" + str(idx_kernel) + "_Y);\n")
        f.write("\n")

    # call Kernel
    f.write("\t// Depends on # of Fused Kernel\n")
    #f.write("\tcudaDeviceSynchronize();\n")

    #
    for idx_kernel in range(1, len_inner_groups + 1):
        if possible_diff[0] == -1:
            f.write("\t")
            f.write(kernel_name + "_" + str(idx_kernel))
            f.write("<<<gridsize_" + str(idx_kernel) + ", blocksize_" + str(idx_kernel) + ">>>(")

            # parameters (devices)
            for t3_var in l_t3_parameters[idx_kernel - 1]:
                f.write(t3_var[0])
                f.write(", ")
            f.write("\n\t")

            for t2_var in l_t2_parameters[idx_kernel - 1]:
                f.write(t2_var[0])
                f.write(", ")
            f.write("\n\t")

            for v2_var in l_v2_parameters[idx_kernel - 1]:
                f.write(v2_var[0])
                f.write(", ")
            f.write("\n\t")

            f.write("size_internal")

            f.write(");\n")
        else:
            # For non_full
            f.write("\t")
            f.write(kernel_name + "_" + str(idx_kernel))
            f.write("<<<gridsize_non_full_" + str(idx_kernel) + ", blocksize_" + str(idx_kernel) + ", 0, streams[0]>>>(")

            # parameters (devices)
            for t3_var in l_t3_parameters_nf[idx_kernel - 1]:
                f.write(t3_var[0])
                f.write(", ")
            f.write("\n\t")

            for t2_var in l_t2_parameters_nf[idx_kernel - 1]:
                f.write(t2_var[0])
                f.write(", ")
            f.write("\n\t")

            for v2_var in l_v2_parameters_nf[idx_kernel - 1]:
                f.write(v2_var[0])
                f.write(", ")
            f.write("\n\t")
            f.write("size_internal")
            f.write(");\n")
            f.write("\n")

            # For full
            f.write("\t")
            f.write(kernel_name + "_" + str(idx_kernel) + "_full")
            f.write("<<<gridsize_full_" + str(idx_kernel) + ", blocksize_" + str(idx_kernel) +  ", 0, streams[1]>>>(")

            # parameters (devices)
            for t3_var in l_t3_parameters_f[idx_kernel - 1]:
                f.write(t3_var[0])
                f.write(", ")
            f.write("\n\t")

            for t2_var in l_t2_parameters_f[idx_kernel - 1]:
                f.write(t2_var[0])
                f.write(", ")
            f.write("\n\t")

            for v2_var in l_v2_parameters_f[idx_kernel - 1]:
                f.write(v2_var[0])
                f.write(", ")
            f.write("\n\t")
            f.write("size_internal")
            f.write(");\n")

    #
    #f.write("\tcudaDeviceSynchronize();\n")
    f.write("\n")

    # post-processing after Kernel
    f.write("\tcudaMemcpy(h_t3, d_t3, sizeof(double) * size_T3, cudaMemcpyDeviceToHost);\n")

    # post_CUDA_Free
    f.write("\n")
    f.write("\tpost_CUDA_Free();\n")

    f.write("}\n")

#
def tc_gen_global_methods(f, num_inner_groups):
    # Global - Methods
    f.write("\n")
    f.write("// created by tc_gen_global_methods()\n")

    #
    for idx_kernel in range(1, num_inner_groups + 1):
        f.write("void pre_SD2_Functions_" + str(idx_kernel) + "();\n")

    #
    for idx_kernel in range(1, num_inner_groups + 1):
        f.write("void pre_BasicBlock_" + str(idx_kernel) + "();\n")

    #
    for idx_kernel in range(1, num_inner_groups + 1):
        f.write("void pre_IndirectArray_" + str(idx_kernel) + "();\n")

    #        
    f.write("void pre_CUDA_Malloc();\n")
    f.write("\n")
    f.write("void fusedKernels();\n")
    f.write("\n")
    f.write("void post_Correctness();\n")
    f.write("void post_HostFree();\n")
    f.write("void post_CUDA_Free();\n")
    f.write("\n")

#
def tc_gen_code_main(f, num_inner_groups):
    # main function
    f.write("\n")
    f.write("// created by tc_gen_code_main()\n")
    f.write("int main(int argc, char** argv)\n")
    f.write("{\n")

    #
    for idx_kernel in range(1, num_inner_groups + 1):
        f.write("\tpre_SD2_Functions_" + str(idx_kernel) + "();\n")
    
    #
    for idx_kernel in range(1, num_inner_groups + 1):
        f.write("\tpre_BasicBlock_" + str(idx_kernel) + "();\n")

    #
    for idx_kernel in range(1, num_inner_groups + 1):
        f.write("\tpre_IndirectArray_" + str(idx_kernel) + "();\n")

    f.write("\n")
    f.write("\tfusedKernels();\n")

    f.write("\n")
    f.write("\tpost_Correctness();\n")
    f.write("\tpost_HostFree();\n")

    #
    f.write("\n")
    f.write("\tprintf (\"This code is made by Ghaly\\n\");\n")

    f.write("}\n")
