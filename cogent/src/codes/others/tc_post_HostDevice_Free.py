#
#
#
def tc_gen_code_post_HostFree(f, l_host_dynamic):
    f.write("\n")
    f.write("// created by tc_gen_code_post_HostFree()\n")
    f.write("void post_HostFree()\n")
    f.write("{\n")

    #   should call "tc_gen_code_helper_hostFree(f, var" for all dynamically allocated memory for host variables)
    f.write("\t// free(var); provided by tc_gen_code_helper_hostFree(var)\n")
    f.write("\t// free " + str(len(l_host_dynamic)) + " of memory\n")
    for h_name in l_host_dynamic:
        tc_gen_code_helper_hostFree(f, h_name)

    f.write("}\n")
    # End of tc_gen_code_post_HostFree()
#
#
#
def tc_gen_code_post_CUDA_Free(f, l_cuda_malloc):
    #
    f.write("\n")
    f.write("// created by tc_gen_code_post_CUDA_Free()\n")
    f.write("void post_CUDA_Free()\n")
    f.write("{\n")

    #   should call "tc_gen_code_helper_cudaFree(f, var)" for all dynamically allocated memory for cuda variables
    f.write("\t// cudaFree(var); provided by tc_gen_code_helper_cudaFree(var)\n")

    #   parameters (devices)
    for each_var in l_cuda_malloc:
        print ("each_var: ", each_var)
        tc_gen_code_helper_cudaFree(f, each_var[0])

    f.write("}\n")
    # End of tc_gen_code_post_CUDA_Free()

#
def tc_gen_code_helper_cudaFree_noline(f, var):
    f.write("\tcudaFree(" + var + ");")

#
def tc_gen_code_helper_hostFree_noline(f, var):
    f.write("\tfree(" + var + ");")

#
def tc_gen_code_helper_cudaFree(f, var):
    f.write("\tcudaFree(")
    f.write(var)
    f.write(");\n")

#
def tc_gen_code_helper_hostFree(f, var):
    f.write("\tfree(")
    f.write(var)
    f.write(");\n")
