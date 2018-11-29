import src.generators.tc_helper     as tc_helper

#
#
#
def tc_gen_code_driver_CUDA_Malloc(f, l_cuda_malloc, l_device_dynamic, opt_pre_computed):
    #
    #   cudaMalloc
    #
    f.write("\t// cudaMalloc()\n")
    for each_var in l_cuda_malloc:
        if opt_pre_computed == -1:
            if "range"  in each_var[0]:
                continue
            if "addr"   in each_var[0]:
                continue
            if "offset" in each_var[0]:
                continue
            if "base"   in each_var[0]:
                continue
            #
            tc_gen_code_helper_cudaMalloc(f, each_var[0], each_var[1], each_var[2])
        else:
            tc_gen_code_helper_cudaMalloc(f, each_var[0], each_var[1], each_var[2])
    f.write("\n")

    #
    #   cudaMemcpy
    #
    f.write("\t// cudaMemcpy()\n")
    for each_var in l_device_dynamic:
        if opt_pre_computed == -1:
            if "range"  in each_var[1]:
                continue
            if "addr"   in each_var[1]:
                continue
            if "offset" in each_var[1]:
                continue
            if "base"   in each_var[1]:
                continue
            #
            tc_gen_code_helper_cudaMemcpy(f, each_var[1], each_var[2], each_var[0], each_var[3], 1)
        else:
            tc_gen_code_helper_cudaMemcpy(f, each_var[1], each_var[2], each_var[0], each_var[3], 1)
    
    #
    f.write("\n")

#
#   void pre_CUDA_Malloc()
#
def tc_gen_code_pre_CUDA_Malloc(f, l_cuda_malloc, l_t3_parameters, l_t2_parameters, l_v2_parameters, l_device_dynamic, l_internal_idx):
    #
    f.write("\n")
    f.write("// created by tc_gen_code_pre_CUDA_Malloc()\n")
    f.write("void pre_CUDA_Malloc()\n")
    f.write("{\n")

    # cudaMalloc
    f.write("\t// cudaMalloc((void**) &var, sizeof(type) * size); provided by tc_gen_code_helper_cudaMalloc()\n")
    for each_var in l_cuda_malloc:
        tc_gen_code_helper_cudaMalloc(f, each_var[0], each_var[1], each_var[2])
    f.write("\n")

    # cudaMemcpy
    f.write("\t// cudaMemcpy(dest, src, sizeof(type) * size, option); provided by tc_gen_code_helper_cudaMemcpy()\n")
    for host_device in l_device_dynamic:
        tc_gen_code_helper_cudaMemcpy(f, host_device[1], host_device[2], host_device[0], host_device[3], 1)

    # if the number of internal indices is equals to or greater than 2,
    if len(l_internal_idx) > 1:
        size_internal   = ""
        idx_count       = 0
        for each_idx in l_internal_idx:
            if idx_count == 0:
                size_internal = "SIZE_IDX_" + each_idx.capitalize()
            else:
                size_internal = size_internal + " * SIZE_IDX_" + each_idx.capitalize()
            idx_count = idx_count + 1

        f.write("\n")
        f.write("\t// cudaMemcpyToSymbol(dest, src, sizeof(type) * size); provided by tc_gen_code_helper_cudaMemcpyToSymbol()\n")
        tc_gen_code_helper_cudaMemcpyToSymbol(f, "const_internal_t2_1_offset", "h_internal_t2_1_offset", "int", size_internal)
        tc_gen_code_helper_cudaMemcpyToSymbol(f, "const_internal_v2_1_offset", "h_internal_v2_1_offset", "int", size_internal)

        f.write("\n")
        f.write("\t// cudaMemcpy(dest, src, sizeof(type) * size, option); provided by tc_gen_code_helper_cudaMemcpy()\n")
        #f.write("\tint* dev_internal_offset_" + + ";\n")
    #
    f.write("}\n")

#
#
#
def tc_gen_code_helper_cudaMemcpyToSymbol(f, dest, src, type, size):
    f.write("\tcudaMemcpyToSymbol(")
    f.write(dest)
    f.write(", ")
    f.write(src)
    f.write(", sizeof(")
    f.write(type)
    f.write(") * ")
    f.write(size)
    f.write(");\n")

#
#   (file, var-name, var-type, var-size)
#
def tc_gen_code_helper_cudaMalloc(f, var, type, size):
    f.write("\tcudaMalloc((void**) &")
    f.write(var)
    f.write(", sizeof(")
    f.write(type)
    f.write(") * ")
    f.write(size)
    f.write(");\n")

#
#   (file, dest-name, src-name, var-type, var-size, copy-type)
#
def tc_gen_code_helper_cudaMemcpy(f, dest, src, type, size, option):
    f.write("\tcudaMemcpy(")
    f.write(dest)
    f.write(", ")
    f.write(src)
    f.write(", sizeof(")
    f.write(type)
    f.write(") * ")
    f.write(size)
    if option == 1:
        f.write(", cudaMemcpyHostToDevice);\n")
    else:
        f.write(", cudaMemcpyDeviceToHost);\n")
