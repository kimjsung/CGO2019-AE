#
#	Tesla V100: sm_70
#	Tesla P100: sm_60
#	Tesla K80:  sm_37
#	Tesla K40:  sm_35
#
CUDA_ARCH=sm_60
DRIVER_TCCG=drivers/tccg
DRIVER_FB=drivers/tccg-float
DRIVER_CCSDT=drivers/ccsd_t

#
#	TCCG: One of Inputs or Both of Inputs has 1 External Indices (The Others)	// 19: 1-19
#
k_tccg_01: $(DRIVER_TCCG)/code_main_tccg_01.c temp__01.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_01.c temp__01.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_02: $(DRIVER_TCCG)/code_main_tccg_02.c temp__02.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_02.c temp__02.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_03: $(DRIVER_TCCG)/code_main_tccg_03.c temp__03.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_03.c temp__03.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_04: $(DRIVER_TCCG)/code_main_tccg_04.c temp__04.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_04.c temp__04.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_05: $(DRIVER_TCCG)/code_main_tccg_05.c temp__05.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_05.c temp__05.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_06: $(DRIVER_TCCG)/code_main_tccg_06.c temp__06.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_06.c temp__06.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_07: $(DRIVER_TCCG)/code_main_tccg_07.c temp__07.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_07.c temp__07.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_08: $(DRIVER_TCCG)/code_main_tccg_08.c temp__08.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_08.c temp__08.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_09: $(DRIVER_TCCG)/code_main_tccg_09.c temp__09.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_09.c temp__09.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_10: $(DRIVER_TCCG)/code_main_tccg_10.c temp__10.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_10.c temp__10.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_11: $(DRIVER_TCCG)/code_main_tccg_11.c temp__11.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_11.c temp__11.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_12: $(DRIVER_TCCG)/code_main_tccg_12.c temp__12.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_12.c temp__12.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_13: $(DRIVER_TCCG)/code_main_tccg_13.c temp__13.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_13.c temp__13.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_14: $(DRIVER_TCCG)/code_main_tccg_14.c temp__14.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_14.c temp__14.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_15: $(DRIVER_TCCG)/code_main_tccg_15.c temp__15.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_15.c temp__15.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_16: $(DRIVER_TCCG)/code_main_tccg_16.c temp__16.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_16.c temp__16.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_17: $(DRIVER_TCCG)/code_main_tccg_17.c temp__17.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_17.c temp__17.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_18: $(DRIVER_TCCG)/code_main_tccg_18.c temp__18.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_18.c temp__18.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_19: $(DRIVER_TCCG)/code_main_tccg_19.c temp__19.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_19.c temp__19.cu -Xptxas "-v -dlcm=cg" -o $@

#
#	TCCG: 4D = 4D * 4D	// 12: 20-30
#
k_tccg_20: $(DRIVER_TCCG)/code_main_tccg_20.c temp__20.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_20.c temp__20.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_21: $(DRIVER_TCCG)/code_main_tccg_21.c temp__21.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_21.c temp__21.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_22: $(DRIVER_TCCG)/code_main_tccg_22.c temp__22.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_22.c temp__22.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_23: $(DRIVER_TCCG)/code_main_tccg_23.c temp__23.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_23.c temp__23.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_24: $(DRIVER_TCCG)/code_main_tccg_24.c temp__24.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_24.c temp__24.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_25: $(DRIVER_TCCG)/code_main_tccg_25.c temp__25.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_25.c temp__25.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_26: $(DRIVER_TCCG)/code_main_tccg_26.c temp__26.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_26.c temp__26.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_27: $(DRIVER_TCCG)/code_main_tccg_27.c temp__27.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_27.c temp__27.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_28: $(DRIVER_TCCG)/code_main_tccg_28.c temp__28.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_28.c temp__28.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_29: $(DRIVER_TCCG)/code_main_tccg_29.c temp__29.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_29.c temp__29.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_30: $(DRIVER_TCCG)/code_main_tccg_30.c temp__30.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_30.c temp__30.cu -Xptxas "-v -dlcm=cg" -o $@

#
#	TCCG: 6D = 4D * 4D	// 18: 31-48
#
k_tccg_31: $(DRIVER_TCCG)/code_main_tccg_31.c temp__31.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_31.c temp__31.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_32: $(DRIVER_TCCG)/code_main_tccg_32.c temp__32.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_32.c temp__32.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_33: $(DRIVER_TCCG)/code_main_tccg_33.c temp__33.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_33.c temp__33.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_34: $(DRIVER_TCCG)/code_main_tccg_34.c temp__34.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_34.c temp__34.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_35: $(DRIVER_TCCG)/code_main_tccg_35.c temp__35.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_35.c temp__35.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_36: $(DRIVER_TCCG)/code_main_tccg_36.c temp__36.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_36.c temp__36.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_37: $(DRIVER_TCCG)/code_main_tccg_37.c temp__37.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_37.c temp__37.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_38: $(DRIVER_TCCG)/code_main_tccg_38.c temp__38.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_38.c temp__38.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_39: $(DRIVER_TCCG)/code_main_tccg_39.c temp__39.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_39.c temp__39.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_40: $(DRIVER_TCCG)/code_main_tccg_40.c temp__40.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_40.c temp__40.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_41: $(DRIVER_TCCG)/code_main_tccg_41.c temp__41.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_41.c temp__41.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_42: $(DRIVER_TCCG)/code_main_tccg_42.c temp__42.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_42.c temp__42.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_43: $(DRIVER_TCCG)/code_main_tccg_44.c temp__43.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_43.c temp__43.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_44: $(DRIVER_TCCG)/code_main_tccg_44.c temp__44.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_44.c temp__44.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_45: $(DRIVER_TCCG)/code_main_tccg_45.c temp__45.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_45.c temp__45.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_46: $(DRIVER_TCCG)/code_main_tccg_46.c temp__46.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_46.c temp__46.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_47: $(DRIVER_TCCG)/code_main_tccg_47.c temp__47.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_47.c temp__47.cu -Xptxas "-v -dlcm=cg" -o $@

k_tccg_48: $(DRIVER_TCCG)/code_main_tccg_48.c temp__48.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_TCCG)/code_main_tccg_48.c temp__48.cu -Xptxas "-v -dlcm=cg" -o $@

#
#	floats
#
k_fb_40: $(DRIVER_FB)/code_main_tccg_40.c temp__40.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_40.c temp__40.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_41: $(DRIVER_FB)/code_main_tccg_41.c temp__41.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_41.c temp__41.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_42: $(DRIVER_FB)/code_main_tccg_42.c temp__42.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_42.c temp__42.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_43: $(DRIVER_FB)/code_main_tccg_44.c temp__43.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_43.c temp__43.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_44: $(DRIVER_FB)/code_main_tccg_44.c temp__44.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_44.c temp__44.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_45: $(DRIVER_FB)/code_main_tccg_45.c temp__45.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_45.c temp__45.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_46: $(DRIVER_FB)/code_main_tccg_46.c temp__46.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_46.c temp__46.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_47: $(DRIVER_FB)/code_main_tccg_47.c temp__47.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_47.c temp__47.cu -Xptxas "-v -dlcm=cg" -o $@

k_fb_48: $(DRIVER_FB)/code_main_tccg_48.c temp__48.cu
	nvcc -O3 -std=c++11 -arch=$(CUDA_ARCH) --use_fast_math $(DRIVER_FB)/code_main_tccg_48.c temp__48.cu -Xptxas "-v -dlcm=cg" -o $@

#
#
#
clean:
	rm k_tccg_* k_fb_*
