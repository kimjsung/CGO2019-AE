//
//	Sample Code:
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void pre_Initializing_Input_Tensors();
void post_Correctness();

int main(int argc, char** argv)
{
	// for sd2
	double *host_t3, *host_t3_chk;
	double *host_t2_1, *host_t2_2, *host_t2_3, *host_t2_4, *host_t2_5, *host_t2_6, *host_t2_7, *host_t2_8, *host_t2_9;
	double *host_v2_1, *host_v2_2, *host_v2_3, *host_v2_4, *host_v2_5, *host_v2_6, *host_v2_7, *host_v2_8, *host_v2_9;
	int size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7;
	int opt_register_transpose;

	// Problem Size
	size_h3 = 16;
	size_h2 = 16;
	size_h1 = 16;
	size_p6 = 16;
	size_p5 = 16;
	size_p4 = 16;
	size_h7 = 16;
	opt_register_transpose = 1;
	
	//
	if (argc == 9)
	{
		size_h3 = atoi(argv[1]);
		size_h2 = atoi(argv[2]);
		size_h1 = atoi(argv[3]);
		size_p6 = atoi(argv[4]);
		size_p5 = atoi(argv[5]);
		size_p4 = atoi(argv[6]);
		size_h7 = atoi(argv[7]);
		opt_register_transpose = atoi(argv[8]);
	}

	printf (">>> Problem Size (h3,h2,h1,p6,p5,p4) and (h7): (%2d,%2d,%2d,%2d,%2d,%2d) and (%2d)\n", size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7);
	printf (">>> Option for Register Transpose: %2d\n", opt_register_transpose);

	int size_T3;
	int size_T2_1, size_T2_2, size_T2_3, size_T2_4, size_T2_5, size_T2_6, size_T2_7, size_T2_8, size_T2_9;
	int size_V2_1, size_V2_2, size_V2_3, size_V2_4, size_V2_5, size_V2_6, size_V2_7, size_V2_8, size_V2_9;
	int size_internal;

	//
	size_internal 	= size_h7;
	size_T3 		= size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;	

    /*
    >> sd1 <<
    1: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_1 [h7,p4,p5,h1] * v2_1 [h3,h2,p6,h7];
    2: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_2 [h7,p4,p5,h2] * v2_2 [h3,h1,p6,h7];
    3: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_3 [h7,p4,p5,h3] * v2_3 [h2,h1,p6,h7];
    4: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_4 [h7,p5,p6,h1] * v2_4 [h3,h2,p4,h7];
    5: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_5 [h7,p5,p6,h2] * v2_5 [h3,h1,p4,h7];
    6: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_6 [h7,p5,p6,h3] * v2_6 [h2,h1,p4,h7];
    7: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_7 [h7,p4,p6,h1] * v2_7 [h3,h2,p5,h7];
    8: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_8 [h7,p4,p6,h2] * v2_8 [h3,h1,p5,h7];
    9: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_8 [h7,p4,p6,h3] * v2_9 [h2,h1,p5,h7];
    */

	//
	size_T2_1 = size_h7 * size_p4 * size_p5 * size_h1;
	size_V2_1 = size_h3 * size_h2 * size_p6 * size_h7;
	size_T2_2 = size_h7 * size_p4 * size_p5 * size_h2;
	size_V2_2 = size_h3 * size_h1 * size_p6 * size_h7;
	size_T2_3 = size_h7 * size_p4 * size_p5 * size_h3;
	size_V2_3 = size_h2 * size_h1 * size_p6 * size_h7;
	size_T2_4 = size_h7 * size_p5 * size_p6 * size_h1;
	size_V2_4 = size_h3 * size_h2 * size_p4 * size_h7;
	size_T2_5 = size_h7 * size_p5 * size_p6 * size_h2;
	size_V2_5 = size_h3 * size_h1 * size_p4 * size_h7;
	size_T2_6 = size_h7 * size_p5 * size_p6 * size_h3;
	size_V2_6 = size_h2 * size_h1 * size_p4 * size_h7;
	size_T2_7 = size_h7 * size_p4 * size_p6 * size_h1;
	size_V2_7 = size_h3 * size_h2 * size_p5 * size_h7;
	size_T2_8 = size_h7 * size_p4 * size_p6 * size_h2;
	size_V2_8 = size_h3 * size_h1 * size_p5 * size_h7;
	size_T2_9 = size_h7 * size_p4 * size_p6 * size_h3;
	size_V2_9 = size_h2 * size_h1 * size_p5 * size_h7;

	host_t3 		= (double*)malloc(sizeof(double) * size_T3);
	host_t3_chk 	= (double*)malloc(sizeof(double) * size_T3);
	host_t2_1 		= (double*)malloc(sizeof(double) * size_T2_1);
	host_t2_2 		= (double*)malloc(sizeof(double) * size_T2_2);
	host_t2_3 		= (double*)malloc(sizeof(double) * size_T2_3);
	host_t2_4 		= (double*)malloc(sizeof(double) * size_T2_4);
	host_t2_5 		= (double*)malloc(sizeof(double) * size_T2_5);
	host_t2_6 		= (double*)malloc(sizeof(double) * size_T2_6);
	host_t2_7 		= (double*)malloc(sizeof(double) * size_T2_7);
	host_t2_8 		= (double*)malloc(sizeof(double) * size_T2_8);
	host_t2_9 		= (double*)malloc(sizeof(double) * size_T2_9);

	host_v2_1 		= (double*)malloc(sizeof(double) * size_V2_1);
	host_v2_2 		= (double*)malloc(sizeof(double) * size_V2_2);
	host_v2_3 		= (double*)malloc(sizeof(double) * size_V2_3);
	host_v2_4 		= (double*)malloc(sizeof(double) * size_V2_4);
	host_v2_5 		= (double*)malloc(sizeof(double) * size_V2_5);
	host_v2_6 		= (double*)malloc(sizeof(double) * size_V2_6);
	host_v2_7 		= (double*)malloc(sizeof(double) * size_V2_7);
	host_v2_8 		= (double*)malloc(sizeof(double) * size_V2_8);
	host_v2_9 		= (double*)malloc(sizeof(double) * size_V2_9);

	printf ("==========================================================================================================\n");
	printf (" >>> %s <<<\n", __func__);
	printf ("   T3: %'12d\n", size_T3);
	printf (" T2_1: %'12d, V2_1: %'12d\n", size_T2_1, size_V2_1);
	printf (" T2_2: %'12d, V2_2: %'12d\n", size_T2_2, size_V2_2);
	printf (" T2_3: %'12d, V2_3: %'12d\n", size_T2_3, size_V2_3);
	printf (" T2_4: %'12d, V2_4: %'12d\n", size_T2_4, size_V2_4);
	printf (" T2_5: %'12d, V2_5: %'12d\n", size_T2_5, size_V2_5);
	printf (" T2_6: %'12d, V2_6: %'12d\n", size_T2_6, size_V2_6);
	printf (" T2_7: %'12d, V2_7: %'12d\n", size_T2_7, size_V2_7);
	printf (" T2_8: %'12d, V2_8: %'12d\n", size_T2_8, size_V2_8);
	printf (" T2_9: %'12d, V2_9: %'12d\n", size_T2_9, size_V2_9);
	printf ("==========================================================================================================\n");
	
	//
	// Initialze "1" Output and "2 x 9" Inputs
	pre_Initializing_Input_Tensors(host_t3, host_t3_chk,
								host_t2_1, host_v2_1,
								host_t2_2, host_v2_2,
								host_t2_3, host_v2_3,
								host_t2_4, host_v2_4,
								host_t2_5, host_v2_5,
								host_t2_6, host_v2_6,
								host_t2_7, host_v2_7,
								host_t2_8, host_v2_8,
								host_t2_9, host_v2_9, 
								size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7);
	// Run the Kernels
	sd_t_d2_fusion_(size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
					host_t3,
					host_t2_9, host_v2_9,
					1,
					opt_register_transpose);

	// Correctness-Check
	post_Correctness(host_t3, host_t3_chk, 
					host_t2_1, host_v2_1,
					host_t2_2, host_v2_2,
					host_t2_3, host_v2_3,
					host_t2_4, host_v2_4,
					host_t2_5, host_v2_5,
					host_t2_6, host_v2_6,
					host_t2_7, host_v2_7,
					host_t2_8, host_v2_8,
					host_t2_9, host_v2_9,
					1, 1, 1, 1, 1, 1, 1, 1, 1, 
					size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7);

	// Free
	free(host_t3); 		free(host_t3_chk);
	free(host_t2_1);	free(host_t2_2);	free(host_t2_3);	free(host_t2_4);	free(host_t2_5);	free(host_t2_6);	free(host_t2_7);	free(host_t2_8);	free(host_t2_9);
	free(host_v2_1);	free(host_v2_2);	free(host_v2_3);	free(host_v2_4);	free(host_v2_5);	free(host_v2_6);	free(host_v2_7); 	free(host_v2_8);	free(host_v2_9);

	return 0;
}

// Initialize t3 (t3_temp), 9 t2 and 9 v2.
void pre_Initializing_Input_Tensors(double* h_t3, 	 double* h_t3_chk, 
									double* h_t2_1,  double* h_v2_1,
									double* h_t2_2,  double* h_v2_2,
									double* h_t2_3,  double* h_v2_3,
									double* h_t2_4,  double* h_v2_4,
									double* h_t2_5,  double* h_v2_5,
									double* h_t2_6,  double* h_v2_6,
									double* h_t2_7,  double* h_v2_7,
									double* h_t2_8,  double* h_v2_8,
									double* h_t2_9,  double* h_v2_9,
									int size_idx_h3, int size_idx_h2, int size_idx_h1, 
								    int size_idx_p6, int size_idx_p5, int size_idx_p4, int size_idx_h7)
{
	int size_T3;
	int size_T2_1, size_T2_2, size_T2_3, size_T2_4, size_T2_5, size_T2_6, size_T2_7, size_T2_8, size_T2_9;
	int size_V2_1, size_V2_2, size_V2_3, size_V2_4, size_V2_5, size_V2_6, size_V2_7, size_V2_8, size_V2_9;
	int size_internal;

	//
	size_internal 	= size_idx_h7;
	size_T3 		= size_idx_h3 * size_idx_h2 * size_idx_h1 * size_idx_p6 * size_idx_p5 * size_idx_p4;

    /*
    >> sd1 <<
    1: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_1 [h7,p4,p5,h1] * v2_1 [h3,h2,p6,h7];
    2: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_2 [h7,p4,p5,h2] * v2_2 [h3,h1,p6,h7];
    3: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_3 [h7,p4,p5,h3] * v2_3 [h2,h1,p6,h7];
    4: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_4 [h7,p5,p6,h1] * v2_4 [h3,h2,p4,h7];
    5: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_5 [h7,p5,p6,h2] * v2_5 [h3,h1,p4,h7];
    6: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_6 [h7,p5,p6,h3] * v2_6 [h2,h1,p4,h7];
    7: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_7 [h7,p4,p6,h1] * v2_7 [h3,h2,p5,h7];
    8: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_8 [h7,p4,p6,h2] * v2_8 [h3,h1,p5,h7];
    9: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_8 [h7,p4,p6,h3] * v2_9 [h2,h1,p5,h7];
    */
	//
	size_T2_1 = size_idx_h7 * size_idx_p4 * size_idx_p5 * size_idx_h1;
	size_V2_1 = size_idx_h3 * size_idx_h2 * size_idx_p6 * size_idx_h7;
	size_T2_2 = size_idx_h7 * size_idx_p4 * size_idx_p5 * size_idx_h2;
	size_V2_2 = size_idx_h3 * size_idx_h1 * size_idx_p6 * size_idx_h7;
	size_T2_3 = size_idx_h7 * size_idx_p4 * size_idx_p5 * size_idx_h3;
	size_V2_3 = size_idx_h2 * size_idx_h1 * size_idx_p6 * size_idx_h7;
	size_T2_4 = size_idx_h7 * size_idx_p5 * size_idx_p6 * size_idx_h1;
	size_V2_4 = size_idx_h3 * size_idx_h2 * size_idx_p4 * size_idx_h7;
	size_T2_5 = size_idx_h7 * size_idx_p5 * size_idx_p6 * size_idx_h2;
	size_V2_5 = size_idx_h3 * size_idx_h1 * size_idx_p4 * size_idx_h7;
	size_T2_6 = size_idx_h7 * size_idx_p5 * size_idx_p6 * size_idx_h3;
	size_V2_6 = size_idx_h2 * size_idx_h1 * size_idx_p4 * size_idx_h7;
	size_T2_7 = size_idx_h7 * size_idx_p4 * size_idx_p6 * size_idx_h1;
	size_V2_7 = size_idx_h3 * size_idx_h2 * size_idx_p5 * size_idx_h7;
	size_T2_8 = size_idx_h7 * size_idx_p4 * size_idx_p6 * size_idx_h2;
	size_V2_8 = size_idx_h3 * size_idx_h1 * size_idx_p5 * size_idx_h7;
	size_T2_9 = size_idx_h7 * size_idx_p4 * size_idx_p6 * size_idx_h3;
	size_V2_9 = size_idx_h2 * size_idx_h1 * size_idx_p5 * size_idx_h7;

	// t3
	int i, j;
	for (i = 0; i < size_T3; i++)
	{
		h_t3[i] 	= 0.0;
		h_t3_chk[i] = 0.0;
	}

	for (j = 0; j < size_T2_1; j++)
	{
		h_t2_1[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_1; j++)
	{
		h_v2_1[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_2; j++)
	{
		h_t2_2[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_2; j++)
	{
		h_v2_2[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_3; j++)
	{
		h_t2_3[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_3; j++)
	{
		h_v2_3[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_4; j++)
	{
		h_t2_4[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_4; j++)
	{
		h_v2_4[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_5; j++)
	{
		h_t2_5[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_5; j++)
	{
		h_v2_5[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_6; j++)
	{
		h_t2_6[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_6; j++)
	{
		h_v2_6[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_7; j++)
	{
		h_t2_7[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_7; j++)
	{
		h_v2_7[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_8; j++)
	{
		h_t2_8[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_8; j++)
	{
		h_v2_8[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_T2_9; j++)
	{
		h_t2_9[j] = ((double)rand() / RAND_MAX);
	}

	for (j = 0; j < size_V2_9; j++)
	{
		h_v2_9[j] = ((double)rand() / RAND_MAX);
	}

	printf ("==========================================================================================================\n");
	printf (" >>> %s <<<\n", __func__);
	printf ("   T3: %'12d\n", size_T3);
	printf (" T2_1: %'12d, V2_1: %'12d\n", size_T2_1, size_V2_1);
	printf (" T2_2: %'12d, V2_2: %'12d\n", size_T2_2, size_V2_2);
	printf (" T2_3: %'12d, V2_3: %'12d\n", size_T2_3, size_V2_3);
	printf (" T2_4: %'12d, V2_4: %'12d\n", size_T2_4, size_V2_4);
	printf (" T2_5: %'12d, V2_5: %'12d\n", size_T2_5, size_V2_5);
	printf (" T2_6: %'12d, V2_6: %'12d\n", size_T2_6, size_V2_6);
	printf (" T2_7: %'12d, V2_7: %'12d\n", size_T2_7, size_V2_7);
	printf (" T2_8: %'12d, V2_8: %'12d\n", size_T2_8, size_V2_8);
	printf (" T2_9: %'12d, V2_9: %'12d\n", size_T2_9, size_V2_9);
	printf ("==========================================================================================================\n");
}

//
void post_Correctness(double* h_t3, double* h_t3_chk, 
									double* h_t2_1,  double* h_v2_1,
									double* h_t2_2,  double* h_v2_2,
									double* h_t2_3,  double* h_v2_3,
									double* h_t2_4,  double* h_v2_4,
									double* h_t2_5,  double* h_v2_5,
									double* h_t2_6,  double* h_v2_6,
									double* h_t2_7,  double* h_v2_7,
									double* h_t2_8,  double* h_v2_8,
									double* h_t2_9,  double* h_v2_9,
									int kernel_1, int kernel_2, int kernel_3,
									int kernel_4, int kernel_5, int kernel_6,
									int kernel_7, int kernel_8, int kernel_9,
									int size_idx_h3, int size_idx_h2, int size_idx_h1, 
									int size_idx_p6, int size_idx_p5, int size_idx_p4, int size_idx_h7)
{
	int SIZE_IDX_H3;
	int SIZE_IDX_H2;
	int SIZE_IDX_H1;
	int SIZE_IDX_P6;
	int SIZE_IDX_P5;
	int SIZE_IDX_P4;
	int SIZE_IDX_H7;

	int STR_SD2_T3_H3;
	int STR_SD2_T3_H2;// STR_SD2_T3_H3 * SIZE_IDX_H3
	int STR_SD2_T3_H1;// STR_SD2_T3_H2 * SIZE_IDX_H2
	int STR_SD2_T3_P6;// STR_SD2_T3_H1 * SIZE_IDX_H1
	int STR_SD2_T3_P5;// STR_SD2_T3_P6 * SIZE_IDX_P6
	int STR_SD2_T3_P4;// STR_SD2_T3_P5 * SIZE_IDX_P5

	// t2 for inputs
	int STR_SD2_T2_4_H7;// 1
	int STR_SD2_T2_4_P5;// STR_SD2_T2_4_H7 * SIZE_IDX_H7
	int STR_SD2_T2_4_P6;// STR_SD2_T2_4_P5 * SIZE_IDX_P5
	int STR_SD2_T2_4_H1;// STR_SD2_T2_4_P6 * SIZE_IDX_P6

	// v2 for inputs
	int STR_SD2_V2_4_H3;// 1
	int STR_SD2_V2_4_H2;// STR_SD2_V2_4_H3 * SIZE_IDX_H3
	int STR_SD2_V2_4_P4;// STR_SD2_V2_4_H2 * SIZE_IDX_H2
	int STR_SD2_V2_4_H7;// STR_SD2_V2_4_P4 * SIZE_IDX_P4

	// t2 for inputs
	int STR_SD2_T2_5_H7;// 1
	int STR_SD2_T2_5_P5;// STR_SD2_T2_5_H7 * SIZE_IDX_H7
	int STR_SD2_T2_5_P6;// STR_SD2_T2_5_P5 * SIZE_IDX_P5
	int STR_SD2_T2_5_H2;// STR_SD2_T2_5_P6 * SIZE_IDX_P6

	// v2 for inputs
	int STR_SD2_V2_5_H3;// 1
	int STR_SD2_V2_5_H1;// STR_SD2_V2_5_H3 * SIZE_IDX_H3
	int STR_SD2_V2_5_P4;// STR_SD2_V2_5_H1 * SIZE_IDX_H1
	int STR_SD2_V2_5_H7;// STR_SD2_V2_5_P4 * SIZE_IDX_P4

	// t2 for inputs
	int STR_SD2_T2_6_H7;// 1
	int STR_SD2_T2_6_P5;// STR_SD2_T2_6_H7 * SIZE_IDX_H7
	int STR_SD2_T2_6_P6;// STR_SD2_T2_6_P5 * SIZE_IDX_P5
	int STR_SD2_T2_6_H3;// STR_SD2_T2_6_P6 * SIZE_IDX_P6

	// v2 for inputs
	int STR_SD2_V2_6_H2;// 1
	int STR_SD2_V2_6_H1;// STR_SD2_V2_6_H2 * SIZE_IDX_H2
	int STR_SD2_V2_6_P4;// STR_SD2_V2_6_H1 * SIZE_IDX_H1
	int STR_SD2_V2_6_H7;// STR_SD2_V2_6_P4 * SIZE_IDX_P4

	// t2 for inputs
	int STR_SD2_T2_7_H7;// 1
	int STR_SD2_T2_7_P4;// STR_SD2_T2_7_H7 * SIZE_IDX_H7
	int STR_SD2_T2_7_P6;// STR_SD2_T2_7_P4 * SIZE_IDX_P4
	int STR_SD2_T2_7_H1;// STR_SD2_T2_7_P6 * SIZE_IDX_P6

	// v2 for inputs
	int STR_SD2_V2_7_H3;// 1
	int STR_SD2_V2_7_H2;// STR_SD2_V2_7_H3 * SIZE_IDX_H3
	int STR_SD2_V2_7_P5;// STR_SD2_V2_7_H2 * SIZE_IDX_H2
	int STR_SD2_V2_7_H7;// STR_SD2_V2_7_P5 * SIZE_IDX_P5

	// t2 for inputs
	int STR_SD2_T2_8_H7;// 1
	int STR_SD2_T2_8_P4;// STR_SD2_T2_8_H7 * SIZE_IDX_H7
	int STR_SD2_T2_8_P6;// STR_SD2_T2_8_P4 * SIZE_IDX_P4
	int STR_SD2_T2_8_H2;// STR_SD2_T2_8_P6 * SIZE_IDX_P6

	// v2 for inputs
	int STR_SD2_V2_8_H3;// 1
	int STR_SD2_V2_8_H1;// STR_SD2_V2_8_H3 * SIZE_IDX_H3
	int STR_SD2_V2_8_P5;// STR_SD2_V2_8_H1 * SIZE_IDX_H1
	int STR_SD2_V2_8_H7;// STR_SD2_V2_8_P5 * SIZE_IDX_P5

	// t2 for inputs
	int STR_SD2_T2_9_H7;// 1
	int STR_SD2_T2_9_P4;// STR_SD2_T2_9_H7 * SIZE_IDX_H7
	int STR_SD2_T2_9_P6;// STR_SD2_T2_9_P4 * SIZE_IDX_P4
	int STR_SD2_T2_9_H3;// STR_SD2_T2_9_P6 * SIZE_IDX_P6

	// v2 for inputs
	int STR_SD2_V2_9_H2;// 1
	int STR_SD2_V2_9_H1;// STR_SD2_V2_9_H2 * SIZE_IDX_H2
	int STR_SD2_V2_9_P5;// STR_SD2_V2_9_H1 * SIZE_IDX_H1
	int STR_SD2_V2_9_H7;// STR_SD2_V2_9_P5 * SIZE_IDX_P5

	// t2 for inputs
	int STR_SD2_T2_1_H7;// 1
	int STR_SD2_T2_1_P4;// STR_SD2_T2_1_H7 * SIZE_IDX_H7
	int STR_SD2_T2_1_P5;// STR_SD2_T2_1_P4 * SIZE_IDX_P4
	int STR_SD2_T2_1_H1;// STR_SD2_T2_1_P5 * SIZE_IDX_P5

	// v2 for inputs
	int STR_SD2_V2_1_H3;// 1
	int STR_SD2_V2_1_H2;// STR_SD2_V2_1_H3 * SIZE_IDX_H3
	int STR_SD2_V2_1_P6;// STR_SD2_V2_1_H2 * SIZE_IDX_H2
	int STR_SD2_V2_1_H7;// STR_SD2_V2_1_P6 * SIZE_IDX_P6

	// t2 for inputs
	int STR_SD2_T2_2_H7;// 1
	int STR_SD2_T2_2_P4;// STR_SD2_T2_2_H7 * SIZE_IDX_H7
	int STR_SD2_T2_2_P5;// STR_SD2_T2_2_P4 * SIZE_IDX_P4
	int STR_SD2_T2_2_H2;// STR_SD2_T2_2_P5 * SIZE_IDX_P5

	// v2 for inputs
	int STR_SD2_V2_2_H3;// 1
	int STR_SD2_V2_2_H1;// STR_SD2_V2_2_H3 * SIZE_IDX_H3
	int STR_SD2_V2_2_P6;// STR_SD2_V2_2_H1 * SIZE_IDX_H1
	int STR_SD2_V2_2_H7;// STR_SD2_V2_2_P6 * SIZE_IDX_P6

	// t2 for inputs
	int STR_SD2_T2_3_H7;// 1
	int STR_SD2_T2_3_P4;// STR_SD2_T2_3_H7 * SIZE_IDX_H7
	int STR_SD2_T2_3_P5;// STR_SD2_T2_3_P4 * SIZE_IDX_P4
	int STR_SD2_T2_3_H3;// STR_SD2_T2_3_P5 * SIZE_IDX_P5

	// v2 for inputs
	int STR_SD2_V2_3_H2;// 1
	int STR_SD2_V2_3_H1;// STR_SD2_V2_3_H2 * SIZE_IDX_H2
	int STR_SD2_V2_3_P6;// STR_SD2_V2_3_H1 * SIZE_IDX_H1
	int STR_SD2_V2_3_H7;// STR_SD2_V2_3_P6 * SIZE_IDX_P6

	// Indices
	SIZE_IDX_H3 = size_idx_h3;
	SIZE_IDX_H2 = size_idx_h2;
	SIZE_IDX_H1 = size_idx_h1;
	SIZE_IDX_P6 = size_idx_p6;
	SIZE_IDX_P5 = size_idx_p5;
	SIZE_IDX_P4 = size_idx_p4;
	SIZE_IDX_H7 = size_idx_h7;

	// t3
	STR_SD2_T3_H3 = 1;
	STR_SD2_T3_H2 = STR_SD2_T3_H3 * SIZE_IDX_H3;
	STR_SD2_T3_H1 = STR_SD2_T3_H2 * SIZE_IDX_H2;
	STR_SD2_T3_P6 = STR_SD2_T3_H1 * SIZE_IDX_H1;
	STR_SD2_T3_P5 = STR_SD2_T3_P6 * SIZE_IDX_P6;
	STR_SD2_T3_P4 = STR_SD2_T3_P5 * SIZE_IDX_P5;

    /*
    >> sd1 <<
    1: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_1 [h7,p4,p5,h1] * v2_1 [h3,h2,p6,h7];
    2: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_2 [h7,p4,p5,h2] * v2_2 [h3,h1,p6,h7];
    3: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_3 [h7,p4,p5,h3] * v2_3 [h2,h1,p6,h7];
    4: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_4 [h7,p5,p6,h1] * v2_4 [h3,h2,p4,h7];
    5: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_5 [h7,p5,p6,h2] * v2_5 [h3,h1,p4,h7];
    6: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_6 [h7,p5,p6,h3] * v2_6 [h2,h1,p4,h7];
    7: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_7 [h7,p4,p6,h1] * v2_7 [h3,h2,p5,h7];
    8: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] -= sum(h7,16) * t2_8 [h7,p4,p6,h2] * v2_8 [h3,h1,p5,h7];
    9: t3 [h3,16,h2,16,h1,16,p6,16,p5,16,p4,16] += sum(h7,16) * t2_8 [h7,p4,p6,h3] * v2_9 [h2,h1,p5,h7];
    */

	// t2 for inputs
	STR_SD2_T2_4_H7 = 1;
	STR_SD2_T2_4_P5 = STR_SD2_T2_4_H7 * SIZE_IDX_H7;
	STR_SD2_T2_4_P6 = STR_SD2_T2_4_P5 * SIZE_IDX_P5;
	STR_SD2_T2_4_H1 = STR_SD2_T2_4_P6 * SIZE_IDX_P6;

	// v2 for inputs
	STR_SD2_V2_4_H3 = 1;
	STR_SD2_V2_4_H2 = STR_SD2_V2_4_H3 * SIZE_IDX_H3;
	STR_SD2_V2_4_P4 = STR_SD2_V2_4_H2 * SIZE_IDX_H2;
	STR_SD2_V2_4_H7 = STR_SD2_V2_4_P4 * SIZE_IDX_P4;

	// t2 for inputs
	STR_SD2_T2_5_H7 = 1;
	STR_SD2_T2_5_P5 = STR_SD2_T2_5_H7 * SIZE_IDX_H7;
	STR_SD2_T2_5_P6 = STR_SD2_T2_5_P5 * SIZE_IDX_P5;
	STR_SD2_T2_5_H2 = STR_SD2_T2_5_P6 * SIZE_IDX_P6;

	// v2 for inputs
	STR_SD2_V2_5_H3 = 1;
	STR_SD2_V2_5_H1 = STR_SD2_V2_5_H3 * SIZE_IDX_H3;
	STR_SD2_V2_5_P4 = STR_SD2_V2_5_H1 * SIZE_IDX_H1;
	STR_SD2_V2_5_H7 = STR_SD2_V2_5_P4 * SIZE_IDX_P4;

	// t2 for inputs
	STR_SD2_T2_6_H7 = 1;
	STR_SD2_T2_6_P5 = STR_SD2_T2_6_H7 * SIZE_IDX_H7;
	STR_SD2_T2_6_P6 = STR_SD2_T2_6_P5 * SIZE_IDX_P5;
	STR_SD2_T2_6_H3 = STR_SD2_T2_6_P6 * SIZE_IDX_P6;

	// v2 for inputs
	STR_SD2_V2_6_H2 = 1;
	STR_SD2_V2_6_H1 = STR_SD2_V2_6_H2 * SIZE_IDX_H2;
	STR_SD2_V2_6_P4 = STR_SD2_V2_6_H1 * SIZE_IDX_H1;
	STR_SD2_V2_6_H7 = STR_SD2_V2_6_P4 * SIZE_IDX_P4;

	// t2 for inputs
	STR_SD2_T2_7_H7 = 1;
	STR_SD2_T2_7_P4 = STR_SD2_T2_7_H7 * SIZE_IDX_H7;
	STR_SD2_T2_7_P6 = STR_SD2_T2_7_P4 * SIZE_IDX_P4;
	STR_SD2_T2_7_H1 = STR_SD2_T2_7_P6 * SIZE_IDX_P6;

	// v2 for inputs
	STR_SD2_V2_7_H3 = 1;
	STR_SD2_V2_7_H2 = STR_SD2_V2_7_H3 * SIZE_IDX_H3;
	STR_SD2_V2_7_P5 = STR_SD2_V2_7_H2 * SIZE_IDX_H2;
	STR_SD2_V2_7_H7 = STR_SD2_V2_7_P5 * SIZE_IDX_P5;

	// t2 for inputs
	STR_SD2_T2_8_H7 = 1;
	STR_SD2_T2_8_P4 = STR_SD2_T2_8_H7 * SIZE_IDX_H7;
	STR_SD2_T2_8_P6 = STR_SD2_T2_8_P4 * SIZE_IDX_P4;
	STR_SD2_T2_8_H2 = STR_SD2_T2_8_P6 * SIZE_IDX_P6;

	// v2 for inputs
	STR_SD2_V2_8_H3 = 1;
	STR_SD2_V2_8_H1 = STR_SD2_V2_8_H3 * SIZE_IDX_H3;
	STR_SD2_V2_8_P5 = STR_SD2_V2_8_H1 * SIZE_IDX_H1;
	STR_SD2_V2_8_H7 = STR_SD2_V2_8_P5 * SIZE_IDX_P5;

	// t2 for inputs
	STR_SD2_T2_9_H7 = 1;
	STR_SD2_T2_9_P4 = STR_SD2_T2_9_H7 * SIZE_IDX_H7;
	STR_SD2_T2_9_P6 = STR_SD2_T2_9_P4 * SIZE_IDX_P4;
	STR_SD2_T2_9_H3 = STR_SD2_T2_9_P6 * SIZE_IDX_P6;

	// v2 for inputs
	STR_SD2_V2_9_H2 = 1;
	STR_SD2_V2_9_H1 = STR_SD2_V2_9_H2 * SIZE_IDX_H2;
	STR_SD2_V2_9_P5 = STR_SD2_V2_9_H1 * SIZE_IDX_H1;
	STR_SD2_V2_9_H7 = STR_SD2_V2_9_P5 * SIZE_IDX_P5;

	// t2 for inputs
	STR_SD2_T2_1_H7 = 1;
	STR_SD2_T2_1_P4 = STR_SD2_T2_1_H7 * SIZE_IDX_H7;
	STR_SD2_T2_1_P5 = STR_SD2_T2_1_P4 * SIZE_IDX_P4;
	STR_SD2_T2_1_H1 = STR_SD2_T2_1_P5 * SIZE_IDX_P5;;

	// v2 for inputs
	STR_SD2_V2_1_H3 = 1;
	STR_SD2_V2_1_H2 = STR_SD2_V2_1_H3 * SIZE_IDX_H3;
	STR_SD2_V2_1_P6 = STR_SD2_V2_1_H2 * SIZE_IDX_H2;
	STR_SD2_V2_1_H7 = STR_SD2_V2_1_P6 * SIZE_IDX_P6;

	// t2 for inputs
	STR_SD2_T2_2_H7 = 1;
	STR_SD2_T2_2_P4 = STR_SD2_T2_2_H7 * SIZE_IDX_H7;
	STR_SD2_T2_2_P5 = STR_SD2_T2_2_P4 * SIZE_IDX_P4;
	STR_SD2_T2_2_H2 = STR_SD2_T2_2_P5 * SIZE_IDX_P5;

	// v2 for inputs
	STR_SD2_V2_2_H3 = 1;
	STR_SD2_V2_2_H1 = STR_SD2_V2_2_H3 * SIZE_IDX_H3;
	STR_SD2_V2_2_P6 = STR_SD2_V2_2_H1 * SIZE_IDX_H1;
	STR_SD2_V2_2_H7 = STR_SD2_V2_2_P6 * SIZE_IDX_P6;

	// t2 for inputs
	STR_SD2_T2_3_H7 = 1;
	STR_SD2_T2_3_P4 = STR_SD2_T2_3_H7 * SIZE_IDX_H7;
	STR_SD2_T2_3_P5 = STR_SD2_T2_3_P4 * SIZE_IDX_P4;
	STR_SD2_T2_3_H3 = STR_SD2_T2_3_P5 * SIZE_IDX_P5;

	// v2 for inputs
	STR_SD2_V2_3_H2 = 1;
	STR_SD2_V2_3_H1 = STR_SD2_V2_3_H2 * SIZE_IDX_H2;
	STR_SD2_V2_3_P6 = STR_SD2_V2_3_H1 * SIZE_IDX_H1;
	STR_SD2_V2_3_H7 = STR_SD2_V2_3_P6 * SIZE_IDX_P6;


	long long int    tmp_ops = 0;
	int              ops     = 0;
	int t3_h3, t3_h2, t3_h1, t3_p6, t3_p5, t3_p4, t3_h7;
	for (t3_h3 = 0; t3_h3 < SIZE_IDX_H3; t3_h3++)
	for (t3_h2 = 0; t3_h2 < SIZE_IDX_H2; t3_h2++)
	for (t3_h1 = 0; t3_h1 < SIZE_IDX_H1; t3_h1++)
	for (t3_p6 = 0; t3_p6 < SIZE_IDX_P6; t3_p6++)
	for (t3_p5 = 0; t3_p5 < SIZE_IDX_P5; t3_p5++)
	for (t3_p4 = 0; t3_p4 < SIZE_IDX_P4; t3_p4++)
	{
		int tmp_r_idx = t3_h3 * STR_SD2_T3_H3 + t3_h2 * STR_SD2_T3_H2 + t3_h1 * STR_SD2_T3_H1 + t3_p6 * STR_SD2_T3_P6 + t3_p5 * STR_SD2_T3_P5 + t3_p4 * STR_SD2_T3_P4;
		for (t3_h7 = 0; t3_h7 < SIZE_IDX_H7; t3_h7++, ops = 0)
		{   
            /*
            h_t3_chk[tmp_r_idx] -= 	h_t2_4[t3_h7 * STR_SD2_T2_4_H7 + t3_p5 * STR_SD2_T2_4_P5 + t3_p6 * STR_SD2_T2_4_P6 + t3_h1 * STR_SD2_T2_4_H1] * 
                                    h_v2_4[t3_h3 * STR_SD2_V2_4_H3 + t3_h2 * STR_SD2_V2_4_H2 + t3_p4 * STR_SD2_V2_4_P4 + t3_h7 * STR_SD2_V2_4_H7];
            ops++;
            */
          /*  
            h_t3_chk[tmp_r_idx] += 	h_t2_5[t3_h7 * STR_SD2_T2_5_H7 + t3_p5 * STR_SD2_T2_5_P5 + t3_p6 * STR_SD2_T2_5_P6 + t3_h2 * STR_SD2_T2_5_H2] * 
                                    h_v2_5[t3_h3 * STR_SD2_V2_5_H3 + t3_h1 * STR_SD2_V2_5_H1 + t3_p4 * STR_SD2_V2_5_P4 + t3_h7 * STR_SD2_V2_5_H7];
            ops++;
        */
			/*
				h_t3_chk[tmp_r_idx] -= 	h_t2_6[t3_h7 * STR_SD2_T2_6_H7 + t3_p5 * STR_SD2_T2_6_P5 + t3_p6 * STR_SD2_T2_6_P6 + t3_h3 * STR_SD2_T2_6_H3] * 
										h_v2_6[t3_h2 * STR_SD2_V2_6_H2 + t3_h1 * STR_SD2_V2_6_H1 + t3_p4 * STR_SD2_V2_6_P4 + t3_h7 * STR_SD2_V2_6_H7];
				ops++;
                */
			/*
				h_t3_chk[tmp_r_idx] += 	h_t2_7[t3_h7 * STR_SD2_T2_7_H7 + t3_p4 * STR_SD2_T2_7_P4 + t3_p6 * STR_SD2_T2_7_P6 + t3_h1 * STR_SD2_T2_7_H1] * 
										h_v2_7[t3_h3 * STR_SD2_V2_7_H3 + t3_h2 * STR_SD2_V2_7_H2 + t3_p5 * STR_SD2_V2_7_P5 + t3_h7 * STR_SD2_V2_7_H7];
				ops++;
                */
			/*
				h_t3_chk[tmp_r_idx] -= 	h_t2_8[t3_h7 * STR_SD2_T2_8_H7 + t3_p4 * STR_SD2_T2_8_P4 + t3_p6 * STR_SD2_T2_8_P6 + t3_h2 * STR_SD2_T2_8_H2] * 
										h_v2_8[t3_h3 * STR_SD2_V2_8_H3 + t3_h1 * STR_SD2_V2_8_H1 + t3_p5 * STR_SD2_V2_8_P5 + t3_h7 * STR_SD2_V2_8_H7];
				ops++;

                */
			
				h_t3_chk[tmp_r_idx] += 	h_t2_9[t3_h7 * STR_SD2_T2_9_H7 + t3_p4 * STR_SD2_T2_9_P4 + t3_p6 * STR_SD2_T2_9_P6 + t3_h3 * STR_SD2_T2_9_H3] * 
										h_v2_9[t3_h2 * STR_SD2_V2_9_H2 + t3_h1 * STR_SD2_V2_9_H1 + t3_p5 * STR_SD2_V2_9_P5 + t3_h7 * STR_SD2_V2_9_H7];
				ops++;
			/*
				h_t3_chk[tmp_r_idx] -= 	h_t2_1[t3_h7 * STR_SD2_T2_1_H7 + t3_p4 * STR_SD2_T2_1_P4 + t3_p5 * STR_SD2_T2_1_P5 + t3_h1 * STR_SD2_T2_1_H1] * 
										h_v2_1[t3_h3 * STR_SD2_V2_1_H3 + t3_h2 * STR_SD2_V2_1_H2 + t3_p6 * STR_SD2_V2_1_P6 + t3_h7 * STR_SD2_V2_1_H7];
				ops++;
                */
	    /*
            h_t3_chk[tmp_r_idx] += 	h_t2_2[t3_h7 * STR_SD2_T2_2_H7 + t3_p4 * STR_SD2_T2_2_P4 + t3_p5 * STR_SD2_T2_2_P5 + t3_h2 * STR_SD2_T2_2_H2] * 
                                    h_v2_2[t3_h3 * STR_SD2_V2_2_H3 + t3_h1 * STR_SD2_V2_2_H1 + t3_p6 * STR_SD2_V2_2_P6 + t3_h7 * STR_SD2_V2_2_H7];
            ops++;
            */
/*
            if (tmp_r_idx == 0)
            {
                printf ("[%d][%2d] t2_2: %f (%d), v2_2: %f (%d)\n", tmp_r_idx, t3_h7, 
                h_t2_2[t3_h7 * STR_SD2_T2_2_H7 + t3_p4 * STR_SD2_T2_2_P4 + t3_p5 * STR_SD2_T2_2_P5 + t3_h2 * STR_SD2_T2_2_H2],
                t3_h7 * STR_SD2_T2_2_H7 + t3_p4 * STR_SD2_T2_2_P4 + t3_p5 * STR_SD2_T2_2_P5 + t3_h2 * STR_SD2_T2_2_H2,
                h_v2_2[t3_h3 * STR_SD2_V2_2_H3 + t3_h1 * STR_SD2_V2_2_H1 + t3_p6 * STR_SD2_V2_2_P6 + t3_h7 * STR_SD2_V2_2_H7],
                t3_h3 * STR_SD2_V2_2_H3 + t3_h1 * STR_SD2_V2_2_H1 + t3_p6 * STR_SD2_V2_2_P6 + t3_h7 * STR_SD2_V2_2_H7);
            }
  */        /*
				h_t3_chk[tmp_r_idx] -= 	h_t2_3[t3_h7 * STR_SD2_T2_3_H7 + t3_p4 * STR_SD2_T2_3_P4 + t3_p5 * STR_SD2_T2_3_P5 + t3_h3 * STR_SD2_T2_3_H3] * 
										h_v2_3[t3_h2 * STR_SD2_V2_3_H2 + t3_h1 * STR_SD2_V2_3_H1 + t3_p6 * STR_SD2_V2_3_P6 + t3_h7 * STR_SD2_V2_3_H7];
				ops++;
*/
			tmp_ops = tmp_ops + ops;
		}
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;
	for (i = 0; i < size_idx_h3 * size_idx_h2 * size_idx_h1 * size_idx_p6 * size_idx_p5 * size_idx_p4; i++)
	{
		double check = h_t3_chk[i] - h_t3[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			if (diff < 8)
			printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_t3_chk[i], h_t3[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_idx_h3 * size_idx_h2 * size_idx_h1 * size_idx_p6 * size_idx_p5 * size_idx_p4);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_idx_h3 * size_idx_h2 * size_idx_h1 * size_idx_p6 * size_idx_p5 * size_idx_p4);
	printf (" >>> Total Operations: %'lld\n", tmp_ops * 2);
	printf ("====================================================================================================\n");
}

