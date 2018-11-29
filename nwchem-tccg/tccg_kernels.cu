#include "header.h"
/*----------------------------------------------------------------------*
 *cc[a,b,c] += aa[b,d,a] * bb[d,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_01_kernel(int ad,int bd,int cd,int dd,int bld_aa,int dld_aa,int ald_aa,int dld_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,dl,dT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(dT=0;dT<dd;dT+=Tcomm){int dl_hi;
dl_hi = MIN(Tcomm+dT,dd)-dT;
aa_d_off=b_0*bld_aa+a_0*ald_aa;
bb_d_off=c_0*cld_bb;
if(thread_y+T1*0<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*0][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*0] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=b_1*bld_aa+a_1*ald_aa;
bb_d_off=c_1*cld_bb;
if(thread_y+T1*1<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*1][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*1] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=b_2*bld_aa+a_2*ald_aa;
bb_d_off=c_2*cld_bb;
if(thread_y+T1*2<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*2][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*2] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=b_3*bld_aa+a_3*ald_aa;
bb_d_off=c_3*cld_bb;
if(thread_y+T1*3<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*3][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*3] = bb_d[bb_d_off+d*dld_bb];
}
__syncthreads();
for(dl=0;dl<dl_hi;++dl){
a1=aa_shm[in1_idxl+T1*0][dl];
a2=aa_shm[in1_idxl+T1*1][dl];
a3=aa_shm[in1_idxl+T1*2][dl];
a4=aa_shm[in1_idxl+T1*3][dl];
b1=bb_shm[dl][in2_idxl+T2*0];
b2=bb_shm[dl][in2_idxl+T2*1];
b3=bb_shm[dl][in2_idxl+T2*2];
b4=bb_shm[dl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_01_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t bld_aa,dld_aa,ald_aa,dld_bb,cld_bb,ald_cc,bld_cc,cld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*sizeof(double);
size_aa=bd*dd*ad*sizeof(double);
size_bb=dd*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_01_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
bld_aa=1;
dld_aa=bd;
ald_aa=dd*bd;
dld_bb=1;
cld_bb=dd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
int total_x = cd;
int total_y = bd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_01_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,bld_aa,dld_aa,ald_aa,dld_bb,cld_bb,ald_cc,bld_cc,cld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_01_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_01_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c] += aa[d,c,a] * bb[b,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_02_kernel(int ad,int bd,int cd,int dd,int dld_aa,int cld_aa,int ald_aa,int bld_bb,int dld_bb,int ald_cc,int bld_cc,int cld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,dl,dT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(dT=0;dT<dd;dT+=Tcomm){int dl_hi;
dl_hi = MIN(Tcomm+dT,dd)-dT;
aa_d_off=c_0*cld_aa+a_0*ald_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*0][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*0] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=c_1*cld_aa+a_1*ald_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*1][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*1] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=c_2*cld_aa+a_2*ald_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*2][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*2] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=c_3*cld_aa+a_3*ald_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*3][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*3] = bb_d[bb_d_off+d*dld_bb];
}
__syncthreads();
for(dl=0;dl<dl_hi;++dl){
a1=aa_shm[in1_idxl+T1*0][dl];
a2=aa_shm[in1_idxl+T1*1][dl];
a3=aa_shm[in1_idxl+T1*2][dl];
a4=aa_shm[in1_idxl+T1*3][dl];
b1=bb_shm[dl][in2_idxl+T2*0];
b2=bb_shm[dl][in2_idxl+T2*1];
b3=bb_shm[dl][in2_idxl+T2*2];
b4=bb_shm[dl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_02_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t dld_aa,cld_aa,ald_aa,bld_bb,dld_bb,ald_cc,bld_cc,cld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*sizeof(double);
size_aa=dd*cd*ad*sizeof(double);
size_bb=bd*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_02_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
cld_aa=dd;
ald_aa=cd*dd;
bld_bb=1;
dld_bb=bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
int total_x = bd*1;
int total_y = cd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_02_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,dld_aa,cld_aa,ald_aa,bld_bb,dld_bb,ald_cc,bld_cc,cld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_02_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_02_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[d,b,e,a] * bb[e,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_03_kernel(int ad,int bd,int cd,int dd,int ed,int dld_aa,int bld_aa,int eld_aa,int ald_aa,int eld_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
d_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
d_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
d_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
d_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=d_0*dld_aa+b_0*bld_aa+a_0*ald_aa;
bb_d_off=c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=d_1*dld_aa+b_1*bld_aa+a_1*ald_aa;
bb_d_off=c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=d_2*dld_aa+b_2*bld_aa+a_2*ald_aa;
bb_d_off=c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=d_3*dld_aa+b_3*bld_aa+a_3*ald_aa;
bb_d_off=c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_03_cuda(int ad, int bd, int cd, int dd, int ed, double *cc, double *aa, double *bb) {
size_t stream;
size_t dld_aa,bld_aa,eld_aa,ald_aa,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=dd*bd*ed*ad*sizeof(double);
size_bb=ed*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_03_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
bld_aa=dd;
eld_aa=bd*dd;
ald_aa=ed*bd*dd;
eld_bb=1;
cld_bb=ed;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = cd;
int total_y = dd*bd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_03_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,dld_aa,bld_aa,eld_aa,ald_aa,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_03_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, double *cc, double *aa, double *bb) {
tccg_03_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[d,e,c,a] * bb[b,e]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_04_kernel(int ad,int bd,int cd,int dd,int ed,int dld_aa,int eld_aa,int cld_aa,int ald_aa,int bld_bb,int eld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_y%cd;
rest_y=rest_y/cd;
d_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_y%cd;
rest_y=rest_y/cd;
d_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_y%cd;
rest_y=rest_y/cd;
d_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_y%cd;
rest_y=rest_y/cd;
d_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=d_0*dld_aa+c_0*cld_aa+a_0*ald_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=d_1*dld_aa+c_1*cld_aa+a_1*ald_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=d_2*dld_aa+c_2*cld_aa+a_2*ald_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=d_3*dld_aa+c_3*cld_aa+a_3*ald_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_04_cuda(int ad, int bd, int cd, int dd, int ed, double *cc, double *aa, double *bb) {
size_t stream;
size_t dld_aa,eld_aa,cld_aa,ald_aa,bld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=dd*ed*cd*ad*sizeof(double);
size_bb=bd*ed*sizeof(double);
cudaFuncSetCacheConfig(tccg_04_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
eld_aa=dd;
cld_aa=ed*dd;
ald_aa=cd*ed*dd;
bld_bb=1;
eld_bb=bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*1;
int total_y = dd*cd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_04_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,dld_aa,eld_aa,cld_aa,ald_aa,bld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_04_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, double *cc, double *aa, double *bb) {
tccg_04_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[e,b,a,d] * bb[c,e]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_05_kernel(int ad,int bd,int cd,int dd,int ed,int eld_aa,int bld_aa,int ald_aa,int dld_aa,int cld_bb,int eld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
d_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
d_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
d_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
d_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=b_0*bld_aa+a_0*ald_aa+d_0*dld_aa;
bb_d_off=c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=b_1*bld_aa+a_1*ald_aa+d_1*dld_aa;
bb_d_off=c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=b_2*bld_aa+a_2*ald_aa+d_2*dld_aa;
bb_d_off=c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=b_3*bld_aa+a_3*ald_aa+d_3*dld_aa;
bb_d_off=c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_05_cuda(int ad, int bd, int cd, int dd, int ed, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,bld_aa,ald_aa,dld_aa,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ed*bd*ad*dd*sizeof(double);
size_bb=cd*ed*sizeof(double);
cudaFuncSetCacheConfig(tccg_05_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
bld_aa=ed;
ald_aa=bd*ed;
dld_aa=ad*bd*ed;
cld_bb=1;
eld_bb=cd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = cd*1;
int total_y = bd*ad*dd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_05_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,eld_aa,bld_aa,ald_aa,dld_aa,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_05_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, double *cc, double *aa, double *bb) {
tccg_05_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,e] += aa[e,f,b,a,d] * bb[c,f]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_06_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int eld_aa,int fld_aa,int bld_aa,int ald_aa,int dld_aa,int cld_bb,int fld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int eld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,fl,fT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
d_0=rest_y%dd;
rest_y=rest_y/dd;
e_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
d_1=rest_y%dd;
rest_y=rest_y/dd;
e_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
d_2=rest_y%dd;
rest_y=rest_y/dd;
e_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
d_3=rest_y%dd;
rest_y=rest_y/dd;
e_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(fT=0;fT<fd;fT+=Tcomm){int fl_hi;
fl_hi = MIN(Tcomm+fT,fd)-fT;
aa_d_off=e_0*eld_aa+b_0*bld_aa+a_0*ald_aa+d_0*dld_aa;
bb_d_off=c_0*cld_bb;
if(thread_y+T1*0<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*0][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*0] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_1*eld_aa+b_1*bld_aa+a_1*ald_aa+d_1*dld_aa;
bb_d_off=c_1*cld_bb;
if(thread_y+T1*1<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*1][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*1] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_2*eld_aa+b_2*bld_aa+a_2*ald_aa+d_2*dld_aa;
bb_d_off=c_2*cld_bb;
if(thread_y+T1*2<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*2][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*2] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_3*eld_aa+b_3*bld_aa+a_3*ald_aa+d_3*dld_aa;
bb_d_off=c_3*cld_bb;
if(thread_y+T1*3<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*3][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*3] = bb_d[bb_d_off+f*fld_bb];
}
__syncthreads();
for(fl=0;fl<fl_hi;++fl){
a1=aa_shm[in1_idxl+T1*0][fl];
a2=aa_shm[in1_idxl+T1*1][fl];
a3=aa_shm[in1_idxl+T1*2][fl];
a4=aa_shm[in1_idxl+T1*3][fl];
b1=bb_shm[fl][in2_idxl+T2*0];
b2=bb_shm[fl][in2_idxl+T2*1];
b3=bb_shm[fl][in2_idxl+T2*2];
b4=bb_shm[fl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_06_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,fld_aa,bld_aa,ald_aa,dld_aa,cld_bb,fld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*ed*sizeof(double);
size_aa=ed*fd*bd*ad*dd*sizeof(double);
size_bb=cd*fd*sizeof(double);
cudaFuncSetCacheConfig(tccg_06_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
fld_aa=ed;
bld_aa=fd*ed;
ald_aa=bd*fd*ed;
dld_aa=ad*bd*fd*ed;
cld_bb=1;
fld_bb=cd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
eld_cc=dd*cd*bd*ad;
int total_x = cd*1;
int total_y = ed*bd*ad*dd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_06_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,eld_aa,fld_aa,bld_aa,ald_aa,dld_aa,cld_bb,fld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_06_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_06_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,e] += aa[e,c,b,f,a] * bb[f,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_07_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int eld_aa,int cld_aa,int bld_aa,int fld_aa,int ald_aa,int fld_bb,int dld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int eld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,fl,fT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
c_0=rest_y%cd;
rest_y=rest_y/cd;
e_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
c_1=rest_y%cd;
rest_y=rest_y/cd;
e_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
c_2=rest_y%cd;
rest_y=rest_y/cd;
e_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
c_3=rest_y%cd;
rest_y=rest_y/cd;
e_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(fT=0;fT<fd;fT+=Tcomm){int fl_hi;
fl_hi = MIN(Tcomm+fT,fd)-fT;
aa_d_off=e_0*eld_aa+c_0*cld_aa+b_0*bld_aa+a_0*ald_aa;
bb_d_off=d_0*dld_bb;
if(thread_y+T1*0<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*0][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*0] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_1*eld_aa+c_1*cld_aa+b_1*bld_aa+a_1*ald_aa;
bb_d_off=d_1*dld_bb;
if(thread_y+T1*1<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*1][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*1] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_2*eld_aa+c_2*cld_aa+b_2*bld_aa+a_2*ald_aa;
bb_d_off=d_2*dld_bb;
if(thread_y+T1*2<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*2][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*2] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_3*eld_aa+c_3*cld_aa+b_3*bld_aa+a_3*ald_aa;
bb_d_off=d_3*dld_bb;
if(thread_y+T1*3<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*3][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*3] = bb_d[bb_d_off+f*fld_bb];
}
__syncthreads();
for(fl=0;fl<fl_hi;++fl){
a1=aa_shm[in1_idxl+T1*0][fl];
a2=aa_shm[in1_idxl+T1*1][fl];
a3=aa_shm[in1_idxl+T1*2][fl];
a4=aa_shm[in1_idxl+T1*3][fl];
b1=bb_shm[fl][in2_idxl+T2*0];
b2=bb_shm[fl][in2_idxl+T2*1];
b3=bb_shm[fl][in2_idxl+T2*2];
b4=bb_shm[fl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_2*dld_cc+e_3*eld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_07_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,cld_aa,bld_aa,fld_aa,ald_aa,fld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*ed*sizeof(double);
size_aa=ed*cd*bd*fd*ad*sizeof(double);
size_bb=fd*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_07_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
cld_aa=ed;
bld_aa=cd*ed;
fld_aa=bd*cd*ed;
ald_aa=fd*bd*cd*ed;
fld_bb=1;
dld_bb=fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
eld_cc=dd*cd*bd*ad;
int total_x = dd;
int total_y = ed*cd*bd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_07_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,eld_aa,cld_aa,bld_aa,fld_aa,ald_aa,fld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_07_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_07_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,e] += aa[e,f,c,a,d] * bb[b,f]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_08_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int eld_aa,int fld_aa,int cld_aa,int ald_aa,int dld_aa,int bld_bb,int fld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int eld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,fl,fT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_y%cd;
rest_y=rest_y/cd;
d_0=rest_y%dd;
rest_y=rest_y/dd;
e_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_y%cd;
rest_y=rest_y/cd;
d_1=rest_y%dd;
rest_y=rest_y/dd;
e_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_y%cd;
rest_y=rest_y/cd;
d_2=rest_y%dd;
rest_y=rest_y/dd;
e_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_y%cd;
rest_y=rest_y/cd;
d_3=rest_y%dd;
rest_y=rest_y/dd;
e_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(fT=0;fT<fd;fT+=Tcomm){int fl_hi;
fl_hi = MIN(Tcomm+fT,fd)-fT;
aa_d_off=e_0*eld_aa+c_0*cld_aa+a_0*ald_aa+d_0*dld_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*0][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*0] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_1*eld_aa+c_1*cld_aa+a_1*ald_aa+d_1*dld_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*1][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*1] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_2*eld_aa+c_2*cld_aa+a_2*ald_aa+d_2*dld_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*2][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*2] = bb_d[bb_d_off+f*fld_bb];
}
aa_d_off=e_3*eld_aa+c_3*cld_aa+a_3*ald_aa+d_3*dld_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(fl=threadIdx.x;fl<fl_hi;fl+=blockDim.x){
f=fl+fT;
aa_shm[in1_idxl+T1*3][fl] = aa_d[aa_d_off+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(fl=threadIdx.y;fl<fl_hi;fl+=blockDim.y){
f=fl+fT;
bb_shm[fl][in2_idxl+T1*3] = bb_d[bb_d_off+f*fld_bb];
}
__syncthreads();
for(fl=0;fl<fl_hi;++fl){
a1=aa_shm[in1_idxl+T1*0][fl];
a2=aa_shm[in1_idxl+T1*1][fl];
a3=aa_shm[in1_idxl+T1*2][fl];
a4=aa_shm[in1_idxl+T1*3][fl];
b1=bb_shm[fl][in2_idxl+T2*0];
b2=bb_shm[fl][in2_idxl+T2*1];
b3=bb_shm[fl][in2_idxl+T2*2];
b4=bb_shm[fl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_08_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,fld_aa,cld_aa,ald_aa,dld_aa,bld_bb,fld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*ed*sizeof(double);
size_aa=ed*fd*cd*ad*dd*sizeof(double);
size_bb=bd*fd*sizeof(double);
cudaFuncSetCacheConfig(tccg_08_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
fld_aa=ed;
cld_aa=fd*ed;
ald_aa=cd*fd*ed;
dld_aa=ad*cd*fd*ed;
bld_bb=1;
fld_bb=bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
eld_cc=dd*cd*bd*ad;
int total_x = bd*1;
int total_y = ed*cd*ad*dd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_08_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,eld_aa,fld_aa,cld_aa,ald_aa,dld_aa,bld_bb,fld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_08_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_08_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,d] += aa[e,a] * bb[e,b,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_09_kernel(int ad,int bd,int dd,int ed,int eld_aa,int ald_aa,int eld_bb,int bld_bb,int dld_bb,int ald_cc,int bld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,d_0,d_1,d_2,d_3,e;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
b_0=rest_x%bd;
rest_x=rest_x/bd;
a_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
b_1=rest_x%bd;
rest_x=rest_x/bd;
a_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
b_2=rest_x%bd;
rest_x=rest_x/bd;
a_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
b_3=rest_x%bd;
rest_x=rest_x/bd;
a_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa;
bb_d_off=b_0*bld_bb+d_0*dld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=a_1*ald_aa;
bb_d_off=b_1*bld_bb+d_1*dld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=a_2*ald_aa;
bb_d_off=b_2*bld_bb+d_2*dld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=a_3*ald_aa;
bb_d_off=b_3*bld_bb+d_3*dld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_09_cuda(int ad, int bd, int cd, int dd, int ed, double *cc, double *aa, double *bb) {
bd=bd*cd;
size_t stream;
size_t eld_aa,ald_aa,eld_bb,bld_bb,dld_bb,ald_cc,bld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*dd*sizeof(double);
size_aa=ed*ad*sizeof(double);
size_bb=ed*bd*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_09_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
ald_aa=ed;
eld_bb=1;
bld_bb=ed;
dld_bb=bd*ed;
ald_cc=1;
bld_cc=ad;
dld_cc=bd*ad;
int total_x = bd*dd;
int total_y = ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_09_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,dd,ed,eld_aa,ald_aa,eld_bb,bld_bb,dld_bb,ald_cc,bld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_09_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, double *cc, double *aa, double *bb) {
tccg_09_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[e,b] * bb[a,e,c,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_10_kernel(int ad,int bd,int cd,int dd,int ed,int eld_aa,int bld_aa,int ald_bb,int eld_bb,int cld_bb,int dld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
b_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
b_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
b_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
b_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=b_0*bld_aa;
bb_d_off=a_0*ald_bb+c_0*cld_bb+d_0*dld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=b_1*bld_aa;
bb_d_off=a_1*ald_bb+c_1*cld_bb+d_1*dld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=b_2*bld_aa;
bb_d_off=a_2*ald_bb+c_2*cld_bb+d_2*dld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=b_3*bld_aa;
bb_d_off=a_3*ald_bb+c_3*cld_bb+d_3*dld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_10_cuda(int ad, int bd, int cd, int dd, int ed, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,bld_aa,ald_bb,eld_bb,cld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ed*bd*sizeof(double);
size_bb=ad*ed*cd*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_10_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
bld_aa=ed;
ald_bb=1;
eld_bb=ad;
cld_bb=ed*ad;
dld_bb=cd*ed*ad;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = ad*cd*dd;
int total_y = bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_10_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,eld_aa,bld_aa,ald_bb,eld_bb,cld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_10_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, double *cc, double *aa, double *bb) {
tccg_10_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,c,d] += aa[e,c] * bb[a,e,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_11_kernel(int ad,int cd,int dd,int ed,int eld_aa,int cld_aa,int ald_bb,int eld_bb,int dld_bb,int ald_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
c_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
c_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
c_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
c_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=c_0*cld_aa;
bb_d_off=a_0*ald_bb+d_0*dld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=c_1*cld_aa;
bb_d_off=a_1*ald_bb+d_1*dld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=c_2*cld_aa;
bb_d_off=a_2*ald_bb+d_2*dld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb];
}
aa_d_off=c_3*cld_aa;
bb_d_off=a_3*ald_bb+d_3*dld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_0*ald_cc+c_3*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_1*ald_cc+c_3*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_2*ald_cc+c_3*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_11_cuda(int ad, int bd, int cd, int dd, int ed, double *cc, double *aa, double *bb) {
ad=ad*bd;
size_t stream;
size_t eld_aa,cld_aa,ald_bb,eld_bb,dld_bb,ald_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*cd*dd*sizeof(double);
size_aa=ed*cd*sizeof(double);
size_bb=ad*ed*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_11_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
cld_aa=ed;
ald_bb=1;
eld_bb=ad;
dld_bb=ed*ad;
ald_cc=1;
cld_cc=ad;
dld_cc=cd*ad;
int total_x = ad*dd;
int total_y = cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_11_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,cd,dd,ed,eld_aa,cld_aa,ald_bb,eld_bb,dld_bb,ald_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_11_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, double *cc, double *aa, double *bb) {
tccg_11_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b] += aa[a,c] * bb[c,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_12_kernel(int ad,int bd,int cd,int ald_aa,int cld_aa,int cld_bb,int bld_bb,int ald_cc,int bld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,cl,cT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(cT=0;cT<cd;cT+=Tcomm){int cl_hi;
cl_hi = MIN(Tcomm+cT,cd)-cT;
aa_d_off=a_0*ald_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*0][cl] = aa_d[aa_d_off+c*cld_aa];
}
if(thread_x+T1*0<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*0] = bb_d[bb_d_off+c*cld_bb];
}
aa_d_off=a_1*ald_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*1][cl] = aa_d[aa_d_off+c*cld_aa];
}
if(thread_x+T1*1<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*1] = bb_d[bb_d_off+c*cld_bb];
}
aa_d_off=a_2*ald_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*2][cl] = aa_d[aa_d_off+c*cld_aa];
}
if(thread_x+T1*2<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*2] = bb_d[bb_d_off+c*cld_bb];
}
aa_d_off=a_3*ald_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*3][cl] = aa_d[aa_d_off+c*cld_aa];
}
if(thread_x+T1*3<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*3] = bb_d[bb_d_off+c*cld_bb];
}
__syncthreads();
for(cl=0;cl<cl_hi;++cl){
a1=aa_shm[in1_idxl+T1*0][cl];
a2=aa_shm[in1_idxl+T1*1][cl];
a3=aa_shm[in1_idxl+T1*2][cl];
a4=aa_shm[in1_idxl+T1*3][cl];
b1=bb_shm[cl][in2_idxl+T2*0];
b2=bb_shm[cl][in2_idxl+T2*1];
b3=bb_shm[cl][in2_idxl+T2*2];
b4=bb_shm[cl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_12_cuda(int ad, int bd, int cd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,cld_aa,cld_bb,bld_bb,ald_cc,bld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*sizeof(double);
size_aa=ad*cd*sizeof(double);
size_bb=cd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_12_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
cld_aa=ad;
cld_bb=1;
bld_bb=cd;
ald_cc=1;
bld_cc=ad;
int total_x = bd;
int total_y = ad*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_12_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,ald_aa,cld_aa,cld_bb,bld_bb,ald_cc,bld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_12_cuda_(Integer *ad, Integer* bd, Integer* cd, double *cc, double *aa, double *bb) {
tccg_12_cuda((int)*ad,(int)*bd,(int)*cd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b] += aa[a,c,d] * bb[d,b,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_13_kernel(int ad,int bd,int cd,int dd,int ald_aa,int cld_aa,int dld_aa,int dld_bb,int bld_bb,int cld_bb,int ald_cc,int bld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,cl,cT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(d=0;d<dd;d++)
for(cT=0;cT<cd;cT+=Tcomm){int cl_hi;
cl_hi = MIN(Tcomm+cT,cd)-cT;
aa_d_off=a_0*ald_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*0][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*0] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
aa_d_off=a_1*ald_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*1][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*1] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
aa_d_off=a_2*ald_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*2][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*2] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
aa_d_off=a_3*ald_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*3][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*3] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
__syncthreads();
for(cl=0;cl<cl_hi;++cl){
a1=aa_shm[in1_idxl+T1*0][cl];
a2=aa_shm[in1_idxl+T1*1][cl];
a3=aa_shm[in1_idxl+T1*2][cl];
a4=aa_shm[in1_idxl+T1*3][cl];
b1=bb_shm[cl][in2_idxl+T2*0];
b2=bb_shm[cl][in2_idxl+T2*1];
b3=bb_shm[cl][in2_idxl+T2*2];
b4=bb_shm[cl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_13_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,cld_aa,dld_aa,dld_bb,bld_bb,cld_bb,ald_cc,bld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*sizeof(double);
size_aa=ad*cd*dd*sizeof(double);
size_bb=dd*bd*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_13_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
cld_aa=ad;
dld_aa=cd*ad;
dld_bb=1;
bld_bb=dd;
cld_bb=bd*dd;
ald_cc=1;
bld_cc=ad;
int total_x = bd*1;
int total_y = ad*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_13_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ald_aa,cld_aa,dld_aa,dld_bb,bld_bb,cld_bb,ald_cc,bld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_13_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_13_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b] += aa[c,a,d] * bb[d,c,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_14_kernel(int ad,int bd,int cd,int dd,int cld_aa,int ald_aa,int dld_aa,int dld_bb,int cld_bb,int bld_bb,int ald_cc,int bld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,cl,cT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(d=0;d<dd;d++)
for(cT=0;cT<cd;cT+=Tcomm){int cl_hi;
cl_hi = MIN(Tcomm+cT,cd)-cT;
aa_d_off=a_0*ald_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*0][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*0] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
aa_d_off=a_1*ald_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*1][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*1] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
aa_d_off=a_2*ald_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*2][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*2] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
aa_d_off=a_3*ald_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(cl=threadIdx.x;cl<cl_hi;cl+=blockDim.x){
c=cl+cT;
aa_shm[in1_idxl+T1*3][cl] = aa_d[aa_d_off+c*cld_aa+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(cl=threadIdx.y;cl<cl_hi;cl+=blockDim.y){
c=cl+cT;
bb_shm[cl][in2_idxl+T1*3] = bb_d[bb_d_off+c*cld_bb+d*dld_bb];
}
__syncthreads();
for(cl=0;cl<cl_hi;++cl){
a1=aa_shm[in1_idxl+T1*0][cl];
a2=aa_shm[in1_idxl+T1*1][cl];
a3=aa_shm[in1_idxl+T1*2][cl];
a4=aa_shm[in1_idxl+T1*3][cl];
b1=bb_shm[cl][in2_idxl+T2*0];
b2=bb_shm[cl][in2_idxl+T2*1];
b3=bb_shm[cl][in2_idxl+T2*2];
b4=bb_shm[cl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_14_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t cld_aa,ald_aa,dld_aa,dld_bb,cld_bb,bld_bb,ald_cc,bld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*sizeof(double);
size_aa=cd*ad*dd*sizeof(double);
size_bb=dd*cd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_14_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
cld_aa=1;
ald_aa=cd;
dld_aa=ad*cd;
dld_bb=1;
cld_bb=dd;
bld_bb=cd*dd;
ald_cc=1;
bld_cc=ad;
int total_x = bd;
int total_y = ad*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_14_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,cld_aa,ald_aa,dld_aa,dld_bb,cld_bb,bld_bb,ald_cc,bld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_14_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_14_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c] += aa[a,c,d] * bb[d,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_15_kernel(int ad,int bd,int cd,int dd,int ald_aa,int cld_aa,int dld_aa,int dld_bb,int bld_bb,int ald_cc,int bld_cc,int cld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,dl,dT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(dT=0;dT<dd;dT+=Tcomm){int dl_hi;
dl_hi = MIN(Tcomm+dT,dd)-dT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*0][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*0] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*1][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*1] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*2][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*2] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*3][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*3] = bb_d[bb_d_off+d*dld_bb];
}
__syncthreads();
for(dl=0;dl<dl_hi;++dl){
a1=aa_shm[in1_idxl+T1*0][dl];
a2=aa_shm[in1_idxl+T1*1][dl];
a3=aa_shm[in1_idxl+T1*2][dl];
a4=aa_shm[in1_idxl+T1*3][dl];
b1=bb_shm[dl][in2_idxl+T2*0];
b2=bb_shm[dl][in2_idxl+T2*1];
b3=bb_shm[dl][in2_idxl+T2*2];
b4=bb_shm[dl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_15_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,cld_aa,dld_aa,dld_bb,bld_bb,ald_cc,bld_cc,cld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*sizeof(double);
size_aa=ad*cd*dd*sizeof(double);
size_bb=dd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_15_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
cld_aa=ad;
dld_aa=cd*ad;
dld_bb=1;
bld_bb=dd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
int total_x = bd;
int total_y = ad*cd*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_15_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ald_aa,cld_aa,dld_aa,dld_bb,bld_bb,ald_cc,bld_cc,cld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_15_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_15_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c] += aa[a,d] * bb[b,d,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_16_kernel(int ad,int bd,int cd,int dd,int ald_aa,int dld_aa,int bld_bb,int dld_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,dl,dT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
b_0=rest_x%bd;
rest_x=rest_x/bd;
a_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
b_1=rest_x%bd;
rest_x=rest_x/bd;
a_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
b_2=rest_x%bd;
rest_x=rest_x/bd;
a_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
b_3=rest_x%bd;
rest_x=rest_x/bd;
a_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(dT=0;dT<dd;dT+=Tcomm){int dl_hi;
dl_hi = MIN(Tcomm+dT,dd)-dT;
aa_d_off=a_0*ald_aa;
bb_d_off=b_0*bld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*0][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*0] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_1*ald_aa;
bb_d_off=b_1*bld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*1][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*1] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_2*ald_aa;
bb_d_off=b_2*bld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*2][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*2] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_3*ald_aa;
bb_d_off=b_3*bld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*3][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*3] = bb_d[bb_d_off+d*dld_bb];
}
__syncthreads();
for(dl=0;dl<dl_hi;++dl){
a1=aa_shm[in1_idxl+T1*0][dl];
a2=aa_shm[in1_idxl+T1*1][dl];
a3=aa_shm[in1_idxl+T1*2][dl];
a4=aa_shm[in1_idxl+T1*3][dl];
b1=bb_shm[dl][in2_idxl+T2*0];
b2=bb_shm[dl][in2_idxl+T2*1];
b3=bb_shm[dl][in2_idxl+T2*2];
b4=bb_shm[dl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_16_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,dld_aa,bld_bb,dld_bb,cld_bb,ald_cc,bld_cc,cld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*sizeof(double);
size_aa=ad*dd*sizeof(double);
size_bb=bd*dd*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_16_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
dld_aa=ad;
bld_bb=1;
dld_bb=bd;
cld_bb=dd*bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
int total_x = bd*cd;
int total_y = ad*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_16_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ald_aa,dld_aa,bld_bb,dld_bb,cld_bb,ald_cc,bld_cc,cld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_16_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_16_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c] += aa[a,d,c] * bb[b,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_17_kernel(int ad,int bd,int cd,int dd,int ald_aa,int dld_aa,int cld_aa,int bld_bb,int dld_bb,int ald_cc,int bld_cc,int cld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,dl,dT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(dT=0;dT<dd;dT+=Tcomm){int dl_hi;
dl_hi = MIN(Tcomm+dT,dd)-dT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*0][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*0] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*1][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*1] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*2][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*2] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*3][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*3] = bb_d[bb_d_off+d*dld_bb];
}
__syncthreads();
for(dl=0;dl<dl_hi;++dl){
a1=aa_shm[in1_idxl+T1*0][dl];
a2=aa_shm[in1_idxl+T1*1][dl];
a3=aa_shm[in1_idxl+T1*2][dl];
a4=aa_shm[in1_idxl+T1*3][dl];
b1=bb_shm[dl][in2_idxl+T2*0];
b2=bb_shm[dl][in2_idxl+T2*1];
b3=bb_shm[dl][in2_idxl+T2*2];
b4=bb_shm[dl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_17_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,dld_aa,cld_aa,bld_bb,dld_bb,ald_cc,bld_cc,cld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*sizeof(double);
size_aa=ad*dd*cd*sizeof(double);
size_bb=bd*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_17_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
dld_aa=ad;
cld_aa=dd*ad;
bld_bb=1;
dld_bb=bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
int total_x = bd*1;
int total_y = ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_17_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ald_aa,dld_aa,cld_aa,bld_bb,dld_bb,ald_cc,bld_cc,cld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_17_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_17_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c] += aa[a,d,c] * bb[d,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_18_kernel(int ad,int bd,int cd,int dd,int ald_aa,int dld_aa,int cld_aa,int dld_bb,int bld_bb,int ald_cc,int bld_cc,int cld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,dl,dT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(dT=0;dT<dd;dT+=Tcomm){int dl_hi;
dl_hi = MIN(Tcomm+dT,dd)-dT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*0][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*0<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*0] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*1][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*1<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*1] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*2][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*2<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*2] = bb_d[bb_d_off+d*dld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*3][dl] = aa_d[aa_d_off+d*dld_aa];
}
if(thread_x+T1*3<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*3] = bb_d[bb_d_off+d*dld_bb];
}
__syncthreads();
for(dl=0;dl<dl_hi;++dl){
a1=aa_shm[in1_idxl+T1*0][dl];
a2=aa_shm[in1_idxl+T1*1][dl];
a3=aa_shm[in1_idxl+T1*2][dl];
a4=aa_shm[in1_idxl+T1*3][dl];
b1=bb_shm[dl][in2_idxl+T2*0];
b2=bb_shm[dl][in2_idxl+T2*1];
b3=bb_shm[dl][in2_idxl+T2*2];
b4=bb_shm[dl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_18_cuda(int ad, int bd, int cd, int dd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,dld_aa,cld_aa,dld_bb,bld_bb,ald_cc,bld_cc,cld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*sizeof(double);
size_aa=ad*dd*cd*sizeof(double);
size_bb=dd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_18_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
dld_aa=ad;
cld_aa=dd*ad;
dld_bb=1;
bld_bb=dd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
int total_x = bd;
int total_y = ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_18_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ald_aa,dld_aa,cld_aa,dld_bb,bld_bb,ald_cc,bld_cc,cld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_18_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, double *cc, double *aa, double *bb) {
tccg_18_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c] += aa[a,d,e,c] * bb[e,b,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_19_kernel(int ad,int bd,int cd,int dd,int ed,int ald_aa,int dld_aa,int eld_aa,int cld_aa,int eld_bb,int bld_bb,int dld_bb,int ald_cc,int bld_cc,int cld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d,e;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,dl,dT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_y;
b_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_y;
b_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_y;
b_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_y;
b_3=rest_x;
int aa_d_off, bb_d_off;for(e=0;e<ed;e++)
for(dT=0;dT<dd;dT+=Tcomm){int dl_hi;
dl_hi = MIN(Tcomm+dT,dd)-dT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb;
if(thread_y+T1*0<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*0][dl] = aa_d[aa_d_off+d*dld_aa+e*eld_aa];
}
if(thread_x+T1*0<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*0] = bb_d[bb_d_off+d*dld_bb+e*eld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb;
if(thread_y+T1*1<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*1][dl] = aa_d[aa_d_off+d*dld_aa+e*eld_aa];
}
if(thread_x+T1*1<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*1] = bb_d[bb_d_off+d*dld_bb+e*eld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb;
if(thread_y+T1*2<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*2][dl] = aa_d[aa_d_off+d*dld_aa+e*eld_aa];
}
if(thread_x+T1*2<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*2] = bb_d[bb_d_off+d*dld_bb+e*eld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb;
if(thread_y+T1*3<total_y)for(dl=threadIdx.x;dl<dl_hi;dl+=blockDim.x){
d=dl+dT;
aa_shm[in1_idxl+T1*3][dl] = aa_d[aa_d_off+d*dld_aa+e*eld_aa];
}
if(thread_x+T1*3<total_x)for(dl=threadIdx.y;dl<dl_hi;dl+=blockDim.y){
d=dl+dT;
bb_shm[dl][in2_idxl+T1*3] = bb_d[bb_d_off+d*dld_bb+e*eld_bb];
}
__syncthreads();
for(dl=0;dl<dl_hi;++dl){
a1=aa_shm[in1_idxl+T1*0][dl];
a2=aa_shm[in1_idxl+T1*1][dl];
a3=aa_shm[in1_idxl+T1*2][dl];
a4=aa_shm[in1_idxl+T1*3][dl];
b1=bb_shm[dl][in2_idxl+T2*0];
b2=bb_shm[dl][in2_idxl+T2*1];
b3=bb_shm[dl][in2_idxl+T2*2];
b4=bb_shm[dl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_19_cuda(int ad, int bd, int cd, int dd, int ed, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,dld_aa,eld_aa,cld_aa,eld_bb,bld_bb,dld_bb,ald_cc,bld_cc,cld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*sizeof(double);
size_aa=ad*dd*ed*cd*sizeof(double);
size_bb=ed*bd*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_19_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
dld_aa=ad;
eld_aa=dd*ad;
cld_aa=ed*dd*ad;
eld_bb=1;
bld_bb=ed;
dld_bb=bd*ed;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
int total_x = bd*1;
int total_y = ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_19_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,ald_aa,dld_aa,eld_aa,cld_aa,eld_bb,bld_bb,dld_bb,ald_cc,bld_cc,cld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_19_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, double *cc, double *aa, double *bb) {
tccg_19_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,b,f] * bb[d,f,c,e]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_20_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int bld_aa,int fld_aa,int dld_bb,int fld_bb,int cld_bb,int eld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
b_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
b_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
b_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
b_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+b_0*bld_aa;
bb_d_off=d_0*dld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+b_1*bld_aa;
bb_d_off=d_1*dld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+b_2*bld_aa;
bb_d_off=d_2*dld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+b_3*bld_aa;
bb_d_off=d_3*dld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_20_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,bld_aa,fld_aa,dld_bb,fld_bb,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*bd*fd*sizeof(double);
size_bb=dd*fd*cd*ed*sizeof(double);
cudaFuncSetCacheConfig(tccg_20_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
bld_aa=ed*ad;
fld_aa=bd*ed*ad;
dld_bb=1;
fld_bb=dd;
cld_bb=fd*dd;
eld_bb=cd*fd*dd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = dd*cd*1;
int total_y = ad*bd*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_20_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,bld_aa,fld_aa,dld_bb,fld_bb,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_20_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_20_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,b,f] * bb[f,d,e,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_21_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int bld_aa,int fld_aa,int fld_bb,int dld_bb,int eld_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
b_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
b_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
b_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
b_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+b_0*bld_aa;
bb_d_off=d_0*dld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+b_1*bld_aa;
bb_d_off=d_1*dld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+b_2*bld_aa;
bb_d_off=d_2*dld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+b_3*bld_aa;
bb_d_off=d_3*dld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_21_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,bld_aa,fld_aa,fld_bb,dld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*bd*fd*sizeof(double);
size_bb=fd*dd*ed*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_21_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
bld_aa=ed*ad;
fld_aa=bd*ed*ad;
fld_bb=1;
dld_bb=fd;
eld_bb=dd*fd;
cld_bb=ed*dd*fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = dd*cd;
int total_y = ad*bd*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_21_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,bld_aa,fld_aa,fld_bb,dld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_21_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_21_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,c,f] * bb[b,f,d,e]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_22_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int cld_aa,int fld_aa,int bld_bb,int fld_bb,int dld_bb,int eld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
c_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
c_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
c_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
c_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb+d_0*dld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb+d_1*dld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb+d_2*dld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb+d_3*dld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_22_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,cld_aa,fld_aa,bld_bb,fld_bb,dld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*cd*fd*sizeof(double);
size_bb=bd*fd*dd*ed*sizeof(double);
cudaFuncSetCacheConfig(tccg_22_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
cld_aa=ed*ad;
fld_aa=cd*ed*ad;
bld_bb=1;
fld_bb=bd;
dld_bb=fd*bd;
eld_bb=dd*fd*bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*dd*1;
int total_y = ad*cd*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_22_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,cld_aa,fld_aa,bld_bb,fld_bb,dld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_22_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_22_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,c,f] * bb[f,b,e,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_23_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int cld_aa,int fld_aa,int fld_bb,int bld_bb,int eld_bb,int dld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
c_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
c_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
c_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
c_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb+d_0*dld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb+d_1*dld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb+d_2*dld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb+d_3*dld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_23_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,cld_aa,fld_aa,fld_bb,bld_bb,eld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*cd*fd*sizeof(double);
size_bb=fd*bd*ed*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_23_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
cld_aa=ed*ad;
fld_aa=cd*ed*ad;
fld_bb=1;
bld_bb=fd;
eld_bb=bd*fd;
dld_bb=ed*bd*fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*dd;
int total_y = ad*cd*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_23_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,cld_aa,fld_aa,fld_bb,bld_bb,eld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_23_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_23_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,d,f] * bb[b,f,c,e]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_24_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int dld_aa,int fld_aa,int bld_bb,int fld_bb,int cld_bb,int eld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
d_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
d_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
d_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
d_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+d_0*dld_aa;
bb_d_off=b_0*bld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+d_1*dld_aa;
bb_d_off=b_1*bld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+d_2*dld_aa;
bb_d_off=b_2*bld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+d_3*dld_aa;
bb_d_off=b_3*bld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_24_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,dld_aa,fld_aa,bld_bb,fld_bb,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*dd*fd*sizeof(double);
size_bb=bd*fd*cd*ed*sizeof(double);
cudaFuncSetCacheConfig(tccg_24_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
dld_aa=ed*ad;
fld_aa=dd*ed*ad;
bld_bb=1;
fld_bb=bd;
cld_bb=fd*bd;
eld_bb=cd*fd*bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*cd*1;
int total_y = ad*dd*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_24_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,dld_aa,fld_aa,bld_bb,fld_bb,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_24_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_24_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,d,f] * bb[f,b,e,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_25_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int dld_aa,int fld_aa,int fld_bb,int bld_bb,int eld_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
d_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
d_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
d_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
d_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+d_0*dld_aa;
bb_d_off=b_0*bld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+d_1*dld_aa;
bb_d_off=b_1*bld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+d_2*dld_aa;
bb_d_off=b_2*bld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+d_3*dld_aa;
bb_d_off=b_3*bld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_25_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,dld_aa,fld_aa,fld_bb,bld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*dd*fd*sizeof(double);
size_bb=fd*bd*ed*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_25_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
dld_aa=ed*ad;
fld_aa=dd*ed*ad;
fld_bb=1;
bld_bb=fd;
eld_bb=bd*fd;
cld_bb=ed*bd*fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*cd;
int total_y = ad*dd*1;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_25_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,dld_aa,fld_aa,fld_bb,bld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_25_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_25_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,f,b] * bb[f,d,c,e]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_26_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int fld_aa,int bld_aa,int fld_bb,int dld_bb,int cld_bb,int eld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
b_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
b_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
b_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
b_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+b_0*bld_aa;
bb_d_off=d_0*dld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+b_1*bld_aa;
bb_d_off=d_1*dld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+b_2*bld_aa;
bb_d_off=d_2*dld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+b_3*bld_aa;
bb_d_off=d_3*dld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_26_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,fld_aa,bld_aa,fld_bb,dld_bb,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*fd*bd*sizeof(double);
size_bb=fd*dd*cd*ed*sizeof(double);
cudaFuncSetCacheConfig(tccg_26_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
fld_aa=ed*ad;
bld_aa=fd*ed*ad;
fld_bb=1;
dld_bb=fd;
cld_bb=dd*fd;
eld_bb=cd*dd*fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = dd*cd*1;
int total_y = ad*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_26_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,fld_aa,bld_aa,fld_bb,dld_bb,cld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_26_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_26_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[a,e,f,c] * bb[f,b,e,d]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_27_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int ald_aa,int eld_aa,int fld_aa,int cld_aa,int fld_bb,int bld_bb,int eld_bb,int dld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
c_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
c_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
c_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
c_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb+d_0*dld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb+d_1*dld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb+d_2*dld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb+d_3*dld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_27_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t ald_aa,eld_aa,fld_aa,cld_aa,fld_bb,bld_bb,eld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ad*ed*fd*cd*sizeof(double);
size_bb=fd*bd*ed*dd*sizeof(double);
cudaFuncSetCacheConfig(tccg_27_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
ald_aa=1;
eld_aa=ad;
fld_aa=ed*ad;
cld_aa=fd*ed*ad;
fld_bb=1;
bld_bb=fd;
eld_bb=bd*fd;
dld_bb=ed*bd*fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*dd;
int total_y = ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_27_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,ald_aa,eld_aa,fld_aa,cld_aa,fld_bb,bld_bb,eld_bb,dld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_27_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_27_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[e,a,f,b] * bb[f,d,e,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_28_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int eld_aa,int ald_aa,int fld_aa,int bld_aa,int fld_bb,int dld_bb,int eld_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
b_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
b_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
b_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
b_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+b_0*bld_aa;
bb_d_off=d_0*dld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+b_1*bld_aa;
bb_d_off=d_1*dld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+b_2*bld_aa;
bb_d_off=d_2*dld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+b_3*bld_aa;
bb_d_off=d_3*dld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_28_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,ald_aa,fld_aa,bld_aa,fld_bb,dld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ed*ad*fd*bd*sizeof(double);
size_bb=fd*dd*ed*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_28_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
ald_aa=ed;
fld_aa=ad*ed;
bld_aa=fd*ad*ed;
fld_bb=1;
dld_bb=fd;
eld_bb=dd*fd;
cld_bb=ed*dd*fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = dd*cd;
int total_y = ad*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_28_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,eld_aa,ald_aa,fld_aa,bld_aa,fld_bb,dld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_28_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_28_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[e,a,f,c] * bb[b,f,d,e]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_29_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int eld_aa,int ald_aa,int fld_aa,int cld_aa,int bld_bb,int fld_bb,int dld_bb,int eld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
c_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
c_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
c_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
c_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+c_0*cld_aa;
bb_d_off=b_0*bld_bb+d_0*dld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+c_1*cld_aa;
bb_d_off=b_1*bld_bb+d_1*dld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+c_2*cld_aa;
bb_d_off=b_2*bld_bb+d_2*dld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+c_3*cld_aa;
bb_d_off=b_3*bld_bb+d_3*dld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_29_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,ald_aa,fld_aa,cld_aa,bld_bb,fld_bb,dld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ed*ad*fd*cd*sizeof(double);
size_bb=bd*fd*dd*ed*sizeof(double);
cudaFuncSetCacheConfig(tccg_29_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
ald_aa=ed;
fld_aa=ad*ed;
cld_aa=fd*ad*ed;
bld_bb=1;
fld_bb=bd;
dld_bb=fd*bd;
eld_bb=dd*fd*bd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*dd*1;
int total_y = ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_29_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,eld_aa,ald_aa,fld_aa,cld_aa,bld_bb,fld_bb,dld_bb,eld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_29_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_29_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d] += aa[e,a,f,d] * bb[f,b,e,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_30_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int eld_aa,int ald_aa,int fld_aa,int dld_aa,int fld_bb,int bld_bb,int eld_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e,f;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,el,eT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
d_0=rest_y;
c_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
d_1=rest_y;
c_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
d_2=rest_y;
c_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
d_3=rest_y;
c_3=rest_x;
int aa_d_off, bb_d_off;for(f=0;f<fd;f++)
for(eT=0;eT<ed;eT+=Tcomm){int el_hi;
el_hi = MIN(Tcomm+eT,ed)-eT;
aa_d_off=a_0*ald_aa+d_0*dld_aa;
bb_d_off=b_0*bld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*0][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*0<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*0] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_1*ald_aa+d_1*dld_aa;
bb_d_off=b_1*bld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*1][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*1<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*1] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_2*ald_aa+d_2*dld_aa;
bb_d_off=b_2*bld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*2][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*2<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*2] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
aa_d_off=a_3*ald_aa+d_3*dld_aa;
bb_d_off=b_3*bld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(el=threadIdx.x;el<el_hi;el+=blockDim.x){
e=el+eT;
aa_shm[in1_idxl+T1*3][el] = aa_d[aa_d_off+e*eld_aa+f*fld_aa];
}
if(thread_x+T1*3<total_x)for(el=threadIdx.y;el<el_hi;el+=blockDim.y){
e=el+eT;
bb_shm[el][in2_idxl+T1*3] = bb_d[bb_d_off+e*eld_bb+f*fld_bb];
}
__syncthreads();
for(el=0;el<el_hi;++el){
a1=aa_shm[in1_idxl+T1*0][el];
a2=aa_shm[in1_idxl+T1*1][el];
a3=aa_shm[in1_idxl+T1*2][el];
a4=aa_shm[in1_idxl+T1*3][el];
b1=bb_shm[el][in2_idxl+T2*0];
b2=bb_shm[el][in2_idxl+T2*1];
b3=bb_shm[el][in2_idxl+T2*2];
b4=bb_shm[el][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_0*cld_cc+d_3*dld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_0*cld_cc+d_2*dld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_0*cld_cc+d_1*dld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_1*cld_cc+d_3*dld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_1*cld_cc+d_2*dld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_1*cld_cc+d_0*dld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_2*cld_cc+d_3*dld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_2*cld_cc+d_1*dld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_2*cld_cc+d_0*dld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_3*cld_cc+d_2*dld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_3*cld_cc+d_1*dld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_3*cld_cc+d_0*dld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_30_cuda(int ad, int bd, int cd, int dd, int ed, int fd, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,ald_aa,fld_aa,dld_aa,fld_bb,bld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*sizeof(double);
size_aa=ed*ad*fd*dd*sizeof(double);
size_bb=fd*bd*ed*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_30_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
ald_aa=ed;
fld_aa=ad*ed;
dld_aa=fd*ad*ed;
fld_bb=1;
bld_bb=fd;
eld_bb=bd*fd;
cld_bb=ed*bd*fd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
int total_x = bd*cd;
int total_y = ad*dd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_30_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,eld_aa,ald_aa,fld_aa,dld_aa,fld_bb,bld_bb,eld_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_30_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, double *cc, double *aa, double *bb) {
tccg_30_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,d,f] += aa[d,g,a] * bb[g,f,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_31_kernel(int ad,int bd,int dd,int fd,int gd,int dld_aa,int gld_aa,int ald_aa,int gld_bb,int fld_bb,int bld_bb,int ald_cc,int bld_cc,int dld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,d_0,d_1,d_2,d_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
d_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
d_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
d_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
d_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+a_0*ald_aa;
bb_d_off=f_0*fld_bb+b_0*bld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+a_1*ald_aa;
bb_d_off=f_1*fld_bb+b_1*bld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+a_2*ald_aa;
bb_d_off=f_2*fld_bb+b_2*bld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+a_3*ald_aa;
bb_d_off=f_3*fld_bb+b_3*bld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+d_3*dld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_31_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
dd=dd*ed;
bd=bd*cd;
size_t stream;
size_t dld_aa,gld_aa,ald_aa,gld_bb,fld_bb,bld_bb,ald_cc,bld_cc,dld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*dd*fd*sizeof(double);
size_aa=dd*gd*ad*sizeof(double);
size_bb=gd*fd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_31_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
gld_aa=dd;
ald_aa=gd*dd;
gld_bb=1;
fld_bb=gd;
bld_bb=fd*gd;
ald_cc=1;
bld_cc=ad;
dld_cc=bd*ad;
fld_cc=dd*bd*ad;
int total_x = fd*bd;
int total_y = dd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_31_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,dd,fd,gd,dld_aa,gld_aa,ald_aa,gld_bb,fld_bb,bld_bb,ald_cc,bld_cc,dld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_31_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_31_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,f] += aa[d,g,b] * bb[g,f,a,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_32_kernel(int ad,int bd,int cd,int dd,int fd,int gd,int dld_aa,int gld_aa,int bld_aa,int gld_bb,int fld_bb,int ald_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
c_0=rest_x%cd;
rest_x=rest_x/cd;
d_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
c_1=rest_x%cd;
rest_x=rest_x/cd;
d_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
c_2=rest_x%cd;
rest_x=rest_x/cd;
d_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
c_3=rest_x%cd;
rest_x=rest_x/cd;
d_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+b_0*bld_aa;
bb_d_off=f_0*fld_bb+a_0*ald_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+b_1*bld_aa;
bb_d_off=f_1*fld_bb+a_1*ald_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+b_2*bld_aa;
bb_d_off=f_2*fld_bb+a_2*ald_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+b_3*bld_aa;
bb_d_off=f_3*fld_bb+a_3*ald_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_32_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
dd=dd*ed;
size_t stream;
size_t dld_aa,gld_aa,bld_aa,gld_bb,fld_bb,ald_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*fd*sizeof(double);
size_aa=dd*gd*bd*sizeof(double);
size_bb=gd*fd*ad*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_32_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
gld_aa=dd;
bld_aa=gd*dd;
gld_bb=1;
fld_bb=gd;
ald_bb=fd*gd;
cld_bb=ad*fd*gd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
fld_cc=dd*cd*bd*ad;
int total_x = fd*ad*cd;
int total_y = dd*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_32_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,fd,gd,dld_aa,gld_aa,bld_aa,gld_bb,fld_bb,ald_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_32_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_32_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,c,d,f] += aa[d,g,c] * bb[g,f,a]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_33_kernel(int ad,int cd,int dd,int fd,int gd,int dld_aa,int gld_aa,int cld_aa,int gld_bb,int fld_bb,int ald_bb,int ald_cc,int cld_cc,int dld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
c_0=rest_y%cd;
rest_y=rest_y/cd;
d_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
c_1=rest_y%cd;
rest_y=rest_y/cd;
d_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
c_2=rest_y%cd;
rest_y=rest_y/cd;
d_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
c_3=rest_y%cd;
rest_y=rest_y/cd;
d_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+c_0*cld_aa;
bb_d_off=f_0*fld_bb+a_0*ald_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+c_1*cld_aa;
bb_d_off=f_1*fld_bb+a_1*ald_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+c_2*cld_aa;
bb_d_off=f_2*fld_bb+a_2*ald_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+c_3*cld_aa;
bb_d_off=f_3*fld_bb+a_3*ald_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+c_3*cld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+c_3*cld_cc+d_3*dld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_33_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
dd=dd*ed;
ad=ad*bd;
size_t stream;
size_t dld_aa,gld_aa,cld_aa,gld_bb,fld_bb,ald_bb,ald_cc,cld_cc,dld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*cd*dd*fd*sizeof(double);
size_aa=dd*gd*cd*sizeof(double);
size_bb=gd*fd*ad*sizeof(double);
cudaFuncSetCacheConfig(tccg_33_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
gld_aa=dd;
cld_aa=gd*dd;
gld_bb=1;
fld_bb=gd;
ald_bb=fd*gd;
ald_cc=1;
cld_cc=ad;
dld_cc=cd*ad;
fld_cc=dd*cd*ad;
int total_x = fd*ad;
int total_y = dd*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_33_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,cd,dd,fd,gd,dld_aa,gld_aa,cld_aa,gld_bb,fld_bb,ald_bb,ald_cc,cld_cc,dld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_33_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_33_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,d,e,f] += aa[d,f,g,a] * bb[g,e,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_34_kernel(int ad,int bd,int dd,int ed,int fd,int gd,int dld_aa,int fld_aa,int gld_aa,int ald_aa,int gld_bb,int eld_bb,int bld_bb,int ald_cc,int bld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
d_0=rest_y%dd;
rest_y=rest_y/dd;
f_0=rest_y;
e_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
d_1=rest_y%dd;
rest_y=rest_y/dd;
f_1=rest_y;
e_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
d_2=rest_y%dd;
rest_y=rest_y/dd;
f_2=rest_y;
e_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
d_3=rest_y%dd;
rest_y=rest_y/dd;
f_3=rest_y;
e_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+f_0*fld_aa+a_0*ald_aa;
bb_d_off=e_0*eld_bb+b_0*bld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+f_1*fld_aa+a_1*ald_aa;
bb_d_off=e_1*eld_bb+b_1*bld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+f_2*fld_aa+a_2*ald_aa;
bb_d_off=e_2*eld_bb+b_2*bld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+f_3*fld_aa+a_3*ald_aa;
bb_d_off=e_3*eld_bb+b_3*bld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_34_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
bd=bd*cd;
size_t stream;
size_t dld_aa,fld_aa,gld_aa,ald_aa,gld_bb,eld_bb,bld_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*dd*ed*fd*sizeof(double);
size_aa=dd*fd*gd*ad*sizeof(double);
size_bb=gd*ed*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_34_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
fld_aa=dd;
gld_aa=fd*dd;
ald_aa=gd*fd*dd;
gld_bb=1;
eld_bb=gd;
bld_bb=ed*gd;
ald_cc=1;
bld_cc=ad;
dld_cc=bd*ad;
eld_cc=dd*bd*ad;
fld_cc=ed*dd*bd*ad;
int total_x = ed*bd;
int total_y = dd*fd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_34_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,dd,ed,fd,gd,dld_aa,fld_aa,gld_aa,ald_aa,gld_bb,eld_bb,bld_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_34_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_34_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,e,f] += aa[d,f,g,b] * bb[g,e,a,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_35_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int gd,int dld_aa,int fld_aa,int gld_aa,int bld_aa,int gld_bb,int eld_bb,int ald_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
c_0=rest_x%cd;
rest_x=rest_x/cd;
d_0=rest_y%dd;
rest_y=rest_y/dd;
f_0=rest_y;
e_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
c_1=rest_x%cd;
rest_x=rest_x/cd;
d_1=rest_y%dd;
rest_y=rest_y/dd;
f_1=rest_y;
e_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
c_2=rest_x%cd;
rest_x=rest_x/cd;
d_2=rest_y%dd;
rest_y=rest_y/dd;
f_2=rest_y;
e_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
c_3=rest_x%cd;
rest_x=rest_x/cd;
d_3=rest_y%dd;
rest_y=rest_y/dd;
f_3=rest_y;
e_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+f_0*fld_aa+b_0*bld_aa;
bb_d_off=e_0*eld_bb+a_0*ald_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+f_1*fld_aa+b_1*bld_aa;
bb_d_off=e_1*eld_bb+a_1*ald_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+f_2*fld_aa+b_2*bld_aa;
bb_d_off=e_2*eld_bb+a_2*ald_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+f_3*fld_aa+b_3*bld_aa;
bb_d_off=e_3*eld_bb+a_3*ald_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_35_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
size_t stream;
size_t dld_aa,fld_aa,gld_aa,bld_aa,gld_bb,eld_bb,ald_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*ed*fd*sizeof(double);
size_aa=dd*fd*gd*bd*sizeof(double);
size_bb=gd*ed*ad*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_35_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
fld_aa=dd;
gld_aa=fd*dd;
bld_aa=gd*fd*dd;
gld_bb=1;
eld_bb=gd;
ald_bb=ed*gd;
cld_bb=ad*ed*gd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
eld_cc=dd*cd*bd*ad;
fld_cc=ed*dd*cd*bd*ad;
int total_x = ed*ad*cd;
int total_y = dd*fd*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_35_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,gd,dld_aa,fld_aa,gld_aa,bld_aa,gld_bb,eld_bb,ald_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_35_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_35_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,c,d,e,f] += aa[d,f,g,c] * bb[g,e,a]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_36_kernel(int ad,int cd,int dd,int ed,int fd,int gd,int dld_aa,int fld_aa,int gld_aa,int cld_aa,int gld_bb,int eld_bb,int ald_bb,int ald_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
c_0=rest_y%cd;
rest_y=rest_y/cd;
d_0=rest_y%dd;
rest_y=rest_y/dd;
f_0=rest_y;
e_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
c_1=rest_y%cd;
rest_y=rest_y/cd;
d_1=rest_y%dd;
rest_y=rest_y/dd;
f_1=rest_y;
e_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
c_2=rest_y%cd;
rest_y=rest_y/cd;
d_2=rest_y%dd;
rest_y=rest_y/dd;
f_2=rest_y;
e_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
c_3=rest_y%cd;
rest_y=rest_y/cd;
d_3=rest_y%dd;
rest_y=rest_y/dd;
f_3=rest_y;
e_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+f_0*fld_aa+c_0*cld_aa;
bb_d_off=e_0*eld_bb+a_0*ald_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+f_1*fld_aa+c_1*cld_aa;
bb_d_off=e_1*eld_bb+a_1*ald_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+f_2*fld_aa+c_2*cld_aa;
bb_d_off=e_2*eld_bb+a_2*ald_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+f_3*fld_aa+c_3*cld_aa;
bb_d_off=e_3*eld_bb+a_3*ald_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+c_3*cld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_36_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
ad=ad*bd;
size_t stream;
size_t dld_aa,fld_aa,gld_aa,cld_aa,gld_bb,eld_bb,ald_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*cd*dd*ed*fd*sizeof(double);
size_aa=dd*fd*gd*cd*sizeof(double);
size_bb=gd*ed*ad*sizeof(double);
cudaFuncSetCacheConfig(tccg_36_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
dld_aa=1;
fld_aa=dd;
gld_aa=fd*dd;
cld_aa=gd*fd*dd;
gld_bb=1;
eld_bb=gd;
ald_bb=ed*gd;
ald_cc=1;
cld_cc=ad;
dld_cc=cd*ad;
eld_cc=dd*cd*ad;
fld_cc=ed*dd*cd*ad;
int total_x = ed*ad;
int total_y = dd*fd*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_36_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,cd,dd,ed,fd,gd,dld_aa,fld_aa,gld_aa,cld_aa,gld_bb,eld_bb,ald_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_36_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_36_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,d,e,f] += aa[e,f,g,a] * bb[g,d,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_37_kernel(int ad,int bd,int dd,int ed,int fd,int gd,int eld_aa,int fld_aa,int gld_aa,int ald_aa,int gld_bb,int dld_bb,int bld_bb,int ald_cc,int bld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
e_0=rest_y%ed;
rest_y=rest_y/ed;
f_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
e_1=rest_y%ed;
rest_y=rest_y/ed;
f_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
e_2=rest_y%ed;
rest_y=rest_y/ed;
f_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
e_3=rest_y%ed;
rest_y=rest_y/ed;
f_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=e_0*eld_aa+f_0*fld_aa+a_0*ald_aa;
bb_d_off=d_0*dld_bb+b_0*bld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_1*eld_aa+f_1*fld_aa+a_1*ald_aa;
bb_d_off=d_1*dld_bb+b_1*bld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_2*eld_aa+f_2*fld_aa+a_2*ald_aa;
bb_d_off=d_2*dld_bb+b_2*bld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_3*eld_aa+f_3*fld_aa+a_3*ald_aa;
bb_d_off=d_3*dld_bb+b_3*bld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_37_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
bd=bd*cd;
size_t stream;
size_t eld_aa,fld_aa,gld_aa,ald_aa,gld_bb,dld_bb,bld_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*dd*ed*fd*sizeof(double);
size_aa=ed*fd*gd*ad*sizeof(double);
size_bb=gd*dd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_37_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
fld_aa=ed;
gld_aa=fd*ed;
ald_aa=gd*fd*ed;
gld_bb=1;
dld_bb=gd;
bld_bb=dd*gd;
ald_cc=1;
bld_cc=ad;
dld_cc=bd*ad;
eld_cc=dd*bd*ad;
fld_cc=ed*dd*bd*ad;
int total_x = dd*bd;
int total_y = ed*fd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_37_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,dd,ed,fd,gd,eld_aa,fld_aa,gld_aa,ald_aa,gld_bb,dld_bb,bld_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_37_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_37_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,e,f] += aa[e,f,g,b] * bb[g,d,a,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_38_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int gd,int eld_aa,int fld_aa,int gld_aa,int bld_aa,int gld_bb,int dld_bb,int ald_bb,int cld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
c_0=rest_x%cd;
rest_x=rest_x/cd;
e_0=rest_y%ed;
rest_y=rest_y/ed;
f_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
c_1=rest_x%cd;
rest_x=rest_x/cd;
e_1=rest_y%ed;
rest_y=rest_y/ed;
f_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
c_2=rest_x%cd;
rest_x=rest_x/cd;
e_2=rest_y%ed;
rest_y=rest_y/ed;
f_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
c_3=rest_x%cd;
rest_x=rest_x/cd;
e_3=rest_y%ed;
rest_y=rest_y/ed;
f_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=e_0*eld_aa+f_0*fld_aa+b_0*bld_aa;
bb_d_off=d_0*dld_bb+a_0*ald_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_1*eld_aa+f_1*fld_aa+b_1*bld_aa;
bb_d_off=d_1*dld_bb+a_1*ald_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_2*eld_aa+f_2*fld_aa+b_2*bld_aa;
bb_d_off=d_2*dld_bb+a_2*ald_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_3*eld_aa+f_3*fld_aa+b_3*bld_aa;
bb_d_off=d_3*dld_bb+a_3*ald_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_38_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
size_t stream;
size_t eld_aa,fld_aa,gld_aa,bld_aa,gld_bb,dld_bb,ald_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*ed*fd*sizeof(double);
size_aa=ed*fd*gd*bd*sizeof(double);
size_bb=gd*dd*ad*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_38_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
fld_aa=ed;
gld_aa=fd*ed;
bld_aa=gd*fd*ed;
gld_bb=1;
dld_bb=gd;
ald_bb=dd*gd;
cld_bb=ad*dd*gd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
eld_cc=dd*cd*bd*ad;
fld_cc=ed*dd*cd*bd*ad;
int total_x = dd*ad*cd;
int total_y = ed*fd*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_38_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,gd,eld_aa,fld_aa,gld_aa,bld_aa,gld_bb,dld_bb,ald_bb,cld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_38_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_38_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,c,d,e,f] += aa[e,f,g,c] * bb[g,d,a]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_39_kernel(int ad,int cd,int dd,int ed,int fd,int gd,int eld_aa,int fld_aa,int gld_aa,int cld_aa,int gld_bb,int dld_bb,int ald_bb,int ald_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
c_0=rest_y%cd;
rest_y=rest_y/cd;
e_0=rest_y%ed;
rest_y=rest_y/ed;
f_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
c_1=rest_y%cd;
rest_y=rest_y/cd;
e_1=rest_y%ed;
rest_y=rest_y/ed;
f_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
c_2=rest_y%cd;
rest_y=rest_y/cd;
e_2=rest_y%ed;
rest_y=rest_y/ed;
f_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
c_3=rest_y%cd;
rest_y=rest_y/cd;
e_3=rest_y%ed;
rest_y=rest_y/ed;
f_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=e_0*eld_aa+f_0*fld_aa+c_0*cld_aa;
bb_d_off=d_0*dld_bb+a_0*ald_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_1*eld_aa+f_1*fld_aa+c_1*cld_aa;
bb_d_off=d_1*dld_bb+a_1*ald_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_2*eld_aa+f_2*fld_aa+c_2*cld_aa;
bb_d_off=d_2*dld_bb+a_2*ald_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_3*eld_aa+f_3*fld_aa+c_3*cld_aa;
bb_d_off=d_3*dld_bb+a_3*ald_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+c_3*cld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_39_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
ad=ad*bd;
size_t stream;
size_t eld_aa,fld_aa,gld_aa,cld_aa,gld_bb,dld_bb,ald_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*cd*dd*ed*fd*sizeof(double);
size_aa=ed*fd*gd*cd*sizeof(double);
size_bb=gd*dd*ad*sizeof(double);
cudaFuncSetCacheConfig(tccg_39_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
eld_aa=1;
fld_aa=ed;
gld_aa=fd*ed;
cld_aa=gd*fd*ed;
gld_bb=1;
dld_bb=gd;
ald_bb=dd*gd;
ald_cc=1;
cld_cc=ad;
dld_cc=cd*ad;
eld_cc=dd*cd*ad;
fld_cc=ed*dd*cd*ad;
int total_x = dd*ad;
int total_y = ed*fd*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_39_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,cd,dd,ed,fd,gd,eld_aa,fld_aa,gld_aa,cld_aa,gld_bb,dld_bb,ald_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_39_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_39_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,c,d,e,f] += aa[g,d,a] * bb[e,f,g,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_40_kernel(int ad,int cd,int dd,int ed,int fd,int gd,int gld_aa,int dld_aa,int ald_aa,int eld_bb,int fld_bb,int gld_bb,int cld_bb,int ald_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
e_0=rest_x%ed;
rest_x=rest_x/ed;
d_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
e_1=rest_x%ed;
rest_x=rest_x/ed;
d_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
e_2=rest_x%ed;
rest_x=rest_x/ed;
d_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
e_3=rest_x%ed;
rest_x=rest_x/ed;
d_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+a_0*ald_aa;
bb_d_off=e_0*eld_bb+f_0*fld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+a_1*ald_aa;
bb_d_off=e_1*eld_bb+f_1*fld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+a_2*ald_aa;
bb_d_off=e_2*eld_bb+f_2*fld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+a_3*ald_aa;
bb_d_off=e_3*eld_bb+f_3*fld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+c_3*cld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+c_3*cld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_40_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
ad=ad*bd;
size_t stream;
size_t gld_aa,dld_aa,ald_aa,eld_bb,fld_bb,gld_bb,cld_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*cd*dd*ed*fd*sizeof(double);
size_aa=gd*dd*ad*sizeof(double);
size_bb=ed*fd*gd*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_40_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
dld_aa=gd;
ald_aa=dd*gd;
eld_bb=1;
fld_bb=ed;
gld_bb=fd*ed;
cld_bb=gd*fd*ed;
ald_cc=1;
cld_cc=ad;
dld_cc=cd*ad;
eld_cc=dd*cd*ad;
fld_cc=ed*dd*cd*ad;
int total_x = ed*fd*cd;
int total_y = dd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_40_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,cd,dd,ed,fd,gd,gld_aa,dld_aa,ald_aa,eld_bb,fld_bb,gld_bb,cld_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_40_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_40_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,e,f] += aa[g,d,a,c] * bb[e,f,g,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_41_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int gd,int gld_aa,int dld_aa,int ald_aa,int cld_aa,int eld_bb,int fld_bb,int gld_bb,int bld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
c_0=rest_y%cd;
rest_y=rest_y/cd;
e_0=rest_x%ed;
rest_x=rest_x/ed;
d_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
c_1=rest_y%cd;
rest_y=rest_y/cd;
e_1=rest_x%ed;
rest_x=rest_x/ed;
d_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
c_2=rest_y%cd;
rest_y=rest_y/cd;
e_2=rest_x%ed;
rest_x=rest_x/ed;
d_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
c_3=rest_y%cd;
rest_y=rest_y/cd;
e_3=rest_x%ed;
rest_x=rest_x/ed;
d_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+a_0*ald_aa+c_0*cld_aa;
bb_d_off=e_0*eld_bb+f_0*fld_bb+b_0*bld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+a_1*ald_aa+c_1*cld_aa;
bb_d_off=e_1*eld_bb+f_1*fld_bb+b_1*bld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+a_2*ald_aa+c_2*cld_aa;
bb_d_off=e_2*eld_bb+f_2*fld_bb+b_2*bld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+a_3*ald_aa+c_3*cld_aa;
bb_d_off=e_3*eld_bb+f_3*fld_bb+b_3*bld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_41_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
size_t stream;
size_t gld_aa,dld_aa,ald_aa,cld_aa,eld_bb,fld_bb,gld_bb,bld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*ed*fd*sizeof(double);
size_aa=gd*dd*ad*cd*sizeof(double);
size_bb=ed*fd*gd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_41_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
dld_aa=gd;
ald_aa=dd*gd;
cld_aa=ad*dd*gd;
eld_bb=1;
fld_bb=ed;
gld_bb=fd*ed;
bld_bb=gd*fd*ed;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
eld_cc=dd*cd*bd*ad;
fld_cc=ed*dd*cd*bd*ad;
int total_x = ed*fd*bd;
int total_y = dd*ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_41_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,gd,gld_aa,dld_aa,ald_aa,cld_aa,eld_bb,fld_bb,gld_bb,bld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_41_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_41_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,d,e,f] += aa[g,d,b] * bb[e,f,g,a]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_42_kernel(int ad,int bd,int dd,int ed,int fd,int gd,int gld_aa,int dld_aa,int bld_aa,int eld_bb,int fld_bb,int gld_bb,int ald_bb,int ald_cc,int bld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
e_0=rest_x%ed;
rest_x=rest_x/ed;
d_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
e_1=rest_x%ed;
rest_x=rest_x/ed;
d_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
e_2=rest_x%ed;
rest_x=rest_x/ed;
d_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
e_3=rest_x%ed;
rest_x=rest_x/ed;
d_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=d_0*dld_aa+b_0*bld_aa;
bb_d_off=e_0*eld_bb+f_0*fld_bb+a_0*ald_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_1*dld_aa+b_1*bld_aa;
bb_d_off=e_1*eld_bb+f_1*fld_bb+a_1*ald_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_2*dld_aa+b_2*bld_aa;
bb_d_off=e_2*eld_bb+f_2*fld_bb+a_2*ald_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=d_3*dld_aa+b_3*bld_aa;
bb_d_off=e_3*eld_bb+f_3*fld_bb+a_3*ald_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+b_3*bld_cc+d_3*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+d_2*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_1*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+b_3*bld_cc+d_3*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+d_2*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_0*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+b_3*bld_cc+d_3*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_1*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_0*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+d_2*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_1*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_0*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_42_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
bd=bd*cd;
size_t stream;
size_t gld_aa,dld_aa,bld_aa,eld_bb,fld_bb,gld_bb,ald_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*dd*ed*fd*sizeof(double);
size_aa=gd*dd*bd*sizeof(double);
size_bb=ed*fd*gd*ad*sizeof(double);
cudaFuncSetCacheConfig(tccg_42_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
dld_aa=gd;
bld_aa=dd*gd;
eld_bb=1;
fld_bb=ed;
gld_bb=fd*ed;
ald_bb=gd*fd*ed;
ald_cc=1;
bld_cc=ad;
dld_cc=bd*ad;
eld_cc=dd*bd*ad;
fld_cc=ed*dd*bd*ad;
int total_x = ed*fd*ad;
int total_y = dd*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_42_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,dd,ed,fd,gd,gld_aa,dld_aa,bld_aa,eld_bb,fld_bb,gld_bb,ald_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_42_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_42_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,c,d,e,f] += aa[g,e,a] * bb[d,f,g,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_43_kernel(int ad,int cd,int dd,int ed,int fd,int gd,int gld_aa,int eld_aa,int ald_aa,int dld_bb,int fld_bb,int gld_bb,int cld_bb,int ald_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
d_0=rest_x%dd;
rest_x=rest_x/dd;
e_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
d_1=rest_x%dd;
rest_x=rest_x/dd;
e_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
d_2=rest_x%dd;
rest_x=rest_x/dd;
e_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
d_3=rest_x%dd;
rest_x=rest_x/dd;
e_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=e_0*eld_aa+a_0*ald_aa;
bb_d_off=d_0*dld_bb+f_0*fld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_1*eld_aa+a_1*ald_aa;
bb_d_off=d_1*dld_bb+f_1*fld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_2*eld_aa+a_2*ald_aa;
bb_d_off=d_2*dld_bb+f_2*fld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_3*eld_aa+a_3*ald_aa;
bb_d_off=d_3*dld_bb+f_3*fld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+c_2*cld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+c_3*cld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+c_3*cld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_43_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
ad=ad*bd;
size_t stream;
size_t gld_aa,eld_aa,ald_aa,dld_bb,fld_bb,gld_bb,cld_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*cd*dd*ed*fd*sizeof(double);
size_aa=gd*ed*ad*sizeof(double);
size_bb=dd*fd*gd*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_43_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
eld_aa=gd;
ald_aa=ed*gd;
dld_bb=1;
fld_bb=dd;
gld_bb=fd*dd;
cld_bb=gd*fd*dd;
ald_cc=1;
cld_cc=ad;
dld_cc=cd*ad;
eld_cc=dd*cd*ad;
fld_cc=ed*dd*cd*ad;
int total_x = dd*fd*cd;
int total_y = ed*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_43_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,cd,dd,ed,fd,gd,gld_aa,eld_aa,ald_aa,dld_bb,fld_bb,gld_bb,cld_bb,ald_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_43_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_43_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,e,f] += aa[g,e,a,c] * bb[d,f,g,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_44_kernel(int ad,int bd,int cd,int dd,int ed,int fd,int gd,int gld_aa,int eld_aa,int ald_aa,int cld_aa,int dld_bb,int fld_bb,int gld_bb,int bld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
c_0=rest_y%cd;
rest_y=rest_y/cd;
d_0=rest_x%dd;
rest_x=rest_x/dd;
e_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
c_1=rest_y%cd;
rest_y=rest_y/cd;
d_1=rest_x%dd;
rest_x=rest_x/dd;
e_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
c_2=rest_y%cd;
rest_y=rest_y/cd;
d_2=rest_x%dd;
rest_x=rest_x/dd;
e_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
c_3=rest_y%cd;
rest_y=rest_y/cd;
d_3=rest_x%dd;
rest_x=rest_x/dd;
e_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=e_0*eld_aa+a_0*ald_aa+c_0*cld_aa;
bb_d_off=d_0*dld_bb+f_0*fld_bb+b_0*bld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_1*eld_aa+a_1*ald_aa+c_1*cld_aa;
bb_d_off=d_1*dld_bb+f_1*fld_bb+b_1*bld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_2*eld_aa+a_2*ald_aa+c_2*cld_aa;
bb_d_off=d_2*dld_bb+f_2*fld_bb+b_2*bld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_3*eld_aa+a_3*ald_aa+c_3*cld_aa;
bb_d_off=d_3*dld_bb+f_3*fld_bb+b_3*bld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_44_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
size_t stream;
size_t gld_aa,eld_aa,ald_aa,cld_aa,dld_bb,fld_bb,gld_bb,bld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*ed*fd*sizeof(double);
size_aa=gd*ed*ad*cd*sizeof(double);
size_bb=dd*fd*gd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_44_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
eld_aa=gd;
ald_aa=ed*gd;
cld_aa=ad*ed*gd;
dld_bb=1;
fld_bb=dd;
gld_bb=fd*dd;
bld_bb=gd*fd*dd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
eld_cc=dd*cd*bd*ad;
fld_cc=ed*dd*cd*bd*ad;
int total_x = dd*fd*bd;
int total_y = ed*ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_44_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,ed,fd,gd,gld_aa,eld_aa,ald_aa,cld_aa,dld_bb,fld_bb,gld_bb,bld_bb,ald_cc,bld_cc,cld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_44_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_44_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,d,e,f] += aa[g,e,b] * bb[d,f,g,a]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_45_kernel(int ad,int bd,int dd,int ed,int fd,int gd,int gld_aa,int eld_aa,int bld_aa,int dld_bb,int fld_bb,int gld_bb,int ald_bb,int ald_cc,int bld_cc,int dld_cc,int eld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,d_0,d_1,d_2,d_3,e_0,e_1,e_2,e_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
d_0=rest_x%dd;
rest_x=rest_x/dd;
e_0=rest_y;
f_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
d_1=rest_x%dd;
rest_x=rest_x/dd;
e_1=rest_y;
f_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
d_2=rest_x%dd;
rest_x=rest_x/dd;
e_2=rest_y;
f_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
d_3=rest_x%dd;
rest_x=rest_x/dd;
e_3=rest_y;
f_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=e_0*eld_aa+b_0*bld_aa;
bb_d_off=d_0*dld_bb+f_0*fld_bb+a_0*ald_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_1*eld_aa+b_1*bld_aa;
bb_d_off=d_1*dld_bb+f_1*fld_bb+a_1*ald_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_2*eld_aa+b_2*bld_aa;
bb_d_off=d_2*dld_bb+f_2*fld_bb+a_2*ald_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=e_3*eld_aa+b_3*bld_aa;
bb_d_off=d_3*dld_bb+f_3*fld_bb+a_3*ald_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+e_3*eld_cc+f_0*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+e_2*eld_cc+f_0*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+e_1*eld_cc+f_0*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+e_0*eld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+e_3*eld_cc+f_1*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+e_2*eld_cc+f_1*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+e_1*eld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+e_0*eld_cc+f_1*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+b_3*bld_cc+d_2*dld_cc+e_3*eld_cc+f_2*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+e_2*eld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+e_1*eld_cc+f_2*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+e_0*eld_cc+f_2*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+d_3*dld_cc+e_3*eld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+d_3*dld_cc+e_2*eld_cc+f_3*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+e_1*eld_cc+f_3*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+e_0*eld_cc+f_3*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_45_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
bd=bd*cd;
size_t stream;
size_t gld_aa,eld_aa,bld_aa,dld_bb,fld_bb,gld_bb,ald_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*dd*ed*fd*sizeof(double);
size_aa=gd*ed*bd*sizeof(double);
size_bb=dd*fd*gd*ad*sizeof(double);
cudaFuncSetCacheConfig(tccg_45_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
eld_aa=gd;
bld_aa=ed*gd;
dld_bb=1;
fld_bb=dd;
gld_bb=fd*dd;
ald_bb=gd*fd*dd;
ald_cc=1;
bld_cc=ad;
dld_cc=bd*ad;
eld_cc=dd*bd*ad;
fld_cc=ed*dd*bd*ad;
int total_x = dd*fd*ad;
int total_y = ed*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_45_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,dd,ed,fd,gd,gld_aa,eld_aa,bld_aa,dld_bb,fld_bb,gld_bb,ald_bb,ald_cc,bld_cc,dld_cc,eld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_45_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_45_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,c,d,f] += aa[g,f,a] * bb[d,g,c]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_46_kernel(int ad,int cd,int dd,int fd,int gd,int gld_aa,int fld_aa,int ald_aa,int dld_bb,int gld_bb,int cld_bb,int ald_cc,int cld_cc,int dld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
c_0=rest_x%cd;
rest_x=rest_x/cd;
f_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
c_1=rest_x%cd;
rest_x=rest_x/cd;
f_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
c_2=rest_x%cd;
rest_x=rest_x/cd;
f_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
c_3=rest_x%cd;
rest_x=rest_x/cd;
f_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=f_0*fld_aa+a_0*ald_aa;
bb_d_off=d_0*dld_bb+c_0*cld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_1*fld_aa+a_1*ald_aa;
bb_d_off=d_1*dld_bb+c_1*cld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_2*fld_aa+a_2*ald_aa;
bb_d_off=d_2*dld_bb+c_2*cld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_3*fld_aa+a_3*ald_aa;
bb_d_off=d_3*dld_bb+c_3*cld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+c_0*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+c_0*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+c_0*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+c_1*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+c_1*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_1*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+c_2*cld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+c_2*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_2*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+c_3*cld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+c_3*cld_cc+d_3*dld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+c_3*cld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+c_3*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+c_3*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_46_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
ad=ad*bd;
dd=dd*ed;
size_t stream;
size_t gld_aa,fld_aa,ald_aa,dld_bb,gld_bb,cld_bb,ald_cc,cld_cc,dld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*cd*dd*fd*sizeof(double);
size_aa=gd*fd*ad*sizeof(double);
size_bb=dd*gd*cd*sizeof(double);
cudaFuncSetCacheConfig(tccg_46_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
fld_aa=gd;
ald_aa=fd*gd;
dld_bb=1;
gld_bb=dd;
cld_bb=gd*dd;
ald_cc=1;
cld_cc=ad;
dld_cc=cd*ad;
fld_cc=dd*cd*ad;
int total_x = dd*cd;
int total_y = fd*ad;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_46_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,cd,dd,fd,gd,gld_aa,fld_aa,ald_aa,dld_bb,gld_bb,cld_bb,ald_cc,cld_cc,dld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_46_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_46_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,c,d,f] += aa[g,f,a,c] * bb[d,g,b]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_47_kernel(int ad,int bd,int cd,int dd,int fd,int gd,int gld_aa,int fld_aa,int ald_aa,int cld_aa,int dld_bb,int gld_bb,int bld_bb,int ald_cc,int bld_cc,int cld_cc,int dld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,c_0,c_1,c_2,c_3,d_0,d_1,d_2,d_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_y%ad;
rest_y=rest_y/ad;
b_0=rest_x%bd;
rest_x=rest_x/bd;
c_0=rest_y%cd;
rest_y=rest_y/cd;
f_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_y%ad;
rest_y=rest_y/ad;
b_1=rest_x%bd;
rest_x=rest_x/bd;
c_1=rest_y%cd;
rest_y=rest_y/cd;
f_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_y%ad;
rest_y=rest_y/ad;
b_2=rest_x%bd;
rest_x=rest_x/bd;
c_2=rest_y%cd;
rest_y=rest_y/cd;
f_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_y%ad;
rest_y=rest_y/ad;
b_3=rest_x%bd;
rest_x=rest_x/bd;
c_3=rest_y%cd;
rest_y=rest_y/cd;
f_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=f_0*fld_aa+a_0*ald_aa+c_0*cld_aa;
bb_d_off=d_0*dld_bb+b_0*bld_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_1*fld_aa+a_1*ald_aa+c_1*cld_aa;
bb_d_off=d_1*dld_bb+b_1*bld_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_2*fld_aa+a_2*ald_aa+c_2*cld_aa;
bb_d_off=d_2*dld_bb+b_2*bld_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_3*fld_aa+a_3*ald_aa+c_3*cld_aa;
bb_d_off=d_3*dld_bb+b_3*bld_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_3*ald_cc+b_0*bld_cc+c_3*cld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_2*ald_cc+b_0*bld_cc+c_2*cld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_1*ald_cc+b_0*bld_cc+c_1*cld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+c_0*cld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_3*ald_cc+b_1*bld_cc+c_3*cld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_2*ald_cc+b_1*bld_cc+c_2*cld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+c_1*cld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_1*bld_cc+c_0*cld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_3*ald_cc+b_2*bld_cc+c_3*cld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+c_2*cld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_1*ald_cc+b_2*bld_cc+c_1*cld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_2*bld_cc+c_0*cld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+c_3*cld_cc+d_3*dld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_2*ald_cc+b_3*bld_cc+c_2*cld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_1*ald_cc+b_3*bld_cc+c_1*cld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_3*bld_cc+c_0*cld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_47_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
dd=dd*ed;
size_t stream;
size_t gld_aa,fld_aa,ald_aa,cld_aa,dld_bb,gld_bb,bld_bb,ald_cc,bld_cc,cld_cc,dld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*cd*dd*fd*sizeof(double);
size_aa=gd*fd*ad*cd*sizeof(double);
size_bb=dd*gd*bd*sizeof(double);
cudaFuncSetCacheConfig(tccg_47_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
fld_aa=gd;
ald_aa=fd*gd;
cld_aa=ad*fd*gd;
dld_bb=1;
gld_bb=dd;
bld_bb=gd*dd;
ald_cc=1;
bld_cc=ad;
cld_cc=bd*ad;
dld_cc=cd*bd*ad;
fld_cc=dd*cd*bd*ad;
int total_x = dd*bd;
int total_y = fd*ad*cd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_47_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,cd,dd,fd,gd,gld_aa,fld_aa,ald_aa,cld_aa,dld_bb,gld_bb,bld_bb,ald_cc,bld_cc,cld_cc,dld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_47_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_47_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}
/*----------------------------------------------------------------------*
 *cc[a,b,d,f] += aa[g,f,b] * bb[d,g,a]
 *----------------------------------------------------------------------*/
#define T1 16
#define T2 16
#define Tcomm 16
__global__ void tccg_48_kernel(int ad,int bd,int dd,int fd,int gd,int gld_aa,int fld_aa,int bld_aa,int dld_bb,int gld_bb,int ald_bb,int ald_cc,int bld_cc,int dld_cc,int fld_cc,double *cc_d, double *aa_d, double *bb_d,int unused_idx, int total_x, int total_y) {
int a_0,a_1,a_2,a_3,b_0,b_1,b_2,b_3,d_0,d_1,d_2,d_3,f_0,f_1,f_2,f_3,g;
double a1,b1;
double a2,b2;
double a3,b3;
double a4,b4;
int in1_idxl,in2_idxl,gl,gT;
__shared__ double aa_shm[4*T1][Tcomm];
__shared__ double bb_shm[Tcomm][4*T2];
int rest_x=blockIdx.x;
int rest_y=blockIdx.y;
int thread_x = T2*4 * rest_x + threadIdx.x;
int thread_y = T1*4 * rest_y + threadIdx.y;
in1_idxl=threadIdx.y;
in2_idxl=threadIdx.x ;
double tlocal1=0;
double tlocal2=0;
double tlocal3=0;
double tlocal4=0;
double tlocal5=0;
double tlocal6=0;
double tlocal7=0;
double tlocal8=0;
double tlocal9=0;
double tlocal10=0;
double tlocal11=0;
double tlocal12=0;
double tlocal13=0;
double tlocal14=0;
double tlocal15=0;
double tlocal16=0;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*0;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*0;
a_0=rest_x%ad;
rest_x=rest_x/ad;
b_0=rest_y%bd;
rest_y=rest_y/bd;
f_0=rest_y;
d_0=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*1;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*1;
a_1=rest_x%ad;
rest_x=rest_x/ad;
b_1=rest_y%bd;
rest_y=rest_y/bd;
f_1=rest_y;
d_1=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*2;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*2;
a_2=rest_x%ad;
rest_x=rest_x/ad;
b_2=rest_y%bd;
rest_y=rest_y/bd;
f_2=rest_y;
d_2=rest_x;
rest_x = T2 *4* blockIdx.x + threadIdx.x+T1*3;
rest_y = T1 *4* blockIdx.y + threadIdx.y+T1*3;
a_3=rest_x%ad;
rest_x=rest_x/ad;
b_3=rest_y%bd;
rest_y=rest_y/bd;
f_3=rest_y;
d_3=rest_x;
int aa_d_off, bb_d_off;for(gT=0;gT<gd;gT+=Tcomm){int gl_hi;
gl_hi = MIN(Tcomm+gT,gd)-gT;
aa_d_off=f_0*fld_aa+b_0*bld_aa;
bb_d_off=d_0*dld_bb+a_0*ald_bb;
if(thread_y+T1*0<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*0][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*0<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*0] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_1*fld_aa+b_1*bld_aa;
bb_d_off=d_1*dld_bb+a_1*ald_bb;
if(thread_y+T1*1<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*1][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*1<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*1] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_2*fld_aa+b_2*bld_aa;
bb_d_off=d_2*dld_bb+a_2*ald_bb;
if(thread_y+T1*2<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*2][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*2<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*2] = bb_d[bb_d_off+g*gld_bb];
}
aa_d_off=f_3*fld_aa+b_3*bld_aa;
bb_d_off=d_3*dld_bb+a_3*ald_bb;
if(thread_y+T1*3<total_y)for(gl=threadIdx.x;gl<gl_hi;gl+=blockDim.x){
g=gl+gT;
aa_shm[in1_idxl+T1*3][gl] = aa_d[aa_d_off+g*gld_aa];
}
if(thread_x+T1*3<total_x)for(gl=threadIdx.y;gl<gl_hi;gl+=blockDim.y){
g=gl+gT;
bb_shm[gl][in2_idxl+T1*3] = bb_d[bb_d_off+g*gld_bb];
}
__syncthreads();
for(gl=0;gl<gl_hi;++gl){
a1=aa_shm[in1_idxl+T1*0][gl];
a2=aa_shm[in1_idxl+T1*1][gl];
a3=aa_shm[in1_idxl+T1*2][gl];
a4=aa_shm[in1_idxl+T1*3][gl];
b1=bb_shm[gl][in2_idxl+T2*0];
b2=bb_shm[gl][in2_idxl+T2*1];
b3=bb_shm[gl][in2_idxl+T2*2];
b4=bb_shm[gl][in2_idxl+T2*3];
tlocal1+=a1*b1;
tlocal2+=a2*b1;
tlocal3+=a3*b1;
tlocal4+=a4*b1;
tlocal5+=a1*b2;
tlocal6+=a2*b2;
tlocal7+=a3*b2;
tlocal8+=a4*b2;
tlocal9+=a1*b3;
tlocal10+=a2*b3;
tlocal11+=a3*b3;
tlocal12+=a4*b3;
tlocal13+=a1*b4;
tlocal14+=a2*b4;
tlocal15+=a3*b4;
tlocal16+=a4*b4;
}
__syncthreads();
}
if(thread_x+T1*0<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal3;
cc_d[a_0*ald_cc+b_3*bld_cc+d_0*dld_cc+f_3*fld_cc]=tlocal4;
}
else if(thread_y+T2*2<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
cc_d[a_0*ald_cc+b_2*bld_cc+d_0*dld_cc+f_2*fld_cc]=tlocal3;
}
else if(thread_y+T2*1<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
cc_d[a_0*ald_cc+b_1*bld_cc+d_0*dld_cc+f_1*fld_cc]=tlocal2;
}
else if(thread_y+T2*0<total_y){
cc_d[a_0*ald_cc+b_0*bld_cc+d_0*dld_cc+f_0*fld_cc]=tlocal1;
}
}
if(thread_x+T1*1<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal7;
cc_d[a_1*ald_cc+b_3*bld_cc+d_1*dld_cc+f_3*fld_cc]=tlocal8;
}
else if(thread_y+T2*2<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
cc_d[a_1*ald_cc+b_2*bld_cc+d_1*dld_cc+f_2*fld_cc]=tlocal7;
}
else if(thread_y+T2*1<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
cc_d[a_1*ald_cc+b_1*bld_cc+d_1*dld_cc+f_1*fld_cc]=tlocal6;
}
else if(thread_y+T2*0<total_y){
cc_d[a_1*ald_cc+b_0*bld_cc+d_1*dld_cc+f_0*fld_cc]=tlocal5;
}
}
if(thread_x+T1*2<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
cc_d[a_2*ald_cc+b_3*bld_cc+d_2*dld_cc+f_3*fld_cc]=tlocal12;
}
else if(thread_y+T2*2<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
cc_d[a_2*ald_cc+b_2*bld_cc+d_2*dld_cc+f_2*fld_cc]=tlocal11;
}
else if(thread_y+T2*1<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
cc_d[a_2*ald_cc+b_1*bld_cc+d_2*dld_cc+f_1*fld_cc]=tlocal10;
}
else if(thread_y+T2*0<total_y){
cc_d[a_2*ald_cc+b_0*bld_cc+d_2*dld_cc+f_0*fld_cc]=tlocal9;
}
}
if(thread_x+T1*3<total_x){
if(thread_y+T2*3<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal15;
cc_d[a_3*ald_cc+b_3*bld_cc+d_3*dld_cc+f_3*fld_cc]=tlocal16;
}
else if(thread_y+T2*2<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
cc_d[a_3*ald_cc+b_2*bld_cc+d_3*dld_cc+f_2*fld_cc]=tlocal15;
}
else if(thread_y+T2*1<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
cc_d[a_3*ald_cc+b_1*bld_cc+d_3*dld_cc+f_1*fld_cc]=tlocal14;
}
else if(thread_y+T2*0<total_y){
cc_d[a_3*ald_cc+b_0*bld_cc+d_3*dld_cc+f_0*fld_cc]=tlocal13;
}
}
__syncthreads();
}
extern "C" void tccg_48_cuda(int ad, int bd, int cd, int dd, int ed, int fd, int gd, double *cc, double *aa, double *bb) {
bd=bd*cd;
dd=dd*ed;
size_t stream;
size_t gld_aa,fld_aa,bld_aa,dld_bb,gld_bb,ald_bb,ald_cc,bld_cc,dld_cc,fld_cc;
size_t size_cc,size_block_cc,size_el_block_cc,size_aa,size_bb;
size_t size_block_in,size_el_block_in;
cudaStream_t *streams;
size_t nstreams,i;
double *cc_d,*aa_d,*bb_d,*cc_p,*in_p,st;
size_cc=ad*bd*dd*fd*sizeof(double);
size_aa=gd*fd*bd*sizeof(double);
size_bb=dd*gd*ad*sizeof(double);
cudaFuncSetCacheConfig(tccg_48_kernel, cudaFuncCachePreferShared);
nstreams=1;
size_block_cc=size_cc/nstreams;
size_el_block_cc=size_block_cc/sizeof(double);
cc_d=(double*)getGpuMem(size_cc);
aa_d=(double*)getGpuMem(size_aa);
bb_d=(double*)getGpuMem(size_bb);
cc_p=(double*)getHostMem(size_cc);
streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
assert(streams!= NULL);
for(i=0;i<nstreams;++i) {
cutilSafeCall(cudaStreamCreate(&streams[i])) ;
}
cutilSafeCall(cudaMemcpy(aa_d,aa,size_aa,cudaMemcpyHostToDevice));
cutilSafeCall(cudaMemcpy(bb_d,bb,size_bb,cudaMemcpyHostToDevice));
gld_aa=1;
fld_aa=gd;
bld_aa=fd*gd;
dld_bb=1;
gld_bb=dd;
ald_bb=gd*dd;
ald_cc=1;
bld_cc=ad;
dld_cc=bd*ad;
fld_cc=dd*bd*ad;
int total_x = dd*ad;
int total_y = fd*bd;
dim3 dimBlock(T2,T1);dim3 dimGrid(DIV_UB(total_x,(4*T2)), DIV_UB(total_y,(4*T1)));
for(i=0;i<nstreams;++i){
tccg_48_kernel<<<dimGrid,dimBlock,0,streams[i]>>>(ad,bd,dd,fd,gd,gld_aa,fld_aa,bld_aa,dld_bb,gld_bb,ald_bb,ald_cc,bld_cc,dld_cc,fld_cc,cc_d,aa_d,bb_d,i,total_x,total_y);
cutilCheckMsg("Kernel execution failed");
}
for(i = 0;i<nstreams;++i){
cutilSafeCall(cudaMemcpyAsync( ((char*)cc_p)+i*size_block_cc,((char*)cc_d)+i*size_block_cc,size_block_cc,cudaMemcpyDeviceToHost,streams[i]));
}
stream=0;
while(stream<nstreams) {
while(cudaStreamQuery(streams[stream])!= cudaSuccess);
double *src= &cc_p[stream*size_el_block_cc];double *dst= &cc[stream*size_el_block_cc];for (i=0;i<size_el_block_cc;++i){
dst[i] +=src[i];
}
stream++;
}
cudaThreadSynchronize();
for(i=0;i<nstreams;++i){
cudaStreamDestroy(streams[i]);}
freeGpuMem(cc_d);
freeHostMem(cc_p);
freeGpuMem(aa_d);
freeGpuMem(bb_d);
free(streams);
}
#undef T1
#undef T2
#undef Tcomm
extern "C" void tccg_48_cuda_(Integer *ad, Integer* bd, Integer* cd, Integer* dd, Integer* ed, Integer* fd, Integer* gd, double *cc, double *aa, double *bb) {
tccg_48_cuda((int)*ad,(int)*bd,(int)*cd,(int)*dd,(int)*ed,(int)*fd,(int)*gd,cc,aa,bb);
}

