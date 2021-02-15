#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "cuda.h"
#include "gputimer.h" //for benchmarking


// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;
float* h_lambda = NULL;
float* d_lambda = NULL;

// Variables to change
int GlobalSize = 5000;         // this is the dimension of the matrix, GlobalSize*GlobalSize
const int BLOCK_SIZE = 32;     //number of threads per block
const float EPS = 0.000005;    // tolerence of the error
int max_iteration = 100;       // the maximum iteration steps

// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
void  Arguments(int, char**);
void checkCardVersion(void);

//Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
  //global thread index
  unsigned int globalid = threadIdx.x + blockIdx.x*blockDim.x;
  float sum = 0;
  if(globalid < N){
    for(int i=0; i<N; i++)
      sum+= g_MatA[i+(globalid*N)]*g_VecV[i];
    g_VecW[globalid]=sum;        
  }
}

__global__ void FindNormW(float* g_VecW, float * g_NormW, int N)
{
  unsigned int globalid = threadIdx.x + blockIdx.x*blockDim.x;
  if(globalid < N)
    atomicAdd(g_NormW,g_VecW[globalid]*g_VecW[globalid]);
}

__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N)
{
  unsigned int globalid = threadIdx.x + blockIdx.x*blockDim.x;
  if(globalid < N)
    g_VecV[globalid]= g_VecW[globalid]/g_NormW[0];
}

__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
  unsigned int globalid = threadIdx.x + blockIdx.x*blockDim.x;
  if(globalid < N)
    atomicAdd(g_Lamda, g_VecV[globalid] * g_VecW[globalid]);
}

// Host code
int main(int argc, char** argv)
{

  GpuTimer mem_timer, kernel_timer, total;
  float mem_time =0., kernel_time = 0.;
  
  Arguments(argc, argv);
		
  int N = GlobalSize;
  printf("%d X %d for threads/Block %d \n", N, N, BLOCK_SIZE);
  size_t vec_size = N * sizeof(float);
  size_t mat_size = N * N * sizeof(float);
  size_t norm_size = sizeof(float);
  
  // Allocate normalized value in host memory
  h_NormW = (float*)malloc(norm_size);
  // Allocate input matrix in host memory
  h_MatA = (float*)malloc(mat_size);
  // Allocate initial vector V in host memory
  h_VecV = (float*)malloc(vec_size);
  // Allocate W vector for computations
  h_VecW = (float*)malloc(vec_size);
  //Allocate lambda
  h_lambda = (float*)malloc(norm_size);

  // Set the kernel arguments
  int threadsPerBlock = BLOCK_SIZE;   
  int blocksPerGrid = N / threadsPerBlock + 1;
  
  // Allocate matrix and vectors in device memory
  cudaMalloc((void**)&d_MatA, mat_size); 
  cudaMalloc((void**)&d_VecV, vec_size); 
  cudaMalloc((void**)&d_VecW, vec_size); 
  cudaMalloc((void**)&d_NormW, norm_size); 
  cudaMalloc((void**)&d_lambda, norm_size);
  
  // Initialize input matrix
  UploadArray(h_MatA, N);
  InitOne(h_VecV,N);

  ////////////////////////////////////////////////
  // This is the starting points of GPU
  checkCardVersion();
  printf("*************************************\n");
  printf("Power Method (global mem) on GPU starts\n");
    
  total.Start(); //total runtime
  
  //Copy from host memory to device memory
  mem_timer.Start();
  cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
  mem_timer.Stop();
  mem_time += (mem_timer.Elapsed()*1e-3);

  kernel_timer.Start();
  Av_Product<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV,
                                                 d_VecW, N);
  cudaDeviceSynchronize(); //Needed, kind of barrier to sychronize all threads
  kernel_timer.Stop();
  kernel_time += (kernel_timer.Elapsed()*1e-3);
  
  //Power method loops
  float OldLamda = 0;
  
  for(int i=0; i<max_iteration;i++)
  {
    h_NormW[0]= 0;

    //Norm
    mem_timer.Start();
    cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
    mem_timer.Stop();
    mem_time += (mem_timer.Elapsed()*1e-3);

    kernel_timer.Start();
    FindNormW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW,d_NormW, N);
    cudaDeviceSynchronize();
    kernel_timer.Stop();
    kernel_time += (kernel_timer.Elapsed()*1e-3);

    //Transfer to host & back
    mem_timer.Start();
    cudaMemcpy(h_NormW,d_NormW, norm_size, cudaMemcpyDeviceToHost);
    h_NormW[0] = sqrt(h_NormW[0]);    
    cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
    mem_timer.Stop();
    mem_time += (mem_timer.Elapsed()*1e-3);

    //Normalize
    kernel_timer.Start();
    NormalizeW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW ,
                                                   d_VecV, N);
    cudaDeviceSynchronize();
    kernel_timer.Stop();
    kernel_time += (kernel_timer.Elapsed()*1e-3);
    
    
    //AvProduct
    kernel_timer.Start();
    Av_Product<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N);
    cudaDeviceSynchronize();
    kernel_timer.Stop();
    kernel_time += (kernel_timer.Elapsed()*1e-3);
    
    //Compute Lambda
    h_lambda[0]=0.;
    mem_timer.Start();
    cudaMemcpy(d_lambda, h_lambda, norm_size, cudaMemcpyHostToDevice);
    mem_timer.Stop();
    mem_time += (mem_timer.Elapsed()*1e-3);

    kernel_timer.Start();
    ComputeLamda<<<blocksPerGrid, threadsPerBlock>>>(d_VecV, d_VecW,
                                                     d_lambda , N);
    cudaDeviceSynchronize();
    kernel_timer.Stop();
    kernel_time += (kernel_timer.Elapsed()*1e-3);

    
    mem_timer.Start();
    cudaMemcpy(h_lambda, d_lambda, norm_size, cudaMemcpyDeviceToHost);
    mem_timer.Stop();
    mem_time += (mem_timer.Elapsed()*1e-3);
    
    printf("GPU lamda at %d: %f \n", i, h_lambda[0]);
    //If residual is lass than epsilon break
    if(abs(OldLamda - h_lambda[0]) < EPS)
      break;
    OldLamda = h_lambda[0];	     
    
  }

  total.Stop();
  printf("Power Method on GPU ends\n");
  printf("*************************************\n\n");
  printf("memtime,kerneltime,total\n");
  printf("%f,%f,%f\n",mem_time,kernel_time,total.Elapsed()*1e-3);  

  Cleanup();
}

//Utility functions
void Cleanup(void)
{
  // Free device memory
  if (d_MatA)
    cudaFree(d_MatA);
  if (d_VecV)
    cudaFree(d_VecV);
  if (d_VecW)
    cudaFree(d_VecW);
  if (d_NormW)
    cudaFree(d_NormW);
  if (d_lambda)
    cudaFree(d_lambda);
		
  // Free host memory
  if (h_MatA)
    free(h_MatA);
  if (h_VecV)
    free(h_VecV);
  if (h_VecW)
    free(h_VecW);
  if (h_NormW)
    free(h_NormW);
  if(h_lambda)
    free(h_lambda);
    
  exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
  for (int i = 0; i < n; i++)
    data[i] = 0;
  data[0]=1;
}

void UploadArray(float* data, int n)
{
  int total = n*n;
  int value=1;
  for (int i = 0; i < total; i++)
  {
    data[i] = (int) (rand() % (int)(101));//1;//value;
    value ++; if(value>n) value =1;
    // data[i] = 1;
  }
}

// Obtain program arguments
void Arguments(int argc, char** argv)
{
  for (int i = 0; i < argc; ++i) 
  {
    if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0)
    {
      GlobalSize = atoi(argv[i+1]);
      i = i + 1;
    }
    if (strcmp(argv[i], "--max_iteration") == 0 || strcmp(argv[i], "-max_iteration") == 0)
    {
      max_iteration = atoi(argv[i+1]);
      i = i + 1;
    }
  }
}


void checkCardVersion()
{
  cudaDeviceProp prop;
   
  cudaGetDeviceProperties(&prop, 0);
   
  printf("This GPU has major architecture %d, minor %d \n",prop.major,prop.minor);
  if(prop.major < 2)
  {
    fprintf(stderr,"Need compute capability 2 or higher.\n");
    exit(1);
  }
}




