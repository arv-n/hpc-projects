#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Input Array Variables
float* h_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* h_VecW = NULL;
float* h_NormW = NULL;

// Variables to change
int GlobalSize = 5000;         // this is the dimension of the matrix, GlobalSize*GlobalSize
const float EPS = 0.000005;    // tolerence of the error
int max_iteration = 100;       // the maximum iteration steps

// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void  Arguments(int, char**);

void CPU_AvProduct()
{
  int N = GlobalSize;
  int matIndex =0;
  for(int i=0;i<N;i++)
  {
    h_VecW[i] = 0;
    for(int j=0;j<N;j++)
    {
      matIndex = i*N + j;
      h_VecW[i] += h_MatA[matIndex] * h_VecV[j];
			
    }
  }
}

void CPU_NormalizeW()
{
  int N = GlobalSize;
  float normW=0;
  for(int i=0;i<N;i++)
    normW += h_VecW[i] * h_VecW[i];
	
  normW = sqrt(normW);
  for(int i=0;i<N;i++)
    h_VecV[i] = h_VecW[i]/normW;
}

float CPU_ComputeLamda()
{
  int N = GlobalSize;
  float lamda =0;
  for(int i=0;i<N;i++)
    lamda += h_VecV[i] * h_VecW[i];
	
  return lamda;
}

void RunCPUPowerMethod()
{
  printf("*************************************\n");
  float oldLamda =0;
  float lamda=0;
	
  //AvProduct
  CPU_AvProduct();
	
  //power loop
  for (int i=0;i<max_iteration;i++)
  {
    CPU_NormalizeW();
    CPU_AvProduct();
    lamda= CPU_ComputeLamda();
    printf("CPU lamda at %d: %f \n", i, lamda);
    //If residual is lass than epsilon break
    if(abs(oldLamda - lamda) < EPS)
      break;
    oldLamda = lamda;	
	
  }
  printf("*************************************\n");
	
}

// Host code
int main(int argc, char** argv)
{

  struct timespec t_start,t_end;
  double runtime;
  Arguments(argc, argv);
		
  int N = GlobalSize;
  printf("Matrix size %d X %d \n", N, N);
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


  // Initialize input matrix
  UploadArray(h_MatA, N);
  InitOne(h_VecV,N);

  printf("Power method in CPU starts\n");	   
  clock_gettime(CLOCK_REALTIME,&t_start);
  RunCPUPowerMethod();   // the lamda is already solved here
  clock_gettime(CLOCK_REALTIME,&t_end);
  runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
  printf("CPU: run time = %f secs.\n",runtime);
  printf("Power method in CPU is finished\n");

  Cleanup();
  
}

void Cleanup(void)
{
  // Free host memory
  if (h_MatA)
    free(h_MatA);
  if (h_VecV)
    free(h_VecV);
  if (h_VecW)
    free(h_VecW);
  if (h_NormW)
    free(h_NormW);
    
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
