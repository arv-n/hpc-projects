/* Simple matrix vector Multiplication example using CuBlas and Unified Memory (Arch 3.0 and up)
 * 
 * Performs the operation : b = A * x
 * 
 * - 'x' and 'b' are two vectors with size N;
 * - 'B' is the a square matrix with size NxN;
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Includes, cuda
#include <cuda.h>
#include <cublas_v2.h>

// These are simple routines stored in a separate source file as they are
// not really important for understanding this example.

void checkCublasStatus(cublasStatus_t status, const char *error);
void checkCudaError(const char *errormsg);

float *initManagedMatrix(int n, int m);
void freeManagedMatrix(float *ptr);
void fillMatrix(float *a, int n, int m, int offset);
void showMatrix(const char *name, float *a, int n, int m);

float *initManagedVector(int n);
void freeManagedVector(float *ptr);
void fillVector(float *a, int n, int offset);
void showVector(const char *name, float *a, int n);

int main(int argc, char** argv)
{
   struct timespec ti1,ti2;
   double runtime;
   cublasStatus_t status;
   cublasHandle_t handle;
   float alpha = 1, beta = 0;
   
   int dim = 2048;

   float tmp __attribute__((unused));
   
   float *A = 0, *x = 0, *b = 0;
   
   if(argc >=2 )
     sscanf(argv[1],"%d",&dim);
   
   status = cublasCreate(&handle);
   checkCublasStatus(status,"Init has failed !");
   
   fprintf(stderr,"Matrix-vector product with dim = %d\n",dim);
      
   // Allocate unified memory for the data
   A = initManagedMatrix(dim,dim);
   x = initManagedVector(dim);
   b = initManagedVector(dim);
   
   fillMatrix(A,dim,dim, 0);
   fillVector(x,dim, 10);
   
   clock_gettime(CLOCK_REALTIME,&ti1);        // read starttime into t1
   
   status = cublasSgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, A, dim, x, 1, &beta, b, 1);
   checkCublasStatus(status,"Kernel execution error !");

   cudaDeviceSynchronize(); // Make sure all threads finished before stopping clock

   tmp = x[0]; // force a copy back to the host for x or else the time is not reliable!

   clock_gettime(CLOCK_REALTIME,&ti2);        // read endtime into t2
   
   showMatrix("A",A,dim,dim);
   showVector("x",x,dim);
   showVector("b",b,dim);
   
   cublasDestroy(handle);

   freeManagedMatrix(A);
   freeManagedVector(x);
   freeManagedVector(b);
   
   fflush(stderr);
  
   runtime = (ti2.tv_sec - ti1.tv_sec) + 1e-9*(ti2.tv_nsec - ti1.tv_nsec);
   fprintf(stderr,"\nCublas : run time including (implicit) data transfers = %.2e secs.\n",runtime);
   
   return EXIT_SUCCESS;
}
