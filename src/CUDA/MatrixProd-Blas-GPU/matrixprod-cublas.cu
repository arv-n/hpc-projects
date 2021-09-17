/* Simple matrix vector Multiplication example using CuBlas
 * 
 * Performs the operation : b = A * x
 * 
 * - 'x' and 'b' are two vectors with size N;
 * - 'B' is the a square matrix with size NxN;
 * 
 * Kees Lemmens, Jan 2012
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

float *initHostMatrix(int n, int m);
void fillMatrix(float *a, int n, int m, int offset);
void showMatrix(const char *name, float *a, int n, int m);
float *initCublasMatrix(int n, int m);
void copytoCublasMatrix(float *d_a, float *h_a, int n, int m);
void copyfromCublasMatrix(float *h_a, float *d_a, int n, int m);
void freeCudaDevice(float *a);
void freeCudaHost(float *a);

float *initHostVector(int n);
void fillVector(float *a, int n, int offset);
void showVector(const char *name, float *a, int n);
float *initCublasVector(int n);
void copytoCublasVector(float *d_a, float *h_a, int n);
void copyfromCublasVector(float *h_a, float *d_a, int n);

int main(int argc, char** argv)
{
   struct timespec ti1,ti2,ti3,ti4;
   double runtime;
   cublasStatus_t status;
   cublasHandle_t handle;
   float alpha = 1, beta = 0;
   
   int dim = 2048;
   
   float *h_A,     *h_x,     *h_b;
   float *d_A = 0, *d_x = 0, *d_b = 0;
   
   if(argc >=2 )
     sscanf(argv[1],"%d",&dim);
   
   status = cublasCreate(&handle);
   checkCublasStatus(status,"Init has failed !");
   
   fprintf(stderr,"Matrix-vector product with dim = %d\n",dim);
   
   // Allocate host memory for the data
   h_A = initHostMatrix(dim,dim);
   h_x = initHostVector(dim);
   h_b = initHostVector(dim);
   
   fillMatrix(h_A,dim,dim, 0);
   fillVector(h_x,dim, 10);
   
   // Allocate device memory for the data
   d_A = initCublasMatrix(dim,dim);
   d_x = initCublasVector(dim);
   d_b = initCublasVector(dim);
   
   clock_gettime(CLOCK_REALTIME,&ti1);        // read starttime into t1
   
   // Initialize the device data with the host data
   copytoCublasMatrix(d_A,h_A,dim,dim);
   copytoCublasVector(d_x,h_x,dim);
   
   // Performs operation using cublas
   
   clock_gettime(CLOCK_REALTIME,&ti2);        // read endtime into t2
   
   status = cublasSgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, d_A, dim, d_x, 1, &beta, d_b, 1);
   checkCublasStatus(status,"Kernel execution error !");

   cudaDeviceSynchronize(); // Make sure all threads finished before stopping clock
   clock_gettime(CLOCK_REALTIME,&ti3);        // read endtime into t3
   
   // Read the result back
   copyfromCublasVector(h_b,d_b,dim);
   clock_gettime(CLOCK_REALTIME,&ti4);        // read endtime into t4
   
   showMatrix("A",h_A,dim,dim);
   showVector("x",h_x,dim);
   showVector("b",h_b,dim);
   
   cublasDestroy(handle);

   // Memory clean up
   freeCudaDevice(d_A);
   freeCudaDevice(d_x);
   freeCudaDevice(d_b);
   
   freeCudaHost(h_A);
   freeCudaHost(h_x);
   freeCudaHost(h_b);
   
   fflush(stderr);
   runtime = (ti3.tv_sec - ti2.tv_sec) + 1e-9*(ti3.tv_nsec - ti2.tv_nsec);
   fprintf(stderr,"\nCublas : run time only for Sgemv = %.2e secs.\n",runtime);
  
   runtime = (ti4.tv_sec - ti1.tv_sec) + 1e-9*(ti4.tv_nsec - ti1.tv_nsec);
   fprintf(stderr,"\nCublas : run time including data transfers = %.2e secs.\n",runtime);
   
   return EXIT_SUCCESS;
}
