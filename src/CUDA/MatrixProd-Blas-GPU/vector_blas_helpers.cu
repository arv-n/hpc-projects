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
#include <sys/time.h>

/* Includes, cuda */
#include <cuda.h>
#include <cublas_v2.h>

void checkCublasStatus(cublasStatus_t status, const char *error);
void checkCudaError(const char *errormsg);

float *initHostVector(int n)
{
   float *ptr;
   
   cudaMallocHost(&ptr, n * sizeof(float)); // rows
   
   if (ptr == NULL)
   {
      fprintf(stderr,"Malloc for vector failed !\n");
      exit(1);
   }
   
   return ptr;
}

void fillVector(float *a, int n, int offset)
{  int x;
   
   for(x=0; x<n; x++)
     a[x] = (float) x + offset;
}

void showVector(const char *name, float *a, int n)
{
   int x;
   
# if (DEBUG > 0)
   for(x=0; x<n; x++)
# else
   x = n - 1;
# endif   
   {
      printf("%s[%02u]=%6.2f  ",name,x,a[x]);
      printf("\n");
   }
}

float *initCublasVector(int n)
{
   float *ptr = 0;
   
   cudaMalloc(&ptr, n * sizeof(float));
   checkCudaError("Malloc for vector on device failed !");
   
   return ptr;
}

void freeCublasVector(float *ptr)
{
   cudaFree(ptr);
}

void copytoCublasVector(float *d_a, float *h_a, int n)
{
   cublasStatus_t status;
   
   status = cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
   checkCublasStatus(status," Vector copy to device failed !");
}

void copyfromCublasVector(float *h_a, float *d_a, int n)
{
   cublasStatus_t status;
   
   status = cublasGetVector(n, sizeof(float), d_a, 1, h_a, 1);
   checkCublasStatus(status,"Vector copy from device failed !");
}

