/* Some common routines for allocating Blas matrices,
 * filling them with some data and printing them.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Includes, cuda */
#include <cuda.h>
#include <cublas_v2.h>


void checkCublasStatus(cublasStatus_t status, const char *error)
{
   if (status != CUBLAS_STATUS_SUCCESS)
   {
      fprintf (stderr, "Cuda CUBLAS : %s\n",error);
      exit(EXIT_FAILURE);
   }
}

void checkCudaError(const char *errormsg)
{
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess)
   {
      fprintf (stderr, "%s\n",errormsg);
      fprintf (stderr, "Cuda: %s\n",cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

float *initHostMatrix(int n, int m)
{
   float *ptr = 0;
   
   cudaMallocHost(&ptr, n * m * sizeof(float)); // rows x columns
   
   if (ptr == NULL)
   {
      fprintf(stderr,"Malloc for matrix on host failed !\n");
      exit(1);
   }
   
   return ptr;
}

// Note that we actually fill the TRANSPOSED matrix here
// as BLAS is Fortran based !!! 
void fillMatrix(float *a, int n, int m, int offset)
{  long x,y;
   
   for(y=0; y<m; y++)   // mind the order of the loops : this is 20x faster ...
     for(x=0; x<n; x++)
       a[y*n + x] = (float) x+y + offset;
}

// Note that we actually show the TRANSPOSED matrix here
// as BLAS is Fortran based !!! 
void showMatrix(const char *name, float *a, int n, int m)
{ 
   long x,y;
   
# if (DEBUG > 0)
   for(y=0; y<m; y++)
# else
   y = m - 1;
# endif
   {
# if (DEBUG > 1)
      for(x=0; x<n; x++)
# else
      x = n - 1;
# endif
      {
         printf("%s[%02ld][%02ld]=%6.2f  ",name,x,y,a[y*n + x]);
      }
      printf("\n");
   }
}

float *initCublasMatrix(int n, int m)
{
   float *ptr = 0;
   
   cudaMalloc(&ptr, n*m * sizeof(float));
   checkCudaError("Malloc for matrix on device failed !");
   
   return ptr;
}

void copytoCublasMatrix(float *d_a, float *h_a, int n, int m)
{
   cublasStatus_t status;
   
   status = cublasSetVector(n*m, sizeof(float), h_a, 1, d_a, 1);
   checkCublasStatus(status," Matrix copy to device failed !");
}

void copyfromCublasMatrix(float *h_a, float *d_a, int n, int m)
{
   cublasStatus_t status;
   
   status = cublasGetVector(n*m, sizeof(float), d_a, 1, h_a, 1);
   checkCublasStatus(status,"Matrix copy from device failed !");
}

void freeCudaHost(float *a)
{
   cudaFreeHost(a);
   checkCudaError("Memory free error on host !");
}

void freeCudaDevice(float *a)
{
   cudaFree(a);
   checkCudaError("Memory free error on device !");
}
