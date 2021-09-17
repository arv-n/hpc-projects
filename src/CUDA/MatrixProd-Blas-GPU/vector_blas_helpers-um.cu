/* Some common routines for allocating Blas vectors,
 * filling them with some data and printing them.
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

float *initManagedVector(int n)
{
   float *ptr;
   
   cudaMallocManaged(&ptr, n * sizeof(float)); // rows
   
   if (ptr == NULL)
   {
      fprintf(stderr,"Malloc for unified memory vector on host failed !\n");
      exit(1);
   }
   
   return ptr;
}

void freeManagedVector(float *ptr)
{
   cudaFree(ptr);
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

