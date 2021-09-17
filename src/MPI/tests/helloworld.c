#include <stdio.h>
#include "mpi.h"

int rank,np;
int main(int argc, char **argv)
{
  //printf("hello\n");
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&np);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  printf("Node %i of %i says:Hello World!\n",rank,np);

  MPI_Finalize();
  return 0;
}
