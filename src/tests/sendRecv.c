#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>

/**
 *   \file sendRecv.c
 *   @brief Illustrates how to create a vector MPI datatype.
 *   @details This code is meant to be run with 2 processes: a sender
 *   a receiver. These two MPI processes will exchange a message made
 *   of integers of col_length of row size row_len = stride (Row Major Storage). 
 */

double **phi;

void assign()
{
  if ((phi = malloc(dim_x * sizeof(*phi))) == NULL)
    printf("malloc failed");
  
  if ((phi[0] = malloc(dim_x * dim_y * sizeof(**phi))) == NULL)
    printf("malloc failed");
  
  for (int x = 1; x < dim_x; x++)
  {
    phi[x] = phi[0] + x * dim_y;
    
  }

  for(int x =0; x < dim_x; x++){
    for (int y = 0; y < dim_y; y++){
      phi[x][y] = (x+1.)*(y+3.);
      printf("%.2f\t",phi[x][y]);
    }
    printf("\n");
  }
  
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if(size!=2){
    printf("This application is meant to run with 2 processes.\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  enum roles {SENDER, RECEIVER};
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int row_len = 5;

  switch(my_rank)
  {
    case SENDER:
      {
        int col_len = 3;
        int mat[row_len][col_len];
        for (int i =0; i<row_len; ++i) {
          for (int j=0; j<col_len; ++j) {
            mat[i][j] = (i+2)*(j+1);
            printf("%i\t",mat[i][j]);
          }
          printf("\n");
        }
        MPI_Type_vector(row_len,1,col_len,MPI_INT,&column_type);
        MPI_Type_commit(&column_type);

        printf("\nProcess rank %i sending vals:", my_rank);
        for (int i =0; i<row_len; ++i) printf("%i \n", mat[i][1]);

        MPI_Send(&mat[0][1],1,column_type,RECEIVER,0,MPI_COMM_WORLD);
        MPI_Type_free(&border_type);
        
        break;
      }

    case RECEIVER:
      {
        int r_mat[row_len];
        MPI_Recv(r_mat,row_len,MPI_INT,SENDER,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("\nProcess rank %i received vals:", my_rank);
        for (int i =0; i<row_len; ++i) printf("%i \n", r_mat[i]);
        break;
      }
  }

  
  MPI_Finalize();
  
  return 0;
}

