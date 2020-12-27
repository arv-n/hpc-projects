#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define N 32

int rank,np;
int length, begin, end;
MPI_Status status;
float f[N];

void recvSend(float f);
void bCast(float f);
void reduce();
void init();
void set_f();
//void distribute_f();
void calc_g(float *g, const int length);
//void collect_g();
void show_g();

int main(int argc, char **argv)
{
  //float f = 0.;
  
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&np);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //recvSend(f);
  //bCast(f);
  //reduce();
  init();
  float g[length];
  
  if(rank==0) set_f();
  MPI_Scatter(f,length,MPI_FLOAT,g,length,MPI_FLOAT,0,MPI_COMM_WORLD); //distribute_f();
  calc_g(g,length); 
  MPI_Gather(g,length,MPI_FLOAT,&f[begin],length,MPI_FLOAT,0,MPI_COMM_WORLD); //collect_g();

  if(rank==0)
    for(int i=0; i<N; i++)
      printf("\nf[%i]=%f",i,f[i]);

  MPI_Finalize();

  return 0;
  
}

void init()
{
  length = N/np;
  begin = rank*length;
  end = begin+length-1;
}

void set_f()
{
  for(int i=0;i<N;i++){
    f[i]=sin(i*(1.0/N));
    printf("Node %i has f(%i) = %f\n",rank,i,f[i]);
  }
    
}

void calc_g(float *g, const int length)
{
  for(int i=0; i<length; i++) g[i]=2.0*g[i];
}

/* void distribute_f() */
/* { */
/*   int their_begin; */
  
/*   if(rank==0){ */
/*     for(int dest=0; dest<np; dest++){ */
/*       their_begin=dest*length; */
/*       MPI_Send(&f[their_begin],length,MPI_FLOAT,dest,12,MPI_COMM_WORLD); */
/*     } */
/*   } */
/*   else { */
/*     MPI_Recv(&f[begin],length,MPI_FLOAT,0,12,MPI_COMM_WORLD,&status); */
/*     printf("\nReceived %i nos at Node %i",length,rank); */
/*   } */
  
/* } */


/* void collect_g() */
/* { */
/*   int their_begin; */
/*   if(rank==0){ */
/*     for(int src=1; src<np; src++){ */
/*       their_begin = src*length; */
/*       MPI_Recv(&g[their_begin],length,MPI_FLOAT,src,13,MPI_COMM_WORLD,&status); */
/*       printf("\nReceived back %i nos at Node %i from Node %i", length,rank,src); */
/*     } */
/*   } */
/*   else MPI_Send(&g[begin],length,MPI_FLOAT,0,13,MPI_COMM_WORLD); */
  
/* } */


void recvSend(float f)
{
  if(rank==0){
    for(int source=1; source<np; source++){
      MPI_Recv(&f,1,MPI_FLOAT,source,source+1,MPI_COMM_WORLD,&status);
      printf("%f came from node %i \n",f,source);
    } 
  }
  else{
    for(int source=1; source<np; source++){
      f = source + 2;
      MPI_Send(&f,1,MPI_FLOAT,0,source+1,MPI_COMM_WORLD);
    }
  }

}

void bCast(float f)
{
  if(rank==0) f = 1.0;

  MPI_Bcast(&f,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  printf("printing %f from node %i\n",f,rank);
  
}

void reduce()
{
  float f,g;
  if(rank==0){
    printf("Enter a number\n");
    scanf("%f",&f);
  }
  MPI_Bcast(&f,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  f = f*f;

  printf("Node %i calculates %f\n",rank,f);
  
  MPI_Reduce(&f,&g,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

  if(rank==0) printf("\n\nFinal Val=%f",g);
    
}
