/*
 * MPI_Poisson.c
 * 2D Poison equation solver using MPI
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define DEBUG 0

#define max(a,b) ((a)>(b)?a:b)

enum
{
  X_DIR, Y_DIR
};

/* global variables */
int gridsize[2];
double precision_goal;		/* precision_goal of solution */
int max_iter, iter;		/* maximum number of iterations alowed, iters reached */

/* benchmark related variables */
clock_t ticks;			/* number of systemticks */
int timer_on = 0;		/* is timer running? */

/* local grid related variables */
double **phi;			/* grid */
int **source;			/* TRUE if subgrid element is a source */
int dim[2];			/* grid dimensions */
int offset[2]; 

// process specific variables
int np,world_rank,proc_rank, np_grid[2];   // process ids
double wtime;                   // wallclock time
int proc_coord[2];              // coords of current process in grid
int proc_top, proc_right,proc_bottom,proc_left; // ranks of neighbours
long data_communicated;

//MPI objects
MPI_Status status;
MPI_Comm grid_comm;
MPI_Datatype border_type[2];

void Setup_Proc_Grid(int argc, char **argv);
void Setup_Grid();
void Jacobi_Solve();
double Do_Step(int parity, const double w);
void GS_Solve(const double w);
void Write_Grid();
void Clean_Up();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();
void print_timer_op(void* params);

void Setup_Proc_Grid(int argc, char **argv)
{
  int wrap_around[2];
  int reorder;
  
  Debug("My_MPI_Init", 0);

  if (argc > 2)
    {
      np_grid[X_DIR] = atoi(argv[1]);
      np_grid[Y_DIR] = atoi(argv[2]);
      if(np_grid[X_DIR]*np_grid[Y_DIR] != np)
	Debug("ERROR: Process grid dimensions do not match with specified", 1);
      
    }
  else Debug("ERROR: Wrong parameter input",1);

  // Create process topology non periodic cartesian
  wrap_around[X_DIR] = 0;
  wrap_around[Y_DIR] = 0;
  reorder = 1;

  MPI_Cart_create(MPI_COMM_WORLD,2,np_grid,wrap_around,reorder,&grid_comm);
  MPI_Comm_rank(grid_comm,&proc_rank);
  MPI_Cart_coords(grid_comm,proc_rank,2,proc_coord);

  //printf(" %i (x,y)=(%i,%i)\n", proc_rank, proc_coord[X_DIR],proc_coord[Y_DIR]);

  MPI_Cart_shift(grid_comm,X_DIR,1,&proc_left,&proc_right);
  MPI_Cart_shift(grid_comm,Y_DIR,1,&proc_bottom,&proc_top);

  if(DEBUG)
    printf("(%i) top %i, right %i, bottom %i, left %i\n", proc_rank, proc_top, proc_right, proc_bottom, proc_left);

  
}

void Setup_Grid()
{
  int x, y, s;
  int upper_offset[2];
  double source_x, source_y, source_val;
  FILE *f;

  Debug("Setup_Subgrid", 0);

  if(world_rank==0){
    f = fopen("./dat/input.dat", "r");
    if (f == NULL)
      Debug("Error opening input.dat", 1);
    fscanf(f, "nx: %i\n", &gridsize[X_DIR]);
    fscanf(f, "ny: %i\n", &gridsize[Y_DIR]);
    fscanf(f, "precision goal: %lf\n", &precision_goal);
    fscanf(f, "max iterations: %i\n", &max_iter);
  }  

  MPI_Bcast(&gridsize,2,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&precision_goal,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(&max_iter,1,MPI_INT,0,MPI_COMM_WORLD);

  
  /* Calculate dimensions of local subgrid */
  offset[X_DIR] = gridsize[X_DIR]* proc_coord[X_DIR]/np_grid[X_DIR];
  upper_offset[X_DIR] = gridsize[X_DIR]*(proc_coord[X_DIR]+1)/np_grid[X_DIR];
  dim[X_DIR] = upper_offset[X_DIR] - offset[X_DIR];
  dim[X_DIR] += 2;
  
  offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR]/np_grid[Y_DIR];
  upper_offset[Y_DIR] = gridsize[Y_DIR]*(proc_coord[Y_DIR]+1)/np_grid[Y_DIR];
  dim[Y_DIR] = upper_offset[Y_DIR] - offset[Y_DIR];
  dim[Y_DIR] +=  2;

  //printf("%i X %i on proc %i\n",dim[X_DIR],dim[Y_DIR],proc_rank);

  /* allocate memory */
  if ((phi = malloc(dim[X_DIR] * sizeof(*phi))) == NULL)
    Debug("Setup_Subgrid : malloc(phi) failed", 1);
  if ((source = malloc(dim[X_DIR] * sizeof(*source))) == NULL)
    Debug("Setup_Subgrid : malloc(source) failed", 1);
  if ((phi[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**phi))) == NULL)
    Debug("Setup_Subgrid : malloc(*phi) failed", 1);
  if ((source[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**source))) == NULL)
    Debug("Setup_Subgrid : malloc(*source) failed", 1);
  for (x = 1; x < dim[X_DIR]; x++)
  {
    phi[x] = phi[0] + x * dim[Y_DIR];
    source[x] = source[0] + x * dim[Y_DIR];
  }

  /* set all values to '0' */
  for (x = 0; x < dim[X_DIR]; x++)
    for (y = 0; y < dim[Y_DIR]; y++)
    {
      phi[x][y] = 0.;
      source[x][y] = 0;
    }

  //put sources in field
  do
  {
    if(world_rank==0)
      s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y, &source_val);
    MPI_Bcast(&s,1,MPI_INT,0,MPI_COMM_WORLD);
    if (s==3)
    {
      MPI_Bcast(&source_x,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&source_y,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&source_val,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      x = source_x * gridsize[X_DIR];
      y = source_y * gridsize[Y_DIR];
      x += 1;
      y += 1;

      x = x - offset[X_DIR];
      y = y - offset[Y_DIR];
      if(x > 0 && x < dim[X_DIR] - 1 &&
         y > 0 && y < dim[Y_DIR] - 1)
      {
        phi[x][y] = source_val;
        source[x][y] = 1;
      }
      
    }
  }
  while (s==3);
  
  if(world_rank==0) fclose(f);
}

void Setup_MPI_Datatypes()
{
  Debug("Setup_MPI_Datatypes", 0);

  //Datatype for vertical data exchange
  MPI_Type_vector(dim[X_DIR]-2, 1 ,dim[Y_DIR],
		  MPI_DOUBLE,&border_type[Y_DIR]);
  MPI_Type_commit(&border_type[Y_DIR]);

  //Datatype for horizontal data exchange
  MPI_Type_vector(dim[Y_DIR]-2,1,1,
		  MPI_DOUBLE, &border_type[X_DIR]);
  MPI_Type_commit(&border_type[X_DIR]);
}

void Exchange_Borders()
{

  resume_timer();
  Debug("Exchange_Borders",0);
  
  MPI_Sendrecv(&phi[1][dim[Y_DIR]-2],1,border_type[Y_DIR], proc_top,1,
	       &phi[1][0],1,border_type[Y_DIR], proc_bottom, 1,
	       grid_comm, &status);
  
  MPI_Sendrecv(&phi[1][1],1,border_type[Y_DIR],proc_bottom,2,
               &phi[1][dim[Y_DIR]-1],1,border_type[Y_DIR],proc_top,2,
               grid_comm,&status);
  
  MPI_Sendrecv(&phi[1][1],1,border_type[X_DIR],proc_left,3,
               &phi[dim[X_DIR]-1][1],1,border_type[X_DIR],proc_right,3,
               grid_comm,&status);

  MPI_Sendrecv(&phi[dim[X_DIR]-2][1],1,border_type[X_DIR],proc_right,4,
               &phi[0][1],1,border_type[X_DIR],proc_left,4,
               grid_comm,&status);
  stop_timer();
  data_communicated += (2 * dim[X_DIR] + 2 * dim[Y_DIR] - 8);
  
}

// Jacobi solver routine for comparision 
void Jacobi_Solve()
{
  iter = 0;
  double global_delta;
  int x, y;
  double old_phi;
  double max_err;

  Debug("Jacobi_Solve", 0);

  /* give global_delta a higher value then precision_goal */
  global_delta = 2 * precision_goal;
  while (global_delta > precision_goal && iter < max_iter)
    {
      max_err = 0.0;
      /* calculate interior of grid */
      for (x = 1; x < dim[X_DIR] - 1; x++)
	for (y = 1; y < dim[Y_DIR] - 1; y++)
	  if(source[x][y] != 1)
	    {
	      old_phi = phi[x][y];
	      phi[x][y] = (phi[x + 1][y] + phi[x - 1][y] +
			   phi[x][y + 1] + phi[x][y - 1]) * 0.25;
	      if (max_err < fabs(old_phi - phi[x][y]))
		max_err = fabs(old_phi - phi[x][y]);
	    }
      Exchange_Borders();
      
      MPI_Allreduce(&max_err, &global_delta,1, MPI_DOUBLE, MPI_MAX, grid_comm);
      if(DEBUG)
	printf("delta = %f\tglobal_delta= %f\n",max_err, global_delta);
    
      iter++;
    }
  
  //printf("Number of iterations : %i at process %i\n",iter, proc_rank);
}
  

double Do_Step(int parity, const double w)
{
  int modified=1;
    
  int x, y;
  double old_phi;
  double max_err = 0.0, c = 0.0;

  int skip=0;
  
  /* calculate interior of grid */

  //modified sweep
  if (modified){
    for (x = 1; x < dim[X_DIR] - 1; x++){
      skip = (x+offset[X_DIR]+offset[Y_DIR]+parity+1)%2;
      for (int y=1 + skip; y<dim[Y_DIR]; y+=2) {
	if(source[x][y] != 1)
	  {
	    old_phi = phi[x][y];
	    c = (phi[x + 1][y] + phi[x - 1][y] +
		 phi[x][y + 1] + phi[x][y - 1]) * 0.25;
	    phi[x][y] = (1.0 - w)*old_phi + w*c; 
	    if (max_err < fabs(old_phi - phi[x][y]))
	      max_err = fabs(old_phi - phi[x][y]); 
	  } 
      }
    }
  }
  //sweep over all grid points
  else {    
    for (x = 1; x < dim[X_DIR] - 1; x++)
      for (y = 1; y < dim[Y_DIR] - 1; y++)
	if ((x + offset[X_DIR] + y + offset[Y_DIR]) % 2 == parity && source[x][y] != 1)
	  {
	    old_phi = phi[x][y];
	    c = (phi[x + 1][y] + phi[x - 1][y] +
		 phi[x][y + 1] + phi[x][y - 1]) * 0.25;
	    phi[x][y] = (1.0 - w)*old_phi + w*c; 
	    if (max_err < fabs(old_phi - phi[x][y]))
	      max_err = fabs(old_phi - phi[x][y]);
	  }
  }

  return max_err;
}

void GS_Solve(const double w)
{
  iter = 0;
  double delta, global_delta;
  double delta1, delta2;
  
  Debug("Solve", 0);

  /* give global_delta a higher value then precision_goal */
  global_delta = 2 * precision_goal;
  

  while (global_delta > precision_goal && iter < max_iter)
  {
    Debug("Do_Step 0", 0);
    delta1 = Do_Step(0,w);

    
    Exchange_Borders();
    
    Debug("Do_Step 1", 0);
    delta2 = Do_Step(1,w);

    Exchange_Borders();
    
    delta = max(delta1, delta2);
    MPI_Allreduce(&delta, &global_delta,1, MPI_DOUBLE, MPI_MAX, grid_comm);

    if(DEBUG)
      printf("delta = %f\tglobal_delta= %f\n",delta, global_delta);

    /* if(world_rank==0) */
    /*   printf("%i,%f\n",iter,global_delta); */
    
    iter++;
  }
  int global_iter;
  MPI_Allreduce(&iter, &global_iter,1, MPI_INT, MPI_MAX, grid_comm);
  
  //if(world_rank==0)
  //	printf("%i,%i\n",gridsize[X_DIR],iter);
  //printf("Number of iterations : %i at process %i\n",iter, proc_rank);

}

void Write_Grid()
{
  int x, y;
  FILE *f;
  char filename[40];
  sprintf(filename,"./dat/output%i.dat",proc_rank);
  if ((f = fopen(filename, "w")) == NULL)
    Debug("Write_Grid : fopen failed", 1);

  Debug("Write_Grid", 0);
  
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      fprintf(f, "%i %i %f\n", x + offset[X_DIR], y + offset[Y_DIR], phi[x][y]);

  fclose(f);
}

//Wrote a custom writefile function that parallely writes all results into one file
void Write_Grid_MPI()
{

  MPI_Datatype localarray;

  int nrows = (gridsize[X_DIR])*(gridsize[Y_DIR]);
  int locnrows = (dim[X_DIR]-2)*(dim[Y_DIR]-2);
  int startrow = world_rank * locnrows;
  int endrow = startrow + locnrows - 1;
  if (world_rank == np-1){
    endrow = nrows - 1;
    locnrows = endrow - startrow + 1;
  }

  //allocate for plot data
  double data[locnrows][3];
  
  //fill the data into one matrix
  int x, y;
  int k = 0;
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++){
      k = (dim[Y_DIR]-2)*(x-1)+(y-1);
      data[k][0] = x + offset[X_DIR];
      data[k][1] = y + offset[Y_DIR];
      data[k][2] = phi[x][y];
    }

  //char formatting
  MPI_Datatype num_as_string;
  const int charspernum = 9;
  char *const fmt="%8.3f ";
  char *const endfmt="%8.3f\n";
  MPI_Type_contiguous(charspernum,MPI_CHAR,&num_as_string);
  MPI_Type_commit(&num_as_string);

  char *data_as_txt = malloc(locnrows*3*charspernum*sizeof(char));
  int count = 0;
  for (int i=0; i<locnrows; i++) {
    for (int j=0; j<2; j++) {
      sprintf(&data_as_txt[count*charspernum], fmt, data[i][j]);
      count++;
    }
    sprintf(&data_as_txt[count*charspernum], endfmt, data[i][2]);
    count++;
  }

  //printf("%d: %s\n", world_rank, data_as_txt);

  //create type for localarray
  int globalsizes[2] = {nrows,3};
  int localsizes[2] = {locnrows, 3};
  int starts[2] = {startrow, 0};

  MPI_Type_create_subarray(2, globalsizes, localsizes, starts, MPI_ORDER_C, num_as_string, &localarray);
  MPI_Type_commit(&localarray);

  //write to file
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, "Poisson_MPI.dat",
                MPI_MODE_CREATE|MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);
  MPI_File_set_view(fh,0,MPI_CHAR,localarray,
                    "native",MPI_INFO_NULL);
  MPI_File_write_all(fh, data_as_txt, locnrows*3, num_as_string, &status);
  MPI_File_close(&fh);

  MPI_Type_free(&localarray);
  MPI_Type_free(&num_as_string);
  free(data_as_txt);
  
  Debug("Write_Grid", 0);
  
}



void start_timer()
{
  if (!timer_on)
  {
    MPI_Barrier(grid_comm);
    ticks = clock();
    wtime = MPI_Wtime();
    timer_on = 1;
  }
}

void resume_timer()
{
  if (!timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 1;
  }
}

void stop_timer()
{
  if (timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 0;
  }
}

void print_timer()
{
  if (timer_on)
  {
    stop_timer();
    printf("Elapsed wtime: %14.6f s (%5.1f%% CPU)\n", proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC)/wtime);
    resume_timer();
  }
  else
    printf("Elapsed wtime: %14.6f s (%5.1f%% CPU)\n", proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC)/wtime);
}

void print_timer_op(void* params)
{
  double time;
  if (timer_on)
    {
      stop_timer();
      MPI_Allreduce(&wtime, &time,1, MPI_DOUBLE, MPI_MAX, grid_comm);
      if(world_rank==0)
	printf("%i,%i,%f\n",gridsize[X_DIR],iter,time);
      resume_timer();
    }
  else
    {
      MPI_Allreduce(&wtime, &time,1, MPI_DOUBLE, MPI_MAX, grid_comm);
      if(world_rank==0)
	printf("%i,%f",gridsize[X_DIR],wtime);
    }
  
}

void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("%s\n", mesg);
  if (terminate)
    exit(1);
}

void Clean_Up()
{
  Debug("Clean_Up", 0);

  free(phi[0]);
  free(phi);
  free(source[0]);
  free(source);
  MPI_Comm_free(&grid_comm);
  MPI_Type_free(&border_type[0]);
  MPI_Type_free(&border_type[1]);
}

int main(int argc, char **argv)
{
  const double w = 1.95;
  int size_mpi_double;
  long data_communicated_bytes;
  long global_communicated_bytes = 0;
  
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD,&np);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);


  Setup_Proc_Grid(argc,argv); //cartesian topology  
  
  Setup_Grid();
  
  Setup_MPI_Datatypes();

  //Jacobi_Solve(); //Jacobi Solver
  //start_timer();
  GS_Solve(w); //Gauss Siedel Solver with SOR, (w = 1, Gauss Siedel w/o SOR) 

  print_timer_op(NULL);

  //Data communicated
  MPI_Type_size(MPI_DOUBLE, &size_mpi_double);
  data_communicated_bytes = data_communicated * size_mpi_double;
  MPI_Reduce(&data_communicated_bytes, &global_communicated_bytes, 1, MPI_LONG, MPI_SUM, 0, grid_comm);

  /* if (world_rank == 0) { */
  /*   printf(",%f\n", global_communicated_bytes*1.e-6); */
  /* } */
  //Write_Grid_MPI();

  //print_timer();
  
  Clean_Up();

  MPI_Finalize();
  
  return 0;
}
