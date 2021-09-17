/**
 *   \file collectiveIO.c
 *   @brief Illustrates how to perform collective IO operations in MPI.
 *   @details This code is meant to create a single file with outputs from N processes using file view in ASCII string format (Expensive).
 * TODO binary format
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

float **alloc2d(int n, int m) {
    float *data = malloc(n*m*sizeof(float));
    float **array = malloc(n*sizeof(float *));
    for (int i=0; i<n; i++)
        array[i] = &(data[i*m]);
    return array;
}

int main(int argc, char **argv) {
    int ierr, rank, size;
    MPI_Offset offset;
    MPI_File   file;
    MPI_Status status;
    MPI_Datatype num_as_string;
    MPI_Datatype localarray;
    const int nrows=10;
    const int ncols=3;
    float **data;
    char *const fmt="%8.3f ";
    char *const endfmt="%8.3f\n";
    int startrow, endrow, locnrows;

    const int charspernum=9;

    ierr = MPI_Init(&argc, &argv);
    ierr|= MPI_Comm_size(MPI_COMM_WORLD, &size);
    ierr|= MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    locnrows = nrows/size;
    startrow = rank * locnrows;
    endrow = startrow + locnrows - 1;
    if (rank == size-1) {
        endrow = nrows - 1;
        locnrows = endrow - startrow + 1;
    }

    /* allocate local data */
    data = alloc2d(locnrows, ncols);

    /* fill local data */
    for (int i=0; i<locnrows; i++) 
        for (int j=0; j<ncols; j++)
            data[i][j] = rank;

    /* each number is represented by charspernum chars */
    MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string); 
    MPI_Type_commit(&num_as_string); 

    /* convert our data into txt */
    char *data_as_txt = malloc(locnrows*ncols*charspernum*sizeof(char));
    int count = 0;
    for (int i=0; i<locnrows; i++) {
        for (int j=0; j<ncols-1; j++) {
            sprintf(&data_as_txt[count*charspernum], fmt, data[i][j]);
            count++;
        }
        sprintf(&data_as_txt[count*charspernum], endfmt, data[i][ncols-1]);
        count++;
    }

    printf("%d: %s\n", rank, data_as_txt);

    /* create a type describing our piece of the array */
    int globalsizes[2] = {nrows, ncols};
    int localsizes [2] = {locnrows, ncols};
    int starts[2]      = {startrow, 0};
    int order          = MPI_ORDER_C;

    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
    MPI_Type_commit(&localarray);

    /* open the file, and set the view */
    MPI_File_open(MPI_COMM_WORLD, "all-data.txt", 
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, localarray, 
                           "native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, locnrows*ncols, num_as_string, &status);
    MPI_File_close(&file);

    MPI_Type_free(&localarray);
    MPI_Type_free(&num_as_string);

    free(data[0]);
    free(data);

    MPI_Finalize();
    return 0;
}
