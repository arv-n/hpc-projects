### Power method on GPU
This directory contains files to solve the eigenvalue problem using the Power method and compare the performance characteristics between the sequential and CUDA parallel versions. The essence of the algorithm lies in the efficient parallelization of the Matrix-vector Multiplication step.

To run, modify the slurm script `runpower.job` for running on any slurm based cluster. Alternatively if running on a personal machine, it is a standard compile and run using `nvcc`:

    $ nvcc nvcc -o power_gpu_glb $power_gpu_glb.cu
    $ ./power_gpu_glb -size 5000 
    
The command line option `size` defines the size of the matrix. 

| Filename    | Description |
| ------------- |:-------------:|
| power_cpu      | single-thread sequential     |
| power_gpu_glb      | CUDA global memory access     |
| power_gpu_shr      | CUDA shared memory access     |