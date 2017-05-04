#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>

#define A(i,j) A[(i) + (j) * nx]

typedef double real;

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  double t1 = MPI_Wtime();
  int numprocs, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);

  if(argc<4)
  {
    std::cout << "Error(" << __LINE__ << "): Insufficient number of input arguments!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int nx= atoi(argv[1]),
  ny =nx;
  std::string method = argv[2];
  int sub_method = atoi(argv[3]);

  if(nx%numprocs!=0)
  {
    std::cout << "Error(" << __LINE__ << "): numprocs mismatch!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  real *A = new real[nx*ny];
  for(int i=0; i<nx; i++)
  {
    for(int j=0; j<ny; j++)
    {
      if(i==j)
      {
        A(i,j) = nx+1;
      }
      else
      {
        A(i,j) = 1;
      }
    }
  }

  real *b = new real[nx];
  for(int i=0; i<nx; i++)
  {
    b[i] = 2*nx;
  }

  real *x = new real[nx]();
  for(int i=0; i<nx; i++)
  {
    x[i] = 0;
  }


  if(method == "j")
  {
    int k;
    double epsilon=1e-3;
    int maxit=2*nx*nx;
    int div_size = nx/numprocs;
    real dx, y_new;
    double global_sum, local_sum;
    real *y = new real[div_size]();
    for(int i = rank * div_size; i < (rank + 1) * div_size; i++)
    {
      y[i-rank*div_size] = x[i];
    }
    for(k = 0; k < maxit; k++)
    {
      local_sum=0.0;
      for(int i = rank * div_size; i < (rank + 1) * div_size; i++)
      {
        if(sub_method==1)
        {
          dx = b[i];
          for(int j=0; j<ny; j++)
          {
            dx -= A(i,j) * x[j];
          }
          dx /= A(i,i);
          y[i-rank*div_size] += dx;
        }
        else if(sub_method==2)
        {
          y_new = b[i];
          for(int j=0; j<ny; j++)
          {
            if(j != i)
            {
              y_new -= A(i,j) * x[j];
            }
          }
          y_new /= A(i,i);
          real w = 2.0/3.0;//w==1 normal
          y_new = w  * y_new + (1.0 - w) * y[i-rank*div_size];
          dx = y[i-rank*div_size] - y_new;
          y[i-rank*div_size] = y_new;
        }
        local_sum += ((dx >= 0.0) ? dx : -dx);
      }
      MPI_Allgather(y,div_size,MPI_DOUBLE,x,div_size,MPI_DOUBLE,MPI_COMM_WORLD);
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if(global_sum <= epsilon)
      {
        break;
      }
    }
    free(y);
    if(rank==0)
    {
      std::cout << k << " " << global_sum << " " << div_size << std::endl;
    }
  }


  if(method == "g")
  {
    int k;
    double epsilon=1e-3;
    int maxit=2*nx*nx;
    int div_size = nx/numprocs;

    int *displs = new int[numprocs]();
    int *recv_counts = new int[numprocs]();
    for(int i=0; i<numprocs; i++)
    {
      displs[i]=i*div_size;
      recv_counts[i]=1;
    }
    real dx, y_new;
    double global_sum, local_sum;
    real *y = new real[div_size]();
    for(int i = rank * div_size; i < (rank + 1) * div_size; i++)
    {
      y[i-rank*div_size] = x[i];
    }
    for(k = 0; k < maxit; k++)
    {
      local_sum=0.0;
      for(int i = rank * div_size; i < (rank + 1) * div_size; i++)
      {
        if(sub_method==1)
        {
          y_new = b[i];
          for(int j=0; j<ny; j++)
          {
            if(i != j)
            {
              y_new -= A(i,j) * x[j];
            }
          }
          y_new /= A(i,i);
          dx = y[i-rank*div_size]-y_new;
          y[i-rank*div_size] = y_new;
          MPI_Allgather(y,div_size,MPI_DOUBLE,x,div_size,MPI_DOUBLE,MPI_COMM_WORLD);
        }
        else if(sub_method==2)
        {
          y_new = b[i];
          for(int j=0; j<ny; j++)
          {
            if(i != j)
            {
              y_new -= A(i,j) * x[j];
            }
          }
          y_new /= A(i,i);
          real w = 0.85;//w==1 normal
          y_new = w  * y_new + (1.0 - w) * x[i];
          dx = x[i]-y_new;
          //y[i-rank*div_size] = y_new;
          MPI_Allgatherv(&y_new,1,MPI_DOUBLE,&x[i-rank*div_size],recv_counts,displs,MPI_DOUBLE,MPI_COMM_WORLD);
        }
        local_sum += ((dx >= 0.0) ? dx : -dx);
      }
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if(global_sum <= epsilon)
      {
        break;
      }
    }
    free(y);
    free(displs);
    free(recv_counts);

    if(rank==0)
    {
      std::cout << k << " " << global_sum << " " << div_size << std::endl;
    }
  }

  if(rank==0)
  {
    std::cout << "Elapsed time:" <<  MPI_Wtime() - t1 << std::endl;

    for(int i=0; i<std::min(10,nx); i++)
    {
      std::cout << "x[" << i << "]= "  << x[i] << std::endl;
    }
  }

  delete [] A;
  delete [] x;
  delete [] b;

  MPI_Finalize();
}
