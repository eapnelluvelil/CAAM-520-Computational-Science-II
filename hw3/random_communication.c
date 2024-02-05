#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

void exchange_standard(const int *values,
                       const int *indices,
                       int *result,
                       int num_values,
                       int comm_size)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Fetch the appropriate random number from the current rank's 
  // value array
  result[rank] = values[indices[rank]];

  for (int r = 0; r < comm_size; ++r) { 
    int send_idx;

    // Send and receive information from every other rank
    if (r != rank) {
      // Current rank sends the rth rank an index
      // Current rank also receives the corresponding index from the rth rank
      // and stores it in send_idx
      MPI_Sendrecv(&indices[r], 1, MPI_INT, r, 999,
                   &send_idx, 1, MPI_INT, r, 999, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Current rank sends values[send_idx] to the rth rank
      // Current rank also receives the corresponding value from the rth rank
      MPI_Sendrecv(&values[send_idx], 1, MPI_INT, r, 999,
                   &result[r], 1, MPI_INT, r, 999, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void exchange_onesided(const int *values,
                       const int *indices,
                       int *result,
                       int num_values,
                       int comm_size)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Fetch the appropriate random number from the current rank's 
  // value array
  result[rank] = values[indices[rank]];

  // Create window for one-sided access to values on each rank
  MPI_Win win;
  MPI_Win_create(values, num_values * sizeof(int),
                 sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  // Once the one-sided MPI communication below is finished,
  // result[r] will contain values[indices[r]], where values
  // resides on the rth rank

  // This fence is necessary as it starts the epoch
  MPI_Win_fence(0, win);

  for (int r = 0; r < comm_size; ++r) {
    if (r != rank) {
      MPI_Get(&result[r], 1, MPI_INT,
              r, indices[r], 1,
              MPI_INT, win);
    }
  }

  // This fence is necessary as it ends the epoch.
  MPI_Win_fence(0, win);

  MPI_Win_free(&win);
}

// Do not modify the main function!
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
 
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
 
  // Initialize random number generator.
  srand(rank*time(NULL));
 
  // Allocate memory.
  const int num_values = 100;
  int *values = malloc(num_values*sizeof(int));
  int *indices = malloc(comm_size*sizeof(int));
  int *result = malloc(comm_size*sizeof(int));
  int *reference = malloc(comm_size*sizeof(int));
  int *buffer = malloc(num_values*sizeof(int));
  if (!values || !indices || !result || !reference || !buffer) {
    fprintf(stderr, "Failed allocation!\n");
    goto abort;
  }
 
  // Initialize array of random numbers.
  for (int i = 0; i < num_values; i++) {
    values[i] = rand();
  }
  // Pick a random index for each rank.
  for (int r = 0; r < comm_size; r++) {
    indices[r] = rand()%num_values;
  }
 
  // Create reference solution.
  for (int r = 0; r < comm_size; r++) {
    if (r == rank) {
      memcpy(buffer, values, num_values*sizeof(int));
    }
    MPI_Bcast(buffer, num_values, MPI_INT, r, MPI_COMM_WORLD);
    reference[r] = buffer[indices[r]];
  }
 
  // Test implementation without one-sided communication.
  exchange_standard(values, indices, result, num_values, comm_size);
  for (int r = 0; r < comm_size; r++) {
    if (result[r] != reference[r]) {
      fprintf(stderr, "exchange_standard() failed on rank %d!\n", rank);
      goto abort;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("exchange_standard() is working correctly!\n");
  }
 
  // Reset the result.
  memset(result, 0, comm_size*sizeof(int));
 
  // Test implementation with one-sided communication.
  exchange_onesided(values, indices, result, num_values, comm_size);
  for (int r = 0; r < comm_size; r++) {
    if (result[r] != reference[r]) {
      fprintf(stderr, "exchange_onesided() failed on rank %d!\n", rank);
      goto abort;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("exchange_onesided() is working correctly!\n");
  }
 
  free(values);
  free(indices);
  free(result);
  free(reference);
  free(buffer);
  MPI_Finalize();
  return 0;
 
abort:
  free(values);
  free(indices);
  free(result);
  free(reference);
  free(buffer);
  MPI_Finalize();
  return -1;
}
