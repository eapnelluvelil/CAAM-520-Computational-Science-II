#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
// #include <math.h>   // For ceil, log2, and pow

/*
  Function: my_bcast
  Purpose:  Broadcasts information from master rank (rank 0) to all 
            other ranks
  Inputs:
    buf:  Void pointer; on master rank, this is the send buffer, and all
          other ranks, is the receive buffer
    count:  Integer specifying how many units of information to send and receive
    dataype:  MPI data type
    comm:   MPI communicator
  Outputs:
    None
*/
void my_bcast(void *buf, int count, MPI_Datatype datatype, MPI_Comm comm)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  // Get current rank and number of processes
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int tag = 999;

  // Check if we're on the master rank (rank 0)
  if (rank == 0) {
    // Send data to all other ranks besides the master rank
    for (int i = 0; i < size; i++) {
      if (i != rank) {
        MPI_Send(buf, count, 
                 datatype, i, 
                 tag, comm);
      }
    }
  }
  else {
    // If current rank is not master rank, receive data from master rank
    MPI_Recv(buf, count, 
             datatype, 0, 
             tag, comm, 
             MPI_STATUS_IGNORE);
  }
}

/*
  Function: my_bcast_tree
  Purpose:  Performs tree style broadcast starting from rank 0
  Inputs:
    buf:  Void pointer; on master rank, this is the send buffer, and all
          other ranks, is the receive buffer
    count:  Integer specifying how many units of information to send and receive
    dataype:  MPI data type
    comm:   MPI communicator
  Outputs:
    None
*/

void my_bcast_tree(void *buf, int count, MPI_Datatype datatype, MPI_Comm comm)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  // Get current rank and number of processes
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int tag = 999;

  int ranks_reached = 1;

  while (ranks_reached < size) {
    // If the current ranks has received information, send to the next available
    // rank (computed via the offset)
    if (rank < ranks_reached) {
      int send_to_rank = rank + ranks_reached;
      // Make sure we are sending to a valid rank
      if (send_to_rank < size) {
        MPI_Send(buf, count, 
                 datatype, send_to_rank, 
                 tag, comm);
      }
    }
    // If the current rank is able to receive information, then do so
    else if (ranks_reached <= rank && rank < (ranks_reached * 2)) {
      int recv_from_rank = rank - ranks_reached;
      MPI_Recv(buf, count,
               datatype, recv_from_rank, 
               tag, comm,
               MPI_STATUS_IGNORE);
    }
    // Update the offset
    ranks_reached *= 2;
  }

  // // Figure out how many stages we need, i.e., the depth of the binary tree
  // int num_stages = (int) ceil(log2(size));

  // // We send to and receive from ranks in stages
  // // For example, if we have 8 processes, 
  // for (int current_stage = 0; current_stage < num_stages; current_stage++) {
  //   int lower_stage = (int) pow(2, current_stage);
  //   int upper_stage = (int) pow(2, current_stage + 1);

  //   // If rank < lower_stage, then we send to rank + lower_stage
  //   if (rank < lower_stage) {
  //     int send_to_rank = rank + lower_stage;
  //     // If the number of processes is not a power of two,
  //     // we check that we are sending to a valid rank
  //     if (send_to_rank < size) {
  //       MPI_Send(buf, count, 
  //                datatype, send_to_rank, 
  //                tag, comm);
  //     }
  //   }
  //   // If lower_stage <= rank < upper_stage, then we receive from
  //   // rank - lower_stage
  //   else if (lower_stage <= rank && rank < upper_stage) {
  //     int recv_from_rank = rank - lower_stage;
  //     MPI_Recv(buf, count, 
  //              datatype, recv_from_rank, 
  //              tag, comm, 
  //              MPI_STATUS_IGNORE);
  //   }
  // }

}

// Do not modify the main function!
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL));

  int seed, result, reference;
  if (rank == 0) seed = rand()%100;
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  seed += rank;
  MPI_Reduce(&seed, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  reference = 0;
  for (int r = 0; r < size; r++) {
    reference += seed + r;
  }
  if (rank == 0) {
    if (result == reference) {
      printf("MPI_Bcast() works correctly!\n");
    }
    else {
      printf("MPI_Bcast() failed!\n");
    }
  }

  if (rank == 0) seed = rand()%100;
  my_bcast(&seed, 1, MPI_INT, MPI_COMM_WORLD);
  seed += rank;
  MPI_Reduce(&seed, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  reference = 0;
  for (int r = 0; r < size; r++) {
    reference += seed + r;
  }
  if (rank == 0) {
    if (result == reference) {
      printf("my_bcast() works correctly!\n");
    }
    else {
      printf("my_bcast() failed!\n");
    }
  }

  if (rank == 0) seed = rand()%100;
  my_bcast_tree(&seed, 1, MPI_INT, MPI_COMM_WORLD);
  seed += rank;
  MPI_Reduce(&seed, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  reference = 0;
  for (int r = 0; r < size; r++) {
    reference += seed + r;
  }
  if (rank == 0) {
    if (result == reference) {
      printf("my_bcast_tree() works correctly!\n");
    }
    else {
      printf("my_bcast_tree() failed!\n");
    }
  }

  MPI_Finalize();
  return 0;
}
