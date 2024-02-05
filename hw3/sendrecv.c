#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

void my_sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  MPI_Request send_request

  // Send sendbuf using a non-blocking send
  MPI_Isend(sendbuf, sendcount, sendtype, dest, sendtag, comm, &send_request);

  // Receive into recvbuf using a blocking receive
  MPI_Irecv(recvbuf, recvcount, recvtype, source, recvtag, comm, &recv_request);

  // Wait for the non-blocking send to finish
  MPI_Wait(&send_request, status);
}

// Do not modify the main function!
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  // We will test the implementation for messages of up to 8 MiB.
  const int max_size = 8*1024*1024;
  char *sendbuf = malloc(max_size);
  char *recvbuf = malloc(max_size);
  if (!sendbuf || !recvbuf) {
    fprintf(stderr, "Allocation failed!\n");
    // A rare case where the use of goto is justified
    goto abort;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0) {
      fprintf(stderr, "This program must be run using exactly two ranks!\n");
    }
    goto abort;
  }

  const int other = 1 - rank;

  // Initialize random number generator.
  srand(rank*time(NULL));

  for (int msg_size = 1; msg_size <= max_size; msg_size *= 2) {
    if (rank == 0) {
      printf("Running test for messages of size %d...\n", msg_size);
    }

    // Create a random message.
    for (int i = 0; i < msg_size; i++) {
      sendbuf[i] = (char) rand()%128;
    }

    // Exchange data and compute reference result.
    MPI_Sendrecv(sendbuf, msg_size, MPI_CHAR,
                 other, 999,
                 recvbuf, msg_size, MPI_CHAR,
                 other, 999,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int reference = 0;
    for (int i = 0; i < msg_size; i++) {
      reference += recvbuf[i];
    }

    // Reset receive buffer.
    memset(recvbuf, 0, msg_size);

    // Exchange data again using the above implementation.
    MPI_Status status;
    my_sendrecv(sendbuf, msg_size, MPI_CHAR,
                other, 999,
                recvbuf, msg_size, MPI_CHAR,
                other, 999,
                MPI_COMM_WORLD, &status);

    // Check the result.
    int result = 0;
    for (int i = 0; i < msg_size; i++) {
      result += recvbuf[i];
    }

    if (result != reference) {
      fprintf(stderr,
              "Mismatch on rank %d for messages of size %d!\n",
              rank, msg_size);
      goto abort;
    }

    // Check the status.
    int count;
    MPI_Get_count(&status, MPI_CHAR, &count);
    if (count != msg_size || status.MPI_SOURCE != other) {
      fprintf(stderr,
              "Incorrect status on rank %d for messages of size %d\n",
              rank, msg_size);
      goto abort;
    }

    if (rank == 0) {
      printf("ok\n");
    }
  }

  free(sendbuf);
  free(recvbuf);
  MPI_Finalize();
  return 0;

abort:
  free(sendbuf);
  free(recvbuf);
  MPI_Finalize();
  return -1;
}
