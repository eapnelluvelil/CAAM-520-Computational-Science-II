#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "csr_matrix.h"

// Computes the sparse matrix-vector product A*x, where
// A is a matrix stored in the CSR matrix format and 
// x is a vector of length n that is to be multiplied by A
// The result is written to the result vector
void sparse_mat_mult(const csr_matrix_t *A, // Sparse matrix
                     double *x,             // Vector to be multiplied by A
                     double *result,        // Vector where we will write A*x to
                     int n)                 // System size
{
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    int row_start = (A->row_ptr)[i];
    int row_end = (A->row_ptr)[i + 1];

    for (int k = row_start; k < row_end; k++) {
      result[i] += (A->val)[k] * x[(A->col_ind)[k]];
    }
  }
}

// Computes the diagonal matrix-vector product diag*x, where
// diag is the vector containing the main diagonal of a 
// diagonal matrix (that is n x n) and x is a vector of
// length n
// The result is written to the result vector
void diag_mat_mul(double *diag,           // Diagonal matrix stored as a vector
                  double *x,              // Vector to be multiplied by diag              
                  double *result,         // Vector where we will write diag*x to
                  int n)                  // System size
{
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    result[i] = diag[i] * x[i];
  }
}

// Computes vector alpha*x + beta*y, where alpha and beta are scalars
// and x and y are vectors of length n
// The result is written to the result vector
void vec_add(double alpha,                // Scalar that pre-multiplies the vector x
             double *x,                   // Vector
             double beta,                 // Scalar that pre-multiplies the vector y
             double *y,                   // Vector
             double *result,              // Vector where we will alpha*x + beta*y to
             int n)                       // System size
{
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    result[i] = (alpha * x[i]) + (beta * y[i]);
  }
}

// Computes the two-norm of a vector x of length n
double norm2(double *x,                   // Vector
             int n)                       // System size
{
  double two_norm = 0.0;

  #pragma omp parallel for reduction(+:two_norm)
  for (int i = 0; i < n; i++) {
    two_norm += (x[i] * x[i]);
  }

  two_norm = sqrt(two_norm);
  return two_norm;
}

// Use the Jacobi method to solve the linear system Ax = b to the desired
// tolerance, i.e., such that ||b - Ax||/||b|| < tol.
void jacobi(const csr_matrix_t *A,  // Sparse matrix
            double *x,              // Solution vector and initial guess
            const double *b,        // Right-hand side
            int n,                  // System size
            double omega,           // Relaxation coefficient
            double tol,             // Relative tolerance
            double *residual,       // Achieved relative residual
            int *iter)              // Number of iterations until convergence
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //              (Feel free to add more functions as needed.)                //
  //////////////////////////////////////////////////////////////////////////////

  // Compute inverse of diagonal part of A
  double *inv_D = (double*) calloc(n, sizeof(double));

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    int row_start = (A->row_ptr)[i];
    int row_end   = (A->row_ptr)[i + 1];

    for (int k = row_start; k < row_end; k++) {
      if ((A->col_ind)[k] == i) {
        inv_D[i] = 1.0/((A->val)[k]);
        break;
      }
    }
  }

  // Compute initial relative residual b - Ax
  // Start by computing Ax
  double *bmAx = (double*) calloc(n, sizeof(double));
  sparse_mat_mult(A, x, bmAx, n);

  // Now, compute b - Ax
  vec_add(1.0, b, -1.0, bmAx, bmAx, n);

  // Compute relative residual ||b - Ax|| / ||b||
  double b_norm = norm2(b, n);
  *residual = norm2(bmAx, n)/b_norm;

  // Initialize iter to be 0
  *iter = 0;

  // Perform Jacobi while relative residual is greater than tolerance
  while ((*residual) > tol) {    
    // Compute D^(-1)*(b - Ax)
    diag_mat_mul(inv_D, bmAx, bmAx, n);

    // Compute new iterate x = x + omega*D^(-1)*(b - Ax)
    vec_add(1.0, x, omega, bmAx, x, n);

    // Compute new residual vector
    sparse_mat_mult(A, x, bmAx, n);
    vec_add(1.0, b, -1.0, bmAx, bmAx, n);

    // Update relative residual and iteration counter
    *residual = norm2(bmAx, n)/b_norm;
    (*iter) += 1;
  }

  // Free allocated memory used to store D^(-1) and the residual (b-Ax)
  free(inv_D);
  free(bmAx);
}

// Do not modify the main function!
int main(int argc, char **argv)
{
  if (argc != 4) {
    fprintf(stderr, "Usage: ./jacobi matrix_file tol omega\n");
    return -1;
  }

  // Read tolerance from command line.
  const double tol = atof(argv[2]);

  // Read relaxation coefficient from command line.
  const double omega = atof(argv[3]);

  // Read matrix from matrix market file.
  csr_matrix_t A;
  const int n = csr_matrix_load(&A, argv[1]);
  if (n < 1) {
    fprintf(stderr, "Error: Could not load matrix market file %s!\n", argv[1]);
    return -1;
  }

  // Create solution vector and right-hand side.
  double *x = (double*) malloc(n*sizeof(double));
  double *b = (double*) malloc(n*sizeof(double));
  if (!x || !b) {
    fprintf(stderr, "Error: Could not create vectors x and b!\n");
    free(x);
    csr_matrix_free(&A);
    return -1;
  }
  for (int i = 0; i < n; i++) {
    x[i] = 0.0;
    b[i] = 1.0;
  }

  // Call Jacobi method.
  int iter;
  double residual;
  jacobi(&A, x, b, n, omega, tol, &residual, &iter);

  // Report results.
  printf("Jacobi method converged in %d iter. to a rel. residual of %e.\n",
         iter,
         residual);
  printf("(Matrix: %s, rel. tol.: %e)\n", argv[1], tol);

  free(x);
  free(b);
  csr_matrix_free(&A);
  return 0;
}
