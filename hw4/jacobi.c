#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

PetscInt solve_jacobi(Mat A, Vec b, Vec x,
                      PetscReal tol, PetscReal omega, PetscInt max_iter)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  // Vectors to 
  // 1. Store copy of x, 
  // 2. diagonal entries of A, 
  // 3. old residual, and 
  // 4. new residual, respectively
  Vec x_new, d, r_old, r_new;

  VecDuplicate(x, &x_new);
  VecCopy(x, x_new);

  VecDuplicate(b, &d);
  VecDuplicate(b, &r_old);
  VecDuplicate(b, &r_new);

  // Obtain diagonal entries of A
  MatGetDiagonal(A, d);

  // Compute inverse of diagonal entries
  PetscInt local_size;
  VecGetLocalSize(d, &local_size);

  PetscScalar *d_local;
  VecGetArray(d, &d_local);

  for (PetscInt i_local = 0; i_local < local_size; i_local++) {
    if (d_local[i_local] != 0) {
      d_local[i_local] = 1.0/d_local[i_local];
    }
    else {
      d_local[i_local] = 0.0;
    }
  }

  VecRestoreArray(d, &d_local);

  // Assemble inverse of diagonal of A
  Mat D_inv;
  MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &D_inv);
  MatDiagonalSet(D_inv, d, INSERT_VALUES);

  // Compute norm of initial residual
  PetscReal norm_r0;
  MatMult(A, x, r_old);
  VecAXPBY(r_old, 1.0, -1.0, b);
  VecNorm(r_old, NORM_2, &norm_r0);

  // Number of iterations performed
  PetscInt jacobi_iter = 0;

  while (1) {
    PetscReal norm_rk;
    MatMult(A, x_new, r_old);
    VecAXPBY(r_old, 1.0, -1.0, b);
    VecNorm(r_old, NORM_2, &norm_rk);

    if ((norm_rk/norm_r0) < tol || jacobi_iter > max_iter) {
      break;
    }

    // Compute D^{-1} * rk, and compute new iteration
    MatMult(D_inv, r_old, r_new);
    VecAXPY(x_new, omega, r_new);

    // Assign new residual to old residual
    VecCopy(r_new, r_old);

    jacobi_iter++;
  }

  // // Clean up
  MatDestroy(&D_inv);
  VecDestroy(&x_new);
  VecDestroy(&d);
  VecDestroy(&r_old);
  VecDestroy(&r_new);

  // Return the number of iterations
  return jacobi_iter;
}

PetscInt solve_cg(Mat A, Vec b, Vec x, PetscReal tol, PetscInt max_iter)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  // Create solver
  KSP solver;
  KSPCreate(PETSC_COMM_WORLD, &solver);
  KSPSetType(solver, KSPCG);
  KSPSetOperators(solver, A, A);

  // The solver should not stop on an absolute tolerance or divergence tolerance
  PetscReal abstol = 0.0;
  PetscReal dtol   = 1.0e12;
  KSPSetTolerances(solver, tol, abstol, dtol, max_iter);

  // Call solver
  KSPSolve(solver, b, x);

  // Get number of iterations for solver to fall below relative tolerance
  PetscInt cg_iter;
  KSPGetIterationNumber(solver, &cg_iter);

  // Clean up
  KSPDestroy(&solver);

  return cg_iter;
}

// Do not change the main function!
int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  // Get command line options.
  char mat_file[1024];
  PetscBool mat_file_set;
  PetscOptionsGetString(NULL,
                        NULL,
                        "-A",
                        mat_file,
                        sizeof(mat_file),
                        &mat_file_set);
  if (!mat_file_set) {
    PetscPrintf(PETSC_COMM_WORLD, "Error: No matrix file specified!\n");
    PetscFinalize();
    return -1;
  }
  PetscInt max_iter = 50000;
  PetscOptionsGetInt(NULL, NULL, "-max_iter", &max_iter, NULL);
  PetscReal tol = 1.0e-3;
  PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL);
  PetscReal omega = 1.0;
  PetscOptionsGetReal(NULL, NULL, "-omega", &omega, NULL);
  PetscBool print_difference = PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-compare", &print_difference, NULL);

  // Load matrix.
  Mat A;
  MatCreate(PETSC_COMM_WORLD, &A);
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_file, FILE_MODE_READ, &viewer);
  MatLoad(A, viewer);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  Vec b, x, e;
  MatCreateVecs(A, &x, &b);
  VecDuplicate(x, &e);

  // Test Jacobi solver.
  VecSet(x, 1.0);
  VecSet(b, 1.0);
  const int iter_jacobi = solve_jacobi(A, b, x, tol, omega, max_iter);
  if (iter_jacobi < max_iter) {
    PetscPrintf(PETSC_COMM_WORLD,
                "Number of iterations with Jacobi: %d\n",
                iter_jacobi);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,
                "Jacobi solver did not converge in %d iterations!\n",
                max_iter);
  }

  VecCopy(x, e);

  // Test CG solver.
  VecSet(x, 1.0);
  VecSet(b, 1.0);
  const int iter_cg = solve_cg(A, b, x, tol, max_iter);
  if (iter_cg < max_iter) {
    PetscPrintf(PETSC_COMM_WORLD,
                "Number of iterations with CG: %d\n",
                iter_cg);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,
                "CG solver did not converge in %d iterations!\n",
                max_iter);
  }

  VecAXPY(e, -1.0, x);
  PetscReal norm_e;
  VecNorm(e, NORM_INFINITY, &norm_e);
  if (print_difference) {
    PetscPrintf(PETSC_COMM_WORLD, "||e|| = %e\n", norm_e);
  }

  // Clean up.
  MatDestroy(&A);
  VecDestroy(&b);
  VecDestroy(&x);
  VecDestroy(&e);
  PetscFinalize();
  return 0;
}
