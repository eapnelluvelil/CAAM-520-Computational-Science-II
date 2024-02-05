#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

PetscInt inverse_iteration(Mat A, Vec v, PetscScalar mu,
                           PetscReal tol, PetscInt max_iter,
                           PetscScalar *lambda)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////

  // Vectors to store old and new solution iterates, respectively
  Vec v_old, v_new;

  VecDuplicate(v, &v_old);
  VecCopy(v, v_old);

  VecDuplicate(v, &v_new);

  // Vector to store the diagonal entries of the scaled identity matrix
  Vec d;
  VecDuplicate(v, &d);
  VecSet(d, -mu);

  // Create (A - mu*I)
  Mat A_mu_I;
  MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &A_mu_I);
  MatDiagonalSet(A_mu_I, d, INSERT_VALUES);
  MatAXPY(A_mu_I, 1.0, A, SAME_NONZERO_PATTERN);

  // Create solver to solve (A - mu*I) v_new = v_old
  KSP solver;
  KSPCreate(PETSC_COMM_WORLD, &solver);
  KSPSetType(solver, KSPGMRES);
  KSPSetOperators(solver, A_mu_I, A_mu_I);

  // Set solver tolerances
  PetscReal rtol   = 1.0e-9;
  PetscReal abstol = 0.0;
  PetscReal dtol   = 1.0e12;
  PetscInt  maxits = 10000;
  KSPSetTolerances(solver, rtol, abstol, dtol, maxits);

  // Compute initial lambda
  Vec Av;
  VecDuplicate(v, &Av);
  MatMult(A, v_old, Av);
  VecDot(v_old, Av, lambda);

  // Compute initial residual
  VecAXPY(Av, -(*lambda), v_old);
  PetscReal norm_r;
  VecNorm(Av, NORM_2, &norm_r);

  // Number of times PETSc's GMRES solver is called
  PetscInt gmres_iter = 0;

  while (1) {
    if (norm_r < tol || gmres_iter > max_iter) {
      break;
    }

    // Call solver to compute new solution iterate, and store solution in v_new
    KSPSolve(solver, v_old, v_new);
    KSPGetSolution(solver, &v_new);

    // Normalize new iterate
    PetscReal norm_v_tilde;
    VecNorm(v_new, NORM_2, &norm_v_tilde);
    VecScale(v_new, 1.0/norm_v_tilde);

    // Compute new lambda
    MatMult(A, v_new, Av);
    VecDot(v_new, Av, lambda);

    // Compute new residual
    VecAXPY(Av, -(*lambda), v_new);
    VecNorm(Av, NORM_2, &norm_r);

    // Assign new iterate to old iterate
    VecCopy(v_new, v_old);

    gmres_iter++;
  }

  // Clean up
  VecDestroy(&Av);
  KSPDestroy(&solver);
  MatDestroy(&A_mu_I);
  VecDestroy(&d);
  VecDestroy(&v_new);
  VecDestroy(&v_old);

  // Return the number of GMRES calls we made
  return gmres_iter;
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
  PetscInt max_iter = 1000;
  PetscOptionsGetInt(NULL, NULL, "-max_iter", &max_iter, NULL);
  PetscReal tol = 1.0e-3;
  PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL);
  PetscScalar mu = 0.0;
  PetscOptionsGetScalar(NULL, NULL, "-mu", &mu, NULL);

  // Load matrix.
  Mat A;
  MatCreate(PETSC_COMM_WORLD, &A);
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_file, FILE_MODE_READ, &viewer);
  MatLoad(A, viewer);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  // Set initial vector.
  Vec v;
  MatCreateVecs(A, &v, NULL);
  VecSet(v, 1.0);
  PetscReal norm_v;
  VecNorm(v, NORM_2, &norm_v);
  VecScale(v, 1.0/norm_v);

  // Perform inverse iteration.
  PetscScalar lambda;
  const int iter = inverse_iteration(A, v, mu, tol, max_iter, &lambda);

  // Print results.
  if (iter < max_iter) {
    PetscPrintf(PETSC_COMM_WORLD, "lambda = %e (%d iterations)\n", lambda, iter);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,
                "Inverse iteration did not converge in %d iterations %e!\n",
                max_iter, lambda);
  }

  // Clean up.
  MatDestroy(&A);
  VecDestroy(&v);
  PetscFinalize();
  return 0;
}
