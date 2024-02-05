# Path to the PETSc installation
PETSC_DIR=~/Software/petsc-3.12.2
# PETSC_DIR=~/petsc-3.12.5

# Fetch information about compilers etc. from PETSc.
include ${PETSC_DIR}/lib/petsc/conf/variables

# A rule to compile eac .c file into the corresponding .o file.
%.o: %.c
	${PETSC_COMPILE} -std=c99 -o $@ $<

jacobi: jacobi.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

inverse_iteration: inverse_iteration.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

clean:
	rm -rf *.o jacobi inverse_iteration

.PHONY: clean
