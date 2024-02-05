bcast: bcast.c
	mpicc -g -std=c99 -o bcast bcast.c

jacobi_3d_slicing: jacobi_3d_slicing.c
	mpicc -g -std=c99 -o jacobi_3d_slicing jacobi_3d_slicing.c -lm

clean:
	rm -f jacobi_3d_slicing

.PHONY: clean
