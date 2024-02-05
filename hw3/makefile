sendrecv: sendrecv.c
	mpicc -g -std=c99 -o sendrecv sendrecv.c

random_communication: random_communication.c
	mpicc -g -std=c99 -o random_communication random_communication.c -lm

clean:
	rm -f sendrecv random_communication

.PHONY: clean
