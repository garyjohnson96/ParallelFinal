#include <mpi.h>

#define MCW MPI_COMM_WORLD

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);

    MPI_Finalize();
    return 0;
}