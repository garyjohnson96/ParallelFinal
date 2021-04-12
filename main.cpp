#include <mpi.h>
#include <chrono>
#include <vector>
#include <iomanip>
#include <algorithm>

using namespace std;

#define MCW MPI_COMM_WORLD
#define MASTER 0

//Tags
#define WORK 1
#define TERMINATE 2

const char * getColor(float temp){
    if (temp < 25) { return "\033[1;34m"; }
    if (temp >= 25 && temp <= 60) { return "\033[1;33m"; }
    if (temp > 60) { return "\033[1;31m"; }
    return "\033[0m";
}

// DEBUG helper function to check the values in the mesh
void printMesh(vector<vector<float>> mesh, int rank, float timestep) {
    cout << string(2,'\n');
    cout << "Rank: " << rank << " Time: "<<timestep<<endl;
    for (size_t row = 0; row < mesh.size(); row++) {
        for (size_t col = 0; col < mesh[row].size(); col++) {
            const char* color = getColor(mesh[row][col]);
            cout << std::fixed << std::setprecision(4) << color << mesh[row][col] << " " << "\033[0m";
        }
        cout << endl;
    }
}

// Sets the hot and cool constant temps to random cells
void setTemps(vector<vector<float>>& mesh, vector<vector<int>> hotCells, vector<vector<int>> coolCells, int rank, float hotTemp, float coolTemp) {

    // set the hot and cool cell coordinates
    for (int cell = 0; cell < hotCells.size(); cell++) {

        MPI_Bcast(hotCells[cell].data(), hotCells[cell].size(), MPI_INT, MASTER, MCW);
        MPI_Bcast(coolCells[cell].data(), coolCells[cell].size(), MPI_INT, MASTER, MCW);

        // Set constant temperature cells accordingly
        mesh[hotCells[cell][0]][hotCells[cell][1]] = hotTemp;
        mesh[coolCells[cell][0]][coolCells[cell][1]] = coolTemp;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double start, end;
    int rank, size;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    // take in the command line variables
    int mesh_size, iterations;
    sscanf(argv[1], "%d", &mesh_size);
    sscanf(argv[5], "%d", &iterations);
    float dt, dx, therm_diff;
    sscanf(argv[2],"%f", &dt);
    sscanf(argv[3],"%f", &dx);
    sscanf(argv[4],"%f", &therm_diff);
//    float dt = 1.5, dx = 0.038, therm_diff = 0.000111; // dt-time step, dx-node separation, thermal diffusivity is for copper
    float fourier = (dt * therm_diff) / (dx * dx); // FOURIER MUST BE < 0.25 FOR STABILITY
    if (fourier > 0.25) {
        if (rank==MASTER) { cout<<"Fourier of "<<fourier<<" is larger than 0.25 reduce timestep for stability, exiting..."<<endl; }
        MPI_Finalize();
        return 0;
    }
    float hot = 75, cold = 20;

    srand(chrono::system_clock::now().time_since_epoch().count());  // seed rand() by current time

    vector<vector<float>> mesh(mesh_size, vector<float>(mesh_size, 25));  // initialize the mesh with the base temperature

    vector<vector<int>> hotCells{ {3, 0} , {4, 2} };  // mesh coordinates of the hot and cool cells
    vector<vector<int>> coolCells{ {1, 3} , {4, 4} };

    setTemps(mesh, hotCells, coolCells, rank, hot, cold);
    MPI_Barrier(MCW);

    start = MPI_Wtime();

    int timeStep = 0;
    while (timeStep < iterations) {

        if (rank == MASTER) {
            printMesh(mesh, rank, timeStep*dt);
            int currentRow = 0;
            int workingProcesses = 0;

            // Begin sending new work to each worker
            for (int worker = 1; worker < size; worker++) {
                // sends index
                MPI_Send(&currentRow, 1, MPI_INT, worker, WORK, MCW);
                currentRow++;
                workingProcesses++;
            }

            int rows = mesh.size();
            MPI_Status workerStatus;

            // Send rows until all rows have been completed
            while (workingProcesses > 0) {
                vector<float> receivedRow(mesh[0].size());

                // Receive row of work from worker. The row index is received as the tag
                MPI_Recv(receivedRow.data(), receivedRow.size(), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MCW, &workerStatus);
                workingProcesses--;

                mesh[workerStatus.MPI_TAG] = receivedRow;
                int readyWorker = workerStatus.MPI_SOURCE;  // Send next row to worker that just finished last one

                if (currentRow < rows) {
                    // sends index
                    MPI_Send(&currentRow, 1, MPI_INT, readyWorker, WORK, MCW);
                    currentRow++;
                    workingProcesses++;
                } else {
                    MPI_Send(NULL, 0, MPI_INT, readyWorker, TERMINATE, MCW);
                }
            }
        } else {  // workers
            int row;
            bool working = true;
            MPI_Status workStatus;
            while (working) {
                // receives index.
                MPI_Recv(&row, 1, MPI_INT, MASTER, MPI_ANY_TAG, MCW, &workStatus);
                if (workStatus.MPI_TAG == TERMINATE) {
                    working = false;
                } else {
                    vector<float> updatedRow(mesh[0].size());
                    float a, b, c, d;

                    for (int i=0; i<mesh[0].size(); i++) {
                        vector<int> key = {row, i};
                        if ((std::find(hotCells.begin(), hotCells.end(), key) != hotCells.end()) ||
                             (std::find(coolCells.begin(), coolCells.end(), key) != coolCells.end())) { // check to see if a constant cell
                            updatedRow[i] = mesh[row][i];
                        } else {
	                        if (row == 0) { // boundary conditions at edges is insulated meaning we use the inner point twice
	                            a = mesh[row+1][i];
	                        } else {
	                            a = mesh[row-1][i]; }
	                        if (row == mesh[0].size()-1) {
	                            b = mesh[row-1][i];
	                        } else {
	                            b = mesh[row+1][i]; }
	                        if (i == 0) {
	                            c = mesh[row][i+1];
	                        } else {
	                            c = mesh[row][i-1]; }
	                        if (i == mesh[0].size()-1) {
	                            d = mesh[row][i-1];
	                        } else {
	                            d = mesh[row][i+1]; }
	                        updatedRow[i] = fourier * (a + b + c + d) + (1 - 4 * fourier) * mesh[row][i];
                        }
                    }

                    // Send updated row back to master with the row index as the tag
                    MPI_Send(updatedRow.data(), updatedRow.size(), MPI_FLOAT, MASTER, row, MCW);
                }
            }
        }

        timeStep++;

        // Update each process's mesh
        for (int row = 0; row < mesh.size(); row++) {
            MPI_Bcast(mesh[row].data(), mesh[row].size(), MPI_FLOAT, MASTER, MCW);
            MPI_Barrier(MCW);
        }
    }

    MPI_Finalize();

    end = MPI_Wtime();
    if (rank==MASTER) { printMesh(mesh,rank, iterations*dt); cout<<"Runtime = "<<end-start<<endl; }

    return 0;
}
