#include <mpi.h>
#include <chrono>
#include <vector>

using namespace std;

#define MCW MPI_COMM_WORLD
#define MASTER 0

//Tags
#define WORK 1
#define NEW_DATA 2
#define TERMINATE 3

// globals for different size meshes
const int TINY = 10;
const int SMALL = 100;
const int MEDIUM = 1000;
const int LARGE = 10000;
const int VERY_LARGE = 100000;

// DEBUG helper function to check the values in the mesh
void printMesh(vector<vector<int>> mesh, int rank) {  
    cout << "Rank: " << rank << endl;
    for (size_t row = 0; row < mesh.size(); row++) {
        for (size_t col = 0; col < mesh[row].size(); col++) { 
            cout << mesh[row][col] << " ";
        }
        cout << endl;
    }
}

// Sets the hot and cool constant temps to random cells
void setTemps(vector<vector<int>>& mesh, int rank, int hotTemp, int coolTemp) {
    int constCells = mesh.size() * 0.2;   // 20% of the mesh cells are constant temperature points

    vector<vector<int>> hotCells(constCells, vector<int>(2));  // mesh coordinates of the hot and cool cells
    vector<vector<int>> coolCells(constCells, vector<int>(2));

    // set the hot and cool cell coordinates
    for (int cell = 0; cell < constCells; cell++) {
        if (rank == MASTER) {
            hotCells[cell][0] = rand() % mesh.size();
            hotCells[cell][1] = rand() % mesh.size();

            do {  // Ensures the hot and cool cells aren't the same
                coolCells[cell][0] = rand() % mesh.size();
                coolCells[cell][1] = rand() % mesh.size();
            } while (coolCells[cell][0] == hotCells[cell][0] && coolCells[cell][1] == hotCells[cell][1]);
        }

        MPI_Bcast(hotCells[cell].data(), hotCells[cell].size(), MPI_INT, MASTER, MCW);
        MPI_Bcast(coolCells[cell].data(), coolCells[cell].size(), MPI_INT, MASTER, MCW);
        
        // Set constant temperature cells accordingly
        mesh[hotCells[cell][0]][hotCells[cell][1]] = hotTemp; 
        mesh[coolCells[cell][0]][coolCells[cell][1]] = coolTemp; 
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);

    srand(chrono::system_clock::now().time_since_epoch().count());  // seed rand() by current time

    vector<vector<int>> mesh(TINY, vector<int>(TINY, 25));  // initialize the mesh with the base temperature

    setTemps(mesh, rank, 75, 20);
    MPI_Barrier(MCW);
    
    int iterations = 10;  // change maybe?
    int timeStep = 0;
    while (timeStep < iterations) {

        printMesh(mesh, rank);

        if (rank == MASTER) {
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
                vector<int> receivedRow(mesh[0].size());
                
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
                    vector<int> updatedRow(mesh[0].size());

                    /*** update row ***/

                    // Send updated row back to master with the row index as the tag
                    MPI_Send(updatedRow.data(), updatedRow.size(), MPI_INT, MASTER, row, MCW);
                }
            }
        }    
        
        timeStep++;
        
        // Update each process's mesh
        for (int row = 0; row < mesh.size(); row++) {
            MPI_Bcast(mesh[row].data(), mesh[row].size(), MPI_INT, MASTER, MCW);
            MPI_Barrier(MCW);
        }
    }
    
    MPI_Finalize();
    return 0;
}