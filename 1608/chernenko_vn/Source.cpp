#include <mpi.h>
#include <iostream>
#include <fstream>
#include <malloc.h>
#include <stdlib.h>
#include <limits.h>

using namespace std;


//path to file input.txt

int main(int argc, char **argv) {

	//Run MPI
	int ProcRank, ProcSize;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);


	//Work 0 Process
	if (ProcRank == 0) {

		//Read array from file
		int x, y;
		double times = MPI_Wtime();
		ifstream fin(argv[1]);
		fin >> x >> y;
		int* arr = new int[x * y];
		for(int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) fin >> *(arr + i*y + j);
		fin.close();

		// Calculate the number of rows for each process
		int k = x / (ProcSize - 1);

		//If the process is only or count rows < ProcSize-1
		if (ProcSize == 1 || k == 0) {
			int *totalResult = new int[y]; //Result string
			for (int i = 0; i < y; i++) totalResult[i] = INT_MIN; //"Zeroing" the result string
			for (int i = 0; i < x; i++) 
				for (int j = 0; j < y; j++)
					if (totalResult[j] < arr[i*y + j])
						totalResult[j] = arr[i*y + j];

			//Result
			for (int i = 0; i < y; i++) 
				cout << totalResult[i] << " ";
			cout << endl << "Time = " << MPI_Wtime() - times << endl;

			//Memory release and end of work
			delete totalResult;
			delete arr;

			if (ProcRank == 0) { char n; cin >> n; }
			MPI_Finalize();
			return 0;
		}

		// If not 1 process.
		// Transmission of information about the size of the matrix
		for (int i = 1; i < ProcSize; i++) {
			MPI_Send(&x, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&y, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		// Sending the shortened matrix to all other processes, except the last
		for (int i = 1; i < ProcSize - 1; i++) {
			MPI_Send(arr + (i - 1)*k*y, k*y, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		// Sending the shortened matrix to the last process
		MPI_Send(arr + (ProcSize - 2)*k*y, (x - (ProcSize - 2)*k)*y, MPI_INT, ProcSize - 1, 0, MPI_COMM_WORLD);

		// Create a temporary and result string
		int* result = new int[y];
		int *totalResult = new int[y];
		MPI_Recv(result, y, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		for (int i = 0; i < y; i++)
			totalResult[i] = result[i];
		// Getting information from processes
		for (int i = 2; i < ProcSize; i++) {
			MPI_Recv(result, y, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			for (int j = 0; j < y; j++) 
				if (totalResult[j] < result[j]) 
					totalResult[j] = result[j];
		}

		//Memory release and end of work
		for (int i = 0; i < y; i++) cout << totalResult[i] << " ";
		cout << endl << "Time = " << MPI_Wtime() - times << endl;

		delete result;
		delete totalResult;
		delete arr;
	}
	// Other processes
	else {
		// Get the dimensions of the matrix and the number of rows per process
		int x, y;
		MPI_Recv(&x, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&y, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int k = x / (ProcSize - 1);
		if (k == 0) return 0;

		// Memory allocation for a shortened matrix
		int* arr;
		if (ProcRank != ProcSize - 1)
			arr = new int[y * k];
		else arr = new int[(x - (ProcSize - 2)*k) * y];
		// Memory allocation for the result
		int* result = new int[y];

		// Calculate the result for not the last process
		if (ProcRank != ProcSize - 1) {
			MPI_Recv(arr, k*y, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			for (int j = 0; j < y; j++) 
				result[j] = arr[j];
			for (int i = 1; i < k; i++)
				for (int j = 0; j < y; j++)
					if (result[j] < arr[i*y + j]) result[j] = arr[i*y + j];
		}
		// Calculate the result for the last process
		else {
			MPI_Recv(arr, (x - (ProcSize - 2)*k) * y, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			for (int j = 0; j < y; j++) 
				result[j] = arr[j];
			for (int i = 0; i < x - (ProcSize - 2)*k; i++)
				for (int j = 0; j < y; j++)
					if (result[j] < arr[i*y + j]) result[j] = arr[i*y + j];
		}
		// Send the result
		MPI_Send(result, y, MPI_INT, 0, 0, MPI_COMM_WORLD);
		delete result;
		delete arr;
	}

	if (ProcRank == 0) { char n; cin >> n; }
	MPI_Finalize();
	return 0;
}