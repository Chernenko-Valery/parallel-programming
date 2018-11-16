#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
using namespace std;

//Getting a random matrix with numbers in range [0, max)
int* GetRandomMatrix(int x, int y, int max) {
	int* arrResult = new int[x*y];
	for (int i = 0; i < x*y; i++) arrResult[i] = rand() % max;
	return arrResult;
}

//Output matrix to console
void PrintMatrix(int* arr, int x, int y) {
	for (int i = 0; i < x; i++) {
		cout.width(4);
		for (int j = 0; j < y; j++) cout << arr[i*y + j] << " "; 
		cout << endl;
	}
}

//Output matrix to console
void PrintMatrix(long* arr, int x, int y) {
	for (int i = 0; i < x; i++) {
		cout.width(4);
		for (int j = 0; j < y; j++) cout << arr[i*y + j] << " ";
		cout << endl;
	}
}

//calculate all matrix
long* MultMatrOnVect(int* matrix, int x, int y, int* vector, int size, int startIndex = 0) {
	if (y != size) {
		cout << "wrong format!" << endl;
		return nullptr;
	}
	if (startIndex < 0 || startIndex > y) startIndex = 0;
	long* resultMatrix = new long[x] {0};
	for (int i = 0; i < x; i++)
		for (int j = startIndex; j < y; j++)
			resultMatrix[i] += matrix[i*y + j] * vector[j];
	return resultMatrix;
}

//calculate patr of columns
long* MultPartMatrOnVect(int* matrix, int x, int y, int* vector, int size, int startIndex, int countColumns) {
	if (startIndex < 0 || startIndex > y) startIndex = 0;
	if (countColumns < 0 || countColumns > y) countColumns = y;
	if (countColumns != size) {
		cout << "wrong format!" << endl;
		return nullptr;
	}
	long* resultMatrix = new long[x] {0};
	for (int i = 0; i < x; i++)
		for (int j = startIndex; j < y; j++)
			resultMatrix[i] += matrix[i*y + j] * vector[j - startIndex];
	return resultMatrix;
}

int main(int argc, char **argv)
{
	setlocale(LC_ALL, "Russian");
	srand(time(0));

	int ProcRank, ProcSize;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	int x, y, k;
	MPI_Status status;
	MPI_Datatype columnsType;
	MPI_Datatype lastColumnsType;

	if (ProcRank == 0) {
		x = atoi(argv[1]);
		y = atoi(argv[2]);
		if (ProcSize == 1) k = 1;
		else k = y / (ProcSize - 1);
		int lastY = (ProcSize - 1)*k;
		int lastCount = y - lastY;

		//Create columnsType
		int* sizes = new int[x] {x, y};
		int* newSizes = new int[x] {x, k};
		int* starts = new int[x] {0, 0};
		MPI_Type_create_subarray(2, sizes, newSizes, starts, MPI_ORDER_C, MPI_INT, &columnsType);
		MPI_Type_commit(&columnsType);

		//Create lastColumnsType
		int* lastSizes = new int[x] {x, y};
		int* lastNewSizes = new int[x] {x, k + lastCount};
		int* lastStarts = new int[x] {0, 0};
		MPI_Type_create_subarray(2, lastSizes, lastNewSizes, lastStarts, MPI_ORDER_C, MPI_INT, &lastColumnsType);
		MPI_Type_commit(&lastColumnsType);

		MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//matrix creation
		int* matrix = GetRandomMatrix(x, y, 10);

		//vector creation
		int* vector = GetRandomMatrix(y, 1, 10);
		
		long* resultVector = new long[x] {0};

		double timeStart = MPI_Wtime();

		//without parallelization
		if (ProcSize == 1 || k < 2) {
			long* resultVector = MultMatrOnVect(matrix, x, y, vector, y);
			double timeEnd = MPI_Wtime();
			cout << "Result = " << endl;
			PrintMatrix(resultVector, 1, x);
			cout << "Time = " << timeEnd - timeStart << endl;
			delete matrix;
			delete vector;
			delete resultVector;
			MPI_Type_free(&columnsType);
			MPI_Finalize();
			return 0;
		}

		int* vectorsendcounts = new int[ProcSize];
		int* vectordispls = new int[ProcSize];
		for (int i = 0; i < ProcSize; i++) {
			vectorsendcounts[i] = k;
			vectordispls[i] = (i - 1)*k;
		}
		vectorsendcounts[ProcSize - 1] = k + lastCount;
		vectorsendcounts[0] = 0;
		vectordispls[0] = 0;
		int* lastVector;;
		MPI_Scatterv(vector, vectorsendcounts, vectordispls, MPI_INT, lastVector, 0, MPI_INT, 0, MPI_COMM_WORLD);
		delete vectorsendcounts;
		delete vectordispls;

		for (int i = 1; i < ProcSize - 1; i++)
			MPI_Send(matrix + (i - 1)*k, 1, columnsType, i, 0, MPI_COMM_WORLD);
		MPI_Send(matrix + (ProcSize - 2)*k, 1, lastColumnsType, ProcSize - 1, 0, MPI_COMM_WORLD);
		long* tmpResultVector = new long[x] {0};

		MPI_Reduce(tmpResultVector, resultVector, x, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

		double timeEnd = MPI_Wtime();
		cout << "Result = ";
		PrintMatrix(resultVector, 1, x);
		cout << "Time with parallelization= " << timeEnd - timeStart << endl;

		//Check
		bool check = true;
		double checkTimeStart = MPI_Wtime();
		long* checkResult = MultMatrOnVect(matrix, x, y, vector, y);
		double checkTimeEnd = MPI_Wtime();
		for (int i = 0; i < x; i++) if (checkResult[i] != resultVector[i]) check = false;
		if (check == true)
			cout << "Check success!" << endl;
		else cout << "Encorrect answer!" << endl;
		cout << "Time without parallelization= " << checkTimeEnd - checkTimeStart << endl;

		delete matrix;
		delete vector;
		delete tmpResultVector;
		delete resultVector;
		delete checkResult;
		delete sizes;
		delete newSizes;
		delete starts;
		delete lastSizes;
		delete lastNewSizes;
		delete lastStarts;
		MPI_Type_free(&lastColumnsType);

	}
	else if (ProcRank == ProcSize - 1) {

		MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
		int lastY = (ProcSize - 1)*k;
		int lastCount = y - lastY;

		int* sizes = new int[x] {x, y};
		int* newSizes = new int[x] {x, k + lastCount};
		int* starts = new int[x] {0, 0};
		MPI_Type_create_subarray(2, sizes, newSizes, starts, MPI_ORDER_FORTRAN, MPI_INT, &columnsType);
		MPI_Type_commit(&columnsType);

		if (k > 1) {
			int* matrix = new int[x*(k+lastCount)]{ 0 };
			int* vector = new int[k];
			MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, vector, k + lastCount, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Recv(matrix, 1, columnsType, 0, 0, MPI_COMM_WORLD, &status);
			double timeStart = MPI_Wtime();
			long* resultVector = MultMatrOnVect(matrix, x, k + lastCount, vector, k + lastCount);
			double timeEnd = MPI_Wtime();
			MPI_Reduce(resultVector, nullptr, x, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
			cout << "Time on Proc " << ProcRank << " = " << timeEnd - timeStart << endl;
			delete matrix;
			delete vector;
			delete resultVector;
		}
		delete sizes;
		delete newSizes;
		delete starts;

	} else {

		MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);


		int* sizes = new int[x] {x, y};
		int* newSizes = new int[x] {x, k};
		int* starts = new int[x] {0, 0};
		MPI_Type_create_subarray(2, sizes, newSizes, starts, MPI_ORDER_FORTRAN, MPI_INT, &columnsType);
		MPI_Type_commit(&columnsType);

		if (k > 1) {
			int* matrix = new int[x*k]{0};
			int* vector = new int[k];
			MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, vector, k, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Recv(matrix, 1, columnsType, 0, 0, MPI_COMM_WORLD, &status);
			double timeStart = MPI_Wtime();
			long* resultVector = MultMatrOnVect(matrix, x, k, vector, k);
			double timeEnd = MPI_Wtime();
			MPI_Reduce(resultVector, nullptr, x, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
			cout << "Time on Proc " << ProcRank << " = " << timeEnd - timeStart << endl;
			delete matrix;
			delete vector;
			delete resultVector;
		}
		delete sizes;
		delete newSizes;
		delete starts;
	}
	MPI_Type_free(&columnsType);
	MPI_Finalize();
	return 0;

}