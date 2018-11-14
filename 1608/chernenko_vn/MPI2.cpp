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

int* MultMatrOnVect(int* matrix, int x, int y, int* vector, int size) {
	if (y != size) {
		cout << "wrong format!" << endl;
		return nullptr;
	}
	int *resultMatrix = new int[x] {0};
	for (int i = 0; i < x; i++)
		for (int j = 0; j < y; j++)
			resultMatrix[i] += matrix[i*y + j] * vector[j];
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

	if (ProcRank == 0) {

		x = atoi(argv[1]);
		y = atoi(argv[2]);
		if (ProcSize == 1) k = 1;
		else k = y / (ProcSize - 1);

		MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//matrix creation
		int* matrix = GetRandomMatrix(x, y, 10);
//		cout << "Matrix =" << endl;
//		PrintMatrix(matrix, x, y);
//		cout << endl;
		//vector creation
		int* vector = GetRandomMatrix(y, 1, 10);
//		cout << "Vector = " << endl;
//		PrintMatrix(vector, y, 1);
//		cout << endl;
		
		//Create columnsType
		MPI_Type_vector(x, k, y, MPI_INT, &columnsType);
		MPI_Type_commit(&columnsType);

		//Start Time
		double timeStart = MPI_Wtime();

		//without parallelization
		if (ProcSize == 1 || k < 2) {
			int* ResultVector = MultMatrOnVect(matrix, x, y, vector, y);
			double timeEnd = MPI_Wtime();
			cout << "Result = " << endl;
			PrintMatrix(ResultVector, 1, x);
			cout << "Time = " << timeEnd - timeStart << endl;
			delete matrix;
			delete vector;
			delete ResultVector;
			MPI_Finalize();
			return 0;
		}


		int* ResultVector = new int[x] {0};
		for (int i = 1; i < ProcSize; i++) {
			int* buf = new int[x*k];
			int* bufVect = new int[k];
			for (int j = 0; j < k; j++) bufVect[j] = vector[(i - 1)*k + j];
			int bufpos = 0;
			MPI_Pack(matrix + (i - 1)*k, 1, columnsType, buf, x*k * sizeof(int), &bufpos, MPI_COMM_WORLD);
			MPI_Send(buf, x*k, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(bufVect, k, MPI_INT, i, 0, MPI_COMM_WORLD);
			delete buf;
			delete bufVect;
			}
		//count last columns
		if ((ProcSize - 1) *k != y) {
			MPI_Datatype lastColumn;
			int lastY = y - (ProcSize - 1)*k;
			MPI_Type_vector(x, lastY, y, MPI_INT, &lastColumn);
			MPI_Type_commit(&lastColumn);
			int* lastMatrix = new int[x*lastY];
			int* lastVector = new int[lastY];
			for (int i = 0; i < lastY; i++) lastVector[i] = vector[(ProcSize - 1)*k + i];
			int lastMatrixPos = 0;
			MPI_Pack(matrix + (ProcSize - 1)*k, 1, lastColumn, lastMatrix, x * lastY * sizeof(int), &lastMatrixPos, MPI_COMM_WORLD);
			MPI_Type_free(&lastColumn);
			delete lastMatrix;
			delete lastVector;
			ResultVector = MultMatrOnVect(lastMatrix, x, lastY, lastVector, lastY);
		}
		int* tmpResult = new int[x];
		for (int i = 1; i < ProcSize; i++) {
			MPI_Recv(tmpResult, x, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			for (int i = 0; i < x; i++) ResultVector[i] += tmpResult[i];
		}
		delete tmpResult;
		double timeEnd = MPI_Wtime();
		cout << "Result = ";
		PrintMatrix(ResultVector, 1, x);
		cout << "Time with parallelization= " << timeEnd - timeStart << endl;

		//Check
		bool check = true;
		double checkTimeStart = MPI_Wtime();
		int* checkResult = MultMatrOnVect(matrix, x, y, vector, y);
		double checkTimeEnd = MPI_Wtime();
		for (int i = 0; i < x; i++) if (checkResult[i] != ResultVector[i]) check = false;
		if (check = true)
			cout << "Check success!" << endl;
		else cout << "Encorrect answer!" << endl;
		cout << "Time without parallelization= " << checkTimeEnd - checkTimeStart << endl;

		MPI_Type_free(&columnsType);
		delete matrix;
		delete vector;
		delete ResultVector;
		delete checkResult;

	} else {
		MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (k > 1) {
			int* matrix = new int[x*k];
			int* vector = new int[k];
			MPI_Recv(matrix, x*k, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(vector, k, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			double timeStart = MPI_Wtime();
			int* resultVector = MultMatrOnVect(matrix, x, k, vector, k);
			double timeEnd = MPI_Wtime();
			MPI_Send(resultVector, x, MPI_INT, 0, 0, MPI_COMM_WORLD);
			cout << "Time on Proc" << ProcRank << " = " << timeEnd - timeStart << endl;
			delete matrix;
			delete vector;
			delete resultVector;
		}
	}

	MPI_Finalize();
	return 0;

}