#include <mpi.h>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <limits.h>

using namespace std;


//������� ���� �� ����� input.txt
char* path = "input.txt";

int main(int argc, char **argv) {

	//������ MPI
	int ProcRank, ProcSize;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);


	//������ 0 ��������
	if (ProcRank == 0) {

		//������ ������� �� �����
		int x, y;
		double times = MPI_Wtime();
		FILE* fin;
		if (!(fin = fopen(path, "r"))) {
			printf("File not open!");
			return 0;
		}
		fscanf(fin, "%d %d", &x, &y);
		int* arr = (int*)malloc(x * y * sizeof(int*));
		for(int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) fscanf(fin, "%d ", arr + i*y + j);
		fclose(fin);

		//���� ������� ����� 1
		if (ProcSize == 1) {
			int *totalResult = (int*)malloc(y * sizeof(int)); //������ ����������
			for (int i = 0; i < y; i++) totalResult[i] = INT_MIN; //"���������" ������ ����������
			for (int i = 0; i < x; i++) 
				for (int j = 0; j < y; j++)
					if (totalResult[j] < arr[i*y + j])
						totalResult[j] = arr[i*y + j];

			//�����
			for (int i = 0; i < y; i++) printf("%d ", totalResult[i]);
			printf("\nTime = %f\n", MPI_Wtime() - times);

			//������������ ������ � ��������� ������
			free(totalResult);
			free(arr);

			MPI_Finalize();
			return 0;
		}

		//���� �� 1 �������. 
		//�������� ���������� � �������� �������
		for (int i = 1; i < ProcSize; i++) {
			MPI_Send(&x, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&y, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		//���������� ���������� ����� ��� ������� ��������
		int k = x / (ProcSize - 1);

		//�������� ���������� ������� ���� ��������� ���������, ����� ����������
		for (int i = 1; i < ProcSize - 1; i++) {
			MPI_Send(arr + (i-1)*k*y, k*y, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
		//����������� ����������� ������� ���������� ��������
		MPI_Send(arr + (ProcSize - 2)*k*y, (x - (ProcSize - 2)*k)*y, MPI_INT, ProcSize - 1, 0, MPI_COMM_WORLD);

		//�������� ��������� � �������������� ������
		int* result = (int*)malloc(y * sizeof(int));
		int *totalResult = (int*)malloc(y * sizeof(int));
		for (int i = 0; i < y; i++)
			totalResult[i] = INT_MIN;
		//��������� ���������� �� ���������
		for (int i = 1; i < ProcSize; i++) {
			MPI_Recv(result, y, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			for (int j = 0; j < y; j++) 
				if (totalResult[j] < result[j]) 
					totalResult[j] = result[j];
		}

		//����� � ������������ ������.
		for (int i = 0; i < y; i++) printf("%d ", totalResult[i]);
		printf("\nTime = %f\n", MPI_Wtime() - times);

		free(result);
		free(totalResult);
		free(arr);
	}
	//������ ��������
	else {
		//��������� �������� ������� � ���������� ����� �� �������
		int x, y;
		MPI_Recv(&x, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&y, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int k = x / (ProcSize - 1);

		//��������� ������ ��� ����������� �������
		int* arr;
		if( ProcRank != ProcSize - 1)
			arr = (int*)malloc(y * k * sizeof(int));
		else arr = (int*)malloc((x - (ProcSize - 2)*k) * y * sizeof(int));

		//��������� ������ ��� ���������
		int* result = (int*)malloc(y * sizeof(int));
		for (int i = 0; i < y; i++)
			result[i] = INT_MIN;

		//���������� ���������� ��� �� ���������� ��������
		if (ProcRank != ProcSize - 1) {
			MPI_Recv(arr, k*y, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			for (int i = 0; i < k; i++)
				for (int j = 0; j < y; j++)
					if (result[j] < arr[i*y + j]) result[j] = arr[i*y + j];
		}
		//���������� ���������� ��� ���������� ��������
		else {
			MPI_Recv(arr, (x - (ProcSize - 2)*k) * y, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			for (int i = 0; i < x - (ProcSize - 2)*k; i++)
				for (int j = 0; j < y; j++)
					if (result[j] < arr[i*y + j]) result[j] = arr[i*y + j];
		}
		//�������� ����������
		MPI_Send(result, y, MPI_INT, 0, 0, MPI_COMM_WORLD);
		free(result);
		free(arr);
	}

	MPI_Finalize();
	return 0;
}

