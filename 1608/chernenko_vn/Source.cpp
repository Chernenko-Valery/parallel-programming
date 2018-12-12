#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

using namespace std;

int* getRandImage(int width, int height, int max = 256) {
	int* result = new int[width * height];
	for (int i = 0; i < width * height; i++) result[i] = rand() % max;
	return result;
}

void printImage(int* image, int wigth, int height) {
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < wigth; i++)
			cout << image[i + j*wigth] << " ";
		cout << endl;
	}
}

int getPixel(int* image, int width, int height, int x, int y) {
	if (x < 0) x = 0;
	if (x >= width) x = width - 1;
	if (y < 0) y = 0;
	if (y >= height) y = height - 1;
	return image[x + y*width];
}

int getSobelPixel(int* image, int width, int height, int x, int y, int* sobelMatrix1, int* sobelMatrix2) {
	int resultX = 0;
	int resultY = 0;
	for(int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++) {
			resultX += getPixel(image, width, height, x + i, y + j) * sobelMatrix1[i + 1 + (j + 1) * 3];
			resultY += getPixel(image, width, height, x + i, y + j) * sobelMatrix2[i + 1 + (j + 1) * 3];
		}
	int result = sqrt(resultX*resultX + resultY*resultY);
	if (result < 0) return 0;
	if (result > 255) return 255;
	return result;
};

int* getSobelImage(int* image, int width, int height, int* sobelMatrix1, int* sobelMatrix2) {
	int* resultImage = new int[width * height] {0};
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++) {
			resultImage[i + j*width] = getSobelPixel(image, width, height, i, j, sobelMatrix1, sobelMatrix2);
		}
	return resultImage;
}

int* getPartlySobelImage(int* image, int width, int height, int startX, int startY, int toX, int toY, int* sobelMatrix1, int* sobelMatrix2) {
	int* resultImage = new int[width * height] {0};
	for (int i = startX; i < toX; i++)
		for (int j = startY; j < toY; j++) {
			resultImage[i + j*width] = getSobelPixel(image, width, height, i, j, sobelMatrix1, sobelMatrix2);
		}
	return resultImage;
}


int main(int argc, char** argv) {

	srand(time(0));
	int ProcRank, ProcSize;
	int* sobelMatrix1 = new int[9]{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	int* sobelMatrix2 = new int[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	if (ProcRank == 0) {

		int width = atoi(argv[1]);
		int height = atoi(argv[2]);
		int RowsForProc = height / ProcSize;
		int RestOfRows = height % ProcSize;
		MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&RowsForProc, 1, MPI_INT, 0, MPI_COMM_WORLD); 
		MPI_Bcast(&RestOfRows, 1, MPI_INT, 0, MPI_COMM_WORLD);

		int* image = getRandImage(width, height);
//		printImage(image, width, height);
//		cout << endl;

		if (ProcSize == 1 || RowsForProc == 0) {
			double startTime = MPI_Wtime();
			int* ResultImage = getSobelImage(image, width, height, sobelMatrix1, sobelMatrix2);
			double finishTime = MPI_Wtime();

			cout << "Time without parallelism = " << finishTime - startTime << endl;
			cout << "Result = " << endl;
//			printImage(ResultImage, width, height);
//			cout << endl;
			delete image;
			delete ResultImage;
			MPI_Finalize();
			return 0;
		}

		int* ResultImage = new int[width * height];
		double startTime = MPI_Wtime();
		int* sendCounts = new int[ProcSize];
		int* displs = new int[ProcSize];
		
		displs[0] = 0;
		if (RestOfRows > 0) {
			sendCounts[0] = (RowsForProc + 2)*width;
			--RestOfRows;
		} else sendCounts[0] = (RowsForProc + 1)*width;
		for (int i = 1; i < ProcSize - 1; i++) {
			displs[i] = displs[i - 1] + sendCounts[i - 1] - 2 * width;
			if (RestOfRows > 0) {
				sendCounts[i] = (RowsForProc + 3) * width;
				--RestOfRows;
			}
			else sendCounts[i] = (RowsForProc + 2) * width;
		}
		displs[ProcSize - 1] = displs[ProcSize - 2] + sendCounts[ProcSize - 2] - 2 * width;
		sendCounts[ProcSize - 1] = (RowsForProc + 1) * width;

		int* WorkImage = new int[sendCounts[0]];
		MPI_Scatterv(image, sendCounts, displs, MPI_INT, WorkImage, sendCounts[0], MPI_INT, 0, MPI_COMM_WORLD);
		
		int* ResultWorkImage = getPartlySobelImage(WorkImage, width, sendCounts[0] / width, 0, 0, width, sendCounts[0] / width - 1, sobelMatrix1, sobelMatrix2);

		RestOfRows = height % ProcSize;
		int* recvCounts = new int[ProcSize];
		displs[0] = 0;
		if (RestOfRows > 0) {
			recvCounts[0] = (RowsForProc + 1) * width;
			--RestOfRows;
		}
		else recvCounts[0] = RowsForProc * width;
		for (int i = 1; i < ProcSize; i++) {
			displs[i] = displs[i-1] + recvCounts[i - 1];
			if (RestOfRows > 0) {
				recvCounts[i] = (RowsForProc + 1) * width;
				--RestOfRows;
			}
			else recvCounts[i] = RowsForProc * width;
		}
	

		MPI_Gatherv(ResultWorkImage, recvCounts[0], MPI_INT, ResultImage, recvCounts, displs, MPI_INT, 0, MPI_COMM_WORLD);


		double finishTime = MPI_Wtime();
		cout << "Time with parallelism = " << finishTime - startTime << endl;
//		cout << "Result = " << endl;
//		printImage(ResultImage, width, height);
		cout << endl;

		startTime = MPI_Wtime();
		int* CheckResultImage = getSobelImage(image, width, height, sobelMatrix1, sobelMatrix2);
		finishTime = MPI_Wtime();
		bool check = true;
		for (int i = 0; i < width * height; i++) if (ResultImage[i] != CheckResultImage[i]) {
			check = false;
			break;
		}

		cout << "Time without parallelism = " << finishTime - startTime << endl;
		if (check) cout << "Check Success!" << endl;
		else cout << "Incorrect Answer!" << endl;

		delete image;
		delete ResultImage;
		delete CheckResultImage;
		delete sendCounts;
		delete recvCounts;
		delete displs;
		delete ResultWorkImage;
		delete WorkImage;
	}
	else {
		int width, height, RowsForProc, RestOfRows;
		MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&RowsForProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&RestOfRows, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (RowsForProc == 0) {
			MPI_Finalize();
			return 0;
		}

		if (ProcRank != ProcSize - 1) {
			if (ProcRank < RestOfRows) {
				int* WorkImage = new int[(RowsForProc + 3)* width];
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage, (RowsForProc + 3)* width, MPI_INT, 0, MPI_COMM_WORLD);
				int* ResultWorkImage = getPartlySobelImage(WorkImage, width, RowsForProc + 3, 0, 1, width, RowsForProc + 2, sobelMatrix1, sobelMatrix2);
				MPI_Gatherv(ResultWorkImage + width, (RowsForProc + 1) * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				delete WorkImage;
			}
			else {
				int* WorkImage = new int[(RowsForProc + 2)* width];
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage, (RowsForProc + 2)* width, MPI_INT, 0, MPI_COMM_WORLD);
				int* ResultWorkImage = getPartlySobelImage(WorkImage, width, RowsForProc + 2, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				MPI_Gatherv(ResultWorkImage + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				delete WorkImage;
			}
		}
		else {
				int* WorkImage = new int[(RowsForProc + 1)* width];
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage, (RowsForProc + 1)* width, MPI_INT, 0, MPI_COMM_WORLD);
				int* ResultWorkImage = getPartlySobelImage(WorkImage, width, RowsForProc + 1, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				MPI_Gatherv(ResultWorkImage + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				delete WorkImage;
		}
	}

	MPI_Finalize();
	return 0;
}