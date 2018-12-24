#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
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

int* getArrayInImage(Mat image, int flag = 0) {
	int* result = new int[image.rows*image.cols];
	if (flag < 0 || flag > 2) flag = 0;
	for (int i = 0; i < image.rows*image.cols; i++)
		result[i] = image.data[i * 3 + flag];
	return result;
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


		Mat image = imread(argv[1]);

		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", image);
		waitKey();

		int width = image.cols;
		int height = image.rows;
		cout << width << ' ' << height << endl;
		int RowsForProc = height / ProcSize;
		int RestOfRows = height % ProcSize;
		MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&RowsForProc, 1, MPI_INT, 0, MPI_COMM_WORLD); 
		MPI_Bcast(&RestOfRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
		int* imageData1 = getArrayInImage(image, 0);
		int* imageData2 = getArrayInImage(image, 1);
		int* imageData3 = getArrayInImage(image, 2);

		if (ProcSize == 1 || RowsForProc == 0) {
			double startTime = MPI_Wtime();
			int* ResultImage1 = getSobelImage(imageData1, width, height, sobelMatrix1, sobelMatrix2);
			int* ResultImage2 = getSobelImage(imageData2, width, height, sobelMatrix1, sobelMatrix2);
			int* ResultImage3 = getSobelImage(imageData3, width, height, sobelMatrix1, sobelMatrix2);
			double finishTime = MPI_Wtime();

			cout << "Time without parallelism = " << finishTime - startTime << endl;

			for (int i = 0; i < image.rows*image.cols; i++) {
				image.data[i * 3] = ResultImage1[i];
				image.data[i * 3 + 1] = ResultImage2[i];
				image.data[i * 3 + 2] = ResultImage3[i];
			}

			namedWindow("Display window", WINDOW_AUTOSIZE);
			imshow("Display window", image);
			waitKey();

			delete imageData1;
			delete imageData2;
			delete imageData3;
			delete ResultImage1;
			delete ResultImage2;
			delete ResultImage3;
			MPI_Finalize();
			return 0;
		}

		int* ResultImage1 = new int[width * height];
		int* ResultImage2 = new int[width * height];
		int* ResultImage3 = new int[width * height];
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

		int* WorkImage1 = new int[sendCounts[0]];
		int* WorkImage2 = new int[sendCounts[0]];
		int* WorkImage3 = new int[sendCounts[0]];
		MPI_Scatterv(imageData1, sendCounts, displs, MPI_INT, WorkImage1, sendCounts[0], MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(imageData1, sendCounts, displs, MPI_INT, WorkImage2, sendCounts[0], MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(imageData1, sendCounts, displs, MPI_INT, WorkImage3, sendCounts[0], MPI_INT, 0, MPI_COMM_WORLD);
		
		int* ResultWorkImage1 = getPartlySobelImage(WorkImage1, width, sendCounts[0] / width, 0, 0, width, sendCounts[0] / width - 1, sobelMatrix1, sobelMatrix2);
		int* ResultWorkImage2 = getPartlySobelImage(WorkImage2, width, sendCounts[0] / width, 0, 0, width, sendCounts[0] / width - 1, sobelMatrix1, sobelMatrix2);
		int* ResultWorkImage3 = getPartlySobelImage(WorkImage3, width, sendCounts[0] / width, 0, 0, width, sendCounts[0] / width - 1, sobelMatrix1, sobelMatrix2);

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
	

		MPI_Gatherv(ResultWorkImage1, recvCounts[0], MPI_INT, ResultImage1, recvCounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gatherv(ResultWorkImage2, recvCounts[0], MPI_INT, ResultImage2, recvCounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gatherv(ResultWorkImage3, recvCounts[0], MPI_INT, ResultImage3, recvCounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < image.rows*image.cols; i++) {
			image.data[i * 3] = ResultImage1[i];
			image.data[i * 3 + 1] = ResultImage2[i];
			image.data[i * 3 + 2] = ResultImage3[i];
		}

		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", image);
		waitKey();

		double finishTime = MPI_Wtime();
		cout << "Time with parallelism = " << finishTime - startTime << endl;

		delete imageData1;
		delete imageData2;
		delete imageData3;
		delete ResultImage1;
		delete ResultImage2;
		delete ResultImage3;
		delete sendCounts;
		delete recvCounts;
		delete displs;
		delete ResultWorkImage1;
		delete ResultWorkImage2;
		delete ResultWorkImage3;
		delete WorkImage1;
		delete WorkImage2;
		delete WorkImage3;
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
				int* WorkImage1 = new int[(RowsForProc + 3)* width];
				int* WorkImage2 = new int[(RowsForProc + 3)* width];
				int* WorkImage3 = new int[(RowsForProc + 3)* width];
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage1, (RowsForProc + 3)* width, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage2, (RowsForProc + 3)* width, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage3, (RowsForProc + 3)* width, MPI_INT, 0, MPI_COMM_WORLD);
				int* ResultWorkImage1 = getPartlySobelImage(WorkImage1, width, RowsForProc + 3, 0, 1, width, RowsForProc + 2, sobelMatrix1, sobelMatrix2);
				int* ResultWorkImage2 = getPartlySobelImage(WorkImage2, width, RowsForProc + 3, 0, 1, width, RowsForProc + 2, sobelMatrix1, sobelMatrix2);
				int* ResultWorkImage3 = getPartlySobelImage(WorkImage3, width, RowsForProc + 3, 0, 1, width, RowsForProc + 2, sobelMatrix1, sobelMatrix2);
				MPI_Gatherv(ResultWorkImage1 + width, (RowsForProc + 1) * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Gatherv(ResultWorkImage2 + width, (RowsForProc + 1) * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Gatherv(ResultWorkImage3 + width, (RowsForProc + 1) * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				delete WorkImage1;
				delete WorkImage2;
				delete WorkImage3;
			}
			else {
				int* WorkImage1 = new int[(RowsForProc + 2)* width];
				int* WorkImage2 = new int[(RowsForProc + 2)* width];
				int* WorkImage3 = new int[(RowsForProc + 2)* width];
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage1, (RowsForProc + 2)* width, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage2, (RowsForProc + 2)* width, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage3, (RowsForProc + 2)* width, MPI_INT, 0, MPI_COMM_WORLD);
				int* ResultWorkImage1 = getPartlySobelImage(WorkImage1, width, RowsForProc + 2, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				int* ResultWorkImage2 = getPartlySobelImage(WorkImage2, width, RowsForProc + 2, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				int* ResultWorkImage3 = getPartlySobelImage(WorkImage3, width, RowsForProc + 2, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				MPI_Gatherv(ResultWorkImage1 + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Gatherv(ResultWorkImage2 + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Gatherv(ResultWorkImage3 + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				delete WorkImage1;
				delete WorkImage2;
				delete WorkImage3;
			}
		}
		else {
				int* WorkImage1 = new int[(RowsForProc + 1)* width];
				int* WorkImage2 = new int[(RowsForProc + 1)* width];
				int* WorkImage3 = new int[(RowsForProc + 1)* width];
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage1, (RowsForProc + 1)* width, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage2, (RowsForProc + 1)* width, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, WorkImage3, (RowsForProc + 1)* width, MPI_INT, 0, MPI_COMM_WORLD);
				int* ResultWorkImage1 = getPartlySobelImage(WorkImage1, width, RowsForProc + 1, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				int* ResultWorkImage2 = getPartlySobelImage(WorkImage2, width, RowsForProc + 1, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				int* ResultWorkImage3 = getPartlySobelImage(WorkImage3, width, RowsForProc + 1, 0, 1, width, RowsForProc + 1, sobelMatrix1, sobelMatrix2);
				MPI_Gatherv(ResultWorkImage1 + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Gatherv(ResultWorkImage2 + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Gatherv(ResultWorkImage3 + width, RowsForProc * width, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
				delete WorkImage1;
				delete WorkImage2;				
				delete WorkImage3;
		}
	}

	MPI_Finalize();
	return 0;
}