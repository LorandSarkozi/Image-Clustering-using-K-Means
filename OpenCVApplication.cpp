

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <cmath> 
#include <iomanip> 
#include <iostream> 

wchar_t* projectPath;

using namespace std;

bool isInside(Mat& img, int i, int j) {
	return (i >= 0 && j >= 0 && i < img.rows&& j < img.cols);
}

vector<int> calculateHistogram(const Mat& image) {
	vector<int> histogram(256, 0);

	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			Vec3b pixel = image.at<Vec3b>(i, j);
			int intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;
			histogram[intensity]++;
		}
	}

	return histogram;
}


vector<Vec3b> extractDominantColors(const Mat& image, int numColors) {
	vector<int> histogram = calculateHistogram(image);

	vector<Vec3b> dominantColors;
	for (int i = 1; i < histogram.size() - 1; ++i) {

		if (histogram[i] > histogram[i - 1] && histogram[i] > histogram[i + 1]) {
			
			dominantColors.push_back(Vec3b(i, i, i));

			if (dominantColors.size() == numColors) {
				break;
			}
		}
	}

	return dominantColors;
}

void FilterCreation(double GKernel[][5])
{

	double sigma = 0.8; //deviatia std

	double r, s = 2.0 * sigma * sigma;
 
	double sum = 0.0;

	
	for (int x = -2; x <= 2; x++) {
		for (int y = -2; y <= 2; y++) {
			r = sqrt(x * x + y * y);
			GKernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (PI * s);
			sum += GKernel[x + 2][y + 2];
		}
	}

	
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			GKernel[i][j] /= sum;
}


Mat applyGaussianFilter(const Mat& src, double GKernel[][5])
{
	Mat dst = src.clone();

	
	for (int c = 0; c < src.channels(); ++c) {
		for (int x = 2; x < src.rows - 2; ++x) {
			for (int y = 2; y < src.cols - 2; ++y) {
				double sum = 0.0;
				for (int i = -2; i <= 2; ++i) {
					for (int j = -2; j <= 2; ++j) {
						sum += src.at<Vec3b>(x + i, y + j)[c] * GKernel[i + 2][j + 2];
					}
				}
				dst.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(sum);
			}
		}
	}

	return dst;
}



Mat testGaussianFilter(Mat src)
{

	double GKernel[5][5];
	FilterCreation(GKernel);

	Mat dst = applyGaussianFilter(src, GKernel);

	return dst;
}

void elemStruct(int range, vector<int>& xAxis, vector<int>& yAxis) {
	//range = 3 or 5 or 7 ...
	if (range == 3) {
		//vecinetate 4
		xAxis = { 0, -1, 1, 0 };
		yAxis = { -1, 0, 0, 1 };
	}
	else if (range == 4) {
		//vecinatate 8
		xAxis = { -1, 0, 1, -1, 1, -1, 0, 1 };
		yAxis = { -1, -1, -1, 0, 0, 1, 1, 1 };
	}
	else if (range == 5) {
		xAxis = { 0, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1, 0 };
		yAxis = { -2, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2 };
	}
}


double euclideanDistance(const Vec3b& p1, const Vec3b& p2) {
	return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

vector<Vec3b> initializeCentroids(const Mat& data, int k) {
	vector<Vec3b> dominantColors = extractDominantColors(data, k);

	return dominantColors;
}


vector<int> assignClusters(const Mat& data, const vector<Vec3b>& centroids) {
	vector<int> labels(data.rows * data.cols, 0);
	for (int i = 0; i < data.rows; i++) {
		for (int j = 0; j < data.cols; j++) {
			Vec3b point = data.at<Vec3b>(i, j);
			double minDist = DBL_MAX;
			int label = 0;
			for (int c = 0; c < centroids.size(); c++) {
				double dist = euclideanDistance(point, centroids[c]);
				if (dist < minDist) {
					minDist = dist;
					label = c;
				}
			}
			labels[i * data.cols + j] = label;
		}
	}
	return labels;
}


vector<Vec3b> updateCentroids(const Mat& data, const vector<int>& labels, int k) {
	vector<Vec3i> centroidsSum(k, Vec3i(0, 0, 0));
	vector<int> count(k, 0);

	for (int i = 0; i < data.rows; i++) {
		for (int j = 0; j < data.cols; j++) {
			int label = labels[i * data.cols + j];
			centroidsSum[label] += data.at<Vec3b>(i, j);
			count[label]++;
		}
	}

	vector<Vec3b> centroids(k);
	for (int c = 0; c < k; c++) {
		if (count[c] != 0) {
			centroids[c][0] = centroidsSum[c][0] / count[c];
			centroids[c][1] = centroidsSum[c][1] / count[c];
			centroids[c][2] = centroidsSum[c][2] / count[c];
		}
	}

	return centroids;
}


void kMeans(const Mat& data, int k, int iterations, Mat& labels,vector<Vec3b>& finalCentroids) {
	
	 finalCentroids = initializeCentroids(data, k);

	vector<int> clusterLabels;
	bool converged = false;

	for (int iter = 0; iter < iterations; iter++) {
		
		clusterLabels = assignClusters(data, finalCentroids);

		
		vector<Vec3b> newCentroids = updateCentroids(data, clusterLabels, k);

		double maxShift = 0;
		for (int c = 0; c < k; c++) {
			double shift = euclideanDistance(finalCentroids[c], newCentroids[c]);
			if (shift > maxShift) {
				maxShift = shift;
			}
		}

		finalCentroids = newCentroids;

		if (maxShift < 1.0) {
			converged = true;
			break;
		}
	}

	
	labels.create(data.size(), CV_32S);
	for (int i = 0; i < data.rows; i++) {
		for (int j = 0; j < data.cols; j++) {
			labels.at<int>(i, j) = clusterLabels[i * data.cols + j];
		}
	}
}



void erodeLabels(Mat& labels) {
	Mat new_labels = labels.clone();
	vector<int> xAxis, yAxis;
	elemStruct(3, xAxis, yAxis);

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if (labels.at<int>(i, j) != -1) {
				int current_label = labels.at<int>(i, j);
				bool same_color_neighbors = true;

				for (int k = 0; k < xAxis.size(); k++) {
					int ni = i + xAxis[k];
					int nj = j + yAxis[k];
					if (isInside(labels, ni, nj)) {
						int neighbor_label = labels.at<int>(ni, nj);
						if (neighbor_label != -1 && neighbor_label != current_label) {
							same_color_neighbors = false;
							break;
						}
					}
				}

				if (!same_color_neighbors) {
					new_labels.at<int>(i, j) = -1;
				}
			}
		}
	}

	labels = new_labels.clone();
}


void dilateLabels(Mat& labels) {
	Mat new_labels = labels.clone();
	vector<int> xAxis, yAxis;
	elemStruct(3, xAxis, yAxis);

	const int rows = labels.rows;
	const int cols = labels.cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			if (labels.at<int>(i, j) != -1) {

				for (int k = 0; k < xAxis.size(); k++) {
					int ni = i + xAxis[k];
					int nj = j + yAxis[k];

					if (isInside(labels, ni, nj)) {

						if (labels.at<int>(ni, nj) == -1)
							new_labels.at<int>(ni, nj) = labels.at<int>(i, j);
					}
				}
			}
		}
	}

	labels = new_labels.clone();
}

void postProcessLabels(Mat& labels) {
	for (int i = 0; i < 3; i++) {
		erodeLabels(labels);
	}

	bool hasUnlabeled;
	do {
		hasUnlabeled = false;
		Mat new_labels = labels.clone();
		vector<int> xAxis, yAxis;
		elemStruct(3, xAxis, yAxis);

		for (int i = 0; i < labels.rows; i++) {
			for (int j = 0; j < labels.cols; j++) {
				if (labels.at<int>(i, j) == -1) {
					int neighbor_label = -1;
					int count = 0;
					for (int k = 0; k < xAxis.size(); k++) {
						int ni = i + xAxis[k];
						int nj = j + yAxis[k];
						if (isInside(labels, ni, nj) && labels.at<int>(ni, nj) != -1) {
							neighbor_label = labels.at<int>(ni, nj);
							count++;
						}
					}
					if (count > 0) {
						new_labels.at<int>(i, j) = neighbor_label;
						hasUnlabeled = true;
					}
				}
			}
		}
		labels = new_labels.clone();
	} while (hasUnlabeled);
}




int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	double GKernel[5][5];
	FilterCreation(GKernel);

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++)
			cout << GKernel[i][j] << "\t";
		cout << endl;
	}
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);


		Mat gaussian = testGaussianFilter(src);
		//imshow("Gaussian", gaussian);

		Mat dst3 = Mat(gaussian.rows, gaussian.cols, CV_8UC3);
		cvtColor(gaussian, dst3, COLOR_BGR2YCrCb);
		//imshow("YCbCr Image", dst3);

		int K;
		cout << "Numar centroide =  ";
		cin >> K;

		int I;
		cout <<"Numar iteratii: = ";
		cin >> I;

		Mat labels;

		vector<Vec3b> finalCentroids(K, 0);
		kMeans(dst3, K, I, labels,finalCentroids);

		postProcessLabels(labels);

		Mat clustered = Mat::zeros(dst3.size(), dst3.type());
		
		
		for (int i = 0; i < dst3.rows; i++) {
			for (int j = 0; j < dst3.cols; j++) {
				int label = labels.at<int>(i, j);
				clustered.at<Vec3b>(i, j) =finalCentroids[label];
			}
		}

		imshow("Original", src);
		imshow("Clustered Image", clustered);

		Mat final = Mat(clustered.rows, clustered.cols, CV_8UC3);
		cvtColor(clustered, final, COLOR_YCrCb2BGR);
		imshow("BGR Reverted", final);

		waitKey();
	}

		return 0;
}