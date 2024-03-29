#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
using namespace cv;
using namespace std;
using namespace cv::ml;

class GMM
{
public:
	void init(Mat img,Mat mask,int clustNum = 5);
	void train();
	double pred(Mat&x);	
private:
	int clustNum;
	int N,M,n;
	
	Ptr<EM> model;
	Mat imgMat,probs;
	
//	double gauss(Mat x,Mat inv,double det);
	double gauss(int m,double X[]);
} ;
