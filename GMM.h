#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
using namespace cv;
using namespace std;

class GMM
{
public:
	void init(Mat img,Mat mask,int clustNum = 5,double eps_end = 0.01,double eps = 1e-16,int max_steps = 30);
	void train();
	double pred(double X[]);
private:
	int clustNum,max_steps;
	int N,M,n;  
	double eps,eps_end,logL_lst;//对数似然函数值 
	vector<int> belong;
	vector<double> det;
	vector<double> prob; // 样本的可能性 
	vector<double> weight; // 高斯分布的权重
	vector<vector<double> > memberships; // p(i|k)
	vector<Mat> means; //均值
	vector<Mat> sigmas; //协方差 
	vector<Mat> sigmas_inv; //协方差 ^-1
	Mat data,imgMat;
	
//	double gauss(Mat x,Mat inv,double det);
	double gauss(int m,double X[]);
} ;
