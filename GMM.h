#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class GMM
{
	int clustNum,max_steps;
	int belong[2024 * 2024]; 
	double eps;
	double prob[10],prob_lst[10]; // ���Ŀ����� 
	double means[10],means_lst[10]; //��ֵ
	double sigmas[10],sigmas_lst[10]; //Э���� 
	Mat data,data_lst;
	int N,M; 
	
	void init(Mat img,int clustNum = 5,double eps = 0.01,int max_steps = 20);
	void train();
	int pred(int x);
} ;
