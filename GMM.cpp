#include "GMM.h"
#define id(i,j) ((i) * M + (j)) //(i-1)´øÈë 

using namespace std;
void GMM :: init(Mat img,int clustNum,double eps,int max_steps)
{
	this -> data = img;
	this -> data_lst = Mat::zeros(img.size(),CV_8UC3);
	this -> clustNum = clustNum;
	this -> max_steps = max_steps;
	this -> eps = eps;
	this -> N = img.rows;
	this -> M = img.cols;
	
	for(int i = 0;i < clustNum;++ i)
	{
		prob[i] = prob_lst[i] = 1.0 / clustNum;
		means[i] = means_lst[i] = 255.0 * i / clustNum;
		sigmas[i] = sigmas_lst[i] = 50;
	}
} 

void GMM :: train()
{
	int step = 0;
	do
	{
		for(int k = 0;k < clustNum; ++ k)
		{
			for(int i = 0;i < N; ++ i)
				for(int j = 0;j < M; ++ j)
				{
					double sum = 0;
					for(int m = 0;m < clustNum; ++ m)
					{
						sum += prob[m] * gauss; // pi u sigma
					}
					
				}
		} 
	}while( (step ++) < max_steps);
}
