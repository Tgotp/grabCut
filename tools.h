#include<opencv2/opencv.hpp>
#include <cstdio>
using namespace std;

void GetHist(Mat gray,Mat &Hist,Mat mask)    //ͳ��8Bit����ͼ��ĻҶ�ֱ��ͼ
{
    const int channels[1] = { 0 }; //ͨ������
    float inRanges[2] = { 0,255 };  //���ط�Χ
    const float* ranges[1] = {inRanges};//���ػҶȼ���Χ
    const int bins[1] = { 256 };   //ֱ��ͼ��ά��
    calcHist(&gray, 1, channels,mask, Hist,1, bins, ranges);
}

void getPro(Mat&img)
{
	float sum = 0;
	for(int i = 0;i < 256;++ i)	sum += img.at<float>(i);
	for(int i = 0;i < 256;++ i)	
	{
		img.at<float>(i) /= sum;
//		cout << i << ' '<< img.at<float>(i) << endl;
	}
}

void showHist(Mat&img,string name = "HistShow")
{
	int mx = 0;
	for(int i = 1;i < 256;++ i)	mx = max(mx,cvRound(img.at<float>(i)));
	for(int i = 1;i < 256;++ i)	img.at<float>(i) *= 400.0/mx;
	Mat a = Mat::zeros(400,512,CV_8UC3);
	for(int i = 1;i < 256;++ i)
		rectangle(a,Point(i * 2,399),Point((i+1) * 2,400 - cvRound(img.at<float>(i))),Scalar(255,255,255),-1);
	namedWindow(name,WINDOW_AUTOSIZE);
	imshow(name,a);
}
