#include<opencv2/opencv.hpp>
#include<iostream>
#include<ctime>
#include "onMouse.h"
#include "tools.h" 
#include "./maxflow-v3.01/graph.h"
#define id(i,j) ((i) * M + (j)) //(i-1)���� 
#define eps 1e-9

using namespace cv;
using namespace std;

const double inf = 1e6;

typedef Graph<double,double,double> GraphType; //S������,T������,������
double pow(double x) { return x * x; }

GraphType* build_Graph(Mat img)
{
	cout <<"maxnode: " << img.rows * img.cols <<endl;
	GraphType *g = new GraphType(img.rows * img.cols,img.rows * img.cols * 12); // estimated of nodes and edges
	return g;
}

int main()
{
	Mat image;   //����һ����ͼ��image
	image = imread("E:\\c++\\Grabcut\\2.png",IMREAD_GRAYSCALE);  //��ȡ�ļ����е�ͼ�񣨻Ҷ�ͼ�� 
//	image = imread("E:\\c++\\Grabcut\\1.jpg");  //��ȡ�ļ����е�ͼ��

	//���ͼ���Ƿ���سɹ�
	if (image.empty())  //���image�������ݣ������� image.empty()���� ��
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	
	GraphType *g = build_Graph(image);
	
	Mat histImage;
	GetHist(image,histImage,Mat());
	
	// showHist(histImage);
	
	Mat showimg;
	showimg = Mat(showimg.size(),CV_8UC3);
	cvtColor(image,showimg,COLOR_GRAY2RGB);
	namedWindow("IMAGE",WINDOW_NORMAL);  //������ʾ���ڣ��������д��룬Ҳ����ʾ��Ĭ�ϴ��ڴ�С���ܸı�
	imshow("IMAGE", showimg);  //�ڴ�����ʾͼ��
	
	mask_source = Mat::zeros(image.size(),CV_8U);  //CV_8U�Ҷ�ͼ,CV_8UC3��ɫͼ 
	mask_target = Mat::zeros(image.size(),CV_8U);
	
	cout << "������������'Y' " << endl;
	while(true)
	{
		setMouseCallback("IMAGE",LOnMouse,reinterpret_cast<void*>(&showimg));
		
		char r_key = waitKey(20);
		if(r_key == 'y' || r_key == 'Y')
		{
			break;
		} 
	}
	
	double st = clock(); 
	
	Mat histSource,Source;
	image.copyTo(Source,mask_source);
	GetHist(Source,histSource,mask_source); 
//	showHist(histSource);
	getPro(histSource);
	
	Mat histTarget,Target;
	image.copyTo(Target,mask_target);
	GetHist(Target,histTarget,mask_target);
	getPro(histTarget);
	
	int N = image.rows,M = image.cols;
	
	for(int i = 0;i < N * M; ++ i) g -> add_node(); 
	for(int i = 0;i < N; ++ i)
		for(int j = 0;j < M; ++ j)
		{	
			//S -> node flow = -ln p(I_p|F) node -> T = -ln p(I_p|B)
			if(Source.at<uchar>(i,j) > 0) g -> add_tweights(id(i,j),inf,0);
			else if(Target.at<uchar>(i,j) > 0) g -> add_tweights(id(i,j),0,inf);
			else 
			{
				float a = histTarget.at<float>(image.at<uchar>(i,j)) ;
				float b = histSource.at<float>(image.at<uchar>(i,j)) ;
				if(a < eps) a = inf; else a = - log(a);
				if(b < eps) b = inf; else b = - log(b);
				
//				cout << "s -> node : " << a << " " << "node -> T : " << b << endl;
				g -> add_tweights(id(i,j),a,b);
			}
			
			if(i != 0) 
			{
				double row = exp(- pow(image.at<uchar>(i,j) - image.at<uchar>(i - 1,j)));
//				cout << i << ' '<<j<< ' '<<row << endl;
				g -> add_edge(id(i,j),id(i - 1,j),row,row); 
			}
			
			if(j != 0) 
			{
				double col = exp(- pow(image.at<uchar>(i,j) - image.at<uchar>(i,j - 1)));
//				cout << i << ' '<<j<< ' '<<col << endl;
				g -> add_edge(id(i,j),id(i,j - 1),col,col); 
			}
		} 
	double mxflow = g -> maxflow();
	cout << fixed << setprecision(10) << mxflow << endl;
	
	int cnt = 0;
	Source = Mat::zeros(image.size(),CV_8U);
	mask_source = Mat::zeros(image.size(),CV_8U);
	for(int i = 0;i < N; ++ i)
		for(int j = 0;j < M; ++ j)
			if(g->what_segment(id(i,j)) == GraphType::SOURCE) 
			{
				++ cnt;
				mask_source.at<uchar>(i,j) = 255;
			}
	image.copyTo(Source,mask_source);
	imshow("Source",Source);
	cout << "N M cnt "<< N << ' ' << M << ' '<< cnt << endl;
	cout << "cost time : " << (clock() - st) / CLOCKS_PER_SEC << endl; 

	waitKey(0);
	
	return 0;
}

