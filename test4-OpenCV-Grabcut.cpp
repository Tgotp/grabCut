#include<opencv2/opencv.hpp>
#include<iostream>
#include<ctime>
#include "onMouse.h"
#include "tools.h" 
#include "GMM.h"
#include "./maxflow-v3.01/graph.h"
#define id(i,j) ((i) * M + (j)) //(i-1)���� 
#define eps 1e-9

using namespace cv;
using namespace std;

const double inf = 1e6;

typedef Graph<double,double,double> GraphType; //S������,T������,������
double pow(double x) { return x * x; }
double pow(Vec3d x) { return (int)x[0] * x[0] + (int)x[1] * x[1] + (int)x[2] * x[2]; }

GraphType* build_Graph(Mat img)
{
	cout <<"maxnode: " << img.rows * img.cols <<endl;
	GraphType *g = new GraphType(img.rows * img.cols,img.rows * img.cols * 12); // estimated of nodes and edges
	return g;
}
GraphType *g;

int N,M;
void Grabcut(Mat&image)
{
	Mat showimg,fmat;
	image.copyTo(showimg);
	fmat = Mat::zeros(image.size(),CV_8UC3);
	
	cout << "������������'Y' " << endl;
	while(true)
	{
		setMouseCallback("IMAGE",RecOnMouse,reinterpret_cast<void*>(&showimg));
		
		char r_key = waitKey(0);
		if(r_key == 'y' || r_key == 'Y')
		{
			rectangle(fmat,Point(A.x,A.y),Point(B.x,B.y),Scalar(255,255,255),-1);
			break;
		} 
	} 
	namedWindow("Rectangle",WINDOW_NORMAL);
	imshow("Rectangle",fmat);
}

int main()
{
	Mat image;   //����һ����ͼ��image
	
	image = imread("E:\\c++\\Grabcut\\30.jpg");  //��ȡ�ļ����е�ͼ��

	//���ͼ���Ƿ���سɹ�
	if (image.empty())  //���image�������ݣ������� image.empty()���� ��
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	
	g = build_Graph(image);
	
	N = image.rows,M = image.cols;
	for(int i = 0;i < N * M; ++ i) g -> add_node(); 
	
	for(int i = 0;i < N; ++ i)
		for(int j = 0;j < M; ++ j)
		{	
			if(i != 0) 
			{
				double row = exp(- pow(image.at<Vec3b>(i,j) - image.at<Vec3b>(i - 1,j)));
//				cout << i << ' '<<j<< ' '<<row << endl;
				g -> add_edge(id(i,j),id(i - 1,j),row,row); 
			}
			
			if(j != 0) 
			{
				double col = exp(- pow(image.at<Vec3b>(i,j) - image.at<Vec3b>(i,j - 1)));
//				cout << i << ' '<<j<< ' '<<col << endl;
//				cout << image.at<Vec3b>(i,j)[0] << ' '<<image.at<Vec3b>(i,j)[1]<< ' '<<image.at<Vec3b>(i,j)[2] << endl;
//				cout << image.at<Vec3b>(i,j- 1)[0] << ' '<<image.at<Vec3b>(i,j- 1)[1]<< ' '<<image.at<Vec3b>(i,j- 1)[2] << endl;

				g -> add_edge(id(i,j),id(i,j - 1),col,col); 
			}
		} 
	
	namedWindow("IMAGE",WINDOW_NORMAL);  //������ʾ���ڣ��������д��룬Ҳ����ʾ��Ĭ�ϴ��ڴ�С���ܸı�
	imshow("IMAGE", image);  //�ڴ�����ʾͼ��
	
	mask_source = Mat::zeros(image.size(),CV_8UC3);  //CV_8U�Ҷ�ͼ,CV_8UC3��ɫͼ 
	mask_target = Mat::zeros(image.size(),CV_8UC3);
	
	while(true)
	{
		cout << "������������� R(�ü�)\\C(���)\\Q(�˳�)" << endl;
		char r_key;
		char ch[6] = {'r','R','c','C','Q','q'};
		do 
		{ 
			r_key = waitKey(0);
			bool op = 0;
			for(int i = 0;i < 6; ++ i)
				if(r_key == ch[i])
					op = 1;
			if(op) break;
			else cout << "�������������������룡" << endl;
		}while(1);
		if(r_key == 'R' || r_key =='r') 
		{
			double st = clock(); 
			Grabcut(image);
			cout << "cost time : " << (clock() - st) / CLOCKS_PER_SEC << endl; 
		}
		if(r_key == 'q' || r_key =='Q') break;
	 } 
	 
	/*
	
	Mat histSource,Source;
	image.copyTo(Source,mask_source);
	GetHist(Source,histSource,mask_source); 
//	showHist(histSource);
	getPro(histSource);
	
	Mat histTarget,Target;
	image.copyTo(Target,mask_target);
	GetHist(Target,histTarget,mask_target);
	getPro(histTarget);
	
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

	waitKey(0);
	*/
	
	return 0;
}

