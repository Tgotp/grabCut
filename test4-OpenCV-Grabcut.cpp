#include<opencv2/opencv.hpp>
#include<iostream>
#include<ctime>
#include "onMouse.h"
#include "tools.h" 
#include "GMM.h"
#include "./maxflow-v3.01/graph.h"
#define id(i,j) ((i) * M + (j)) //(i-1)带入 
#define eps 1e-9

using namespace cv;
using namespace std;

const double inf = 1e6;

typedef Graph<double,double,double> GraphType; //S边流量,T边流量,边流量
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
	
	cout << "如果画完就输入'Y' " << endl;
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
	Mat image;   //创建一个空图像image
	
	image = imread("E:\\c++\\Grabcut\\30.jpg");  //读取文件夹中的图像

	//检测图像是否加载成功
	if (image.empty())  //检测image有无数据，无数据 image.empty()返回 真
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
	
	namedWindow("IMAGE",WINDOW_NORMAL);  //创建显示窗口，不加这行代码，也能显示，默认窗口大小不能改变
	imshow("IMAGE", image);  //在窗口显示图像
	
	mask_source = Mat::zeros(image.size(),CV_8UC3);  //CV_8U灰度图,CV_8UC3彩色图 
	mask_target = Mat::zeros(image.size(),CV_8UC3);
	
	while(true)
	{
		cout << "请输入操作类型 R(裁剪)\\C(标记)\\Q(退出)" << endl;
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
			else cout << "输入类型有误，重新输入！" << endl;
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

