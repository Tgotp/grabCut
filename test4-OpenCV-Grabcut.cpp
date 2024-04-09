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

int N,M;
GraphType* build_Graph(Mat img)
{
//	cout <<"maxnode: " << img.rows * img.cols <<endl;
	GraphType *g = new GraphType(img.rows * img.cols + 2,img.rows * img.cols * 20); // estimated of nodes and edges
	
	N = img.rows,M = img.cols;
	for(int i = 0;i < N * M; ++ i) g -> add_node(); 
	for(int i = 0;i < N; ++ i)
		for(int j = 0;j < M; ++ j)
		{	
			if(i != 0) 
			{
				double row = exp(- pow(img.at<Vec3b>(i,j) - img.at<Vec3b>(i - 1,j)));
//				cout << i << ' '<<j<< ' '<<row << endl;
				g -> add_edge(id(i,j),id(i - 1,j),row,row); 
			}
			
			if(j != 0) 
			{
				double col = exp(- pow(img.at<Vec3b>(i,j) - img.at<Vec3b>(i,j - 1)));
//				cout << i << ' '<<j<< ' '<<col << endl;
//				cout << image.at<Vec3b>(i,j)[0] << ' '<<image.at<Vec3b>(i,j)[1]<< ' '<<image.at<Vec3b>(i,j)[2] << endl;
//				cout << image.at<Vec3b>(i,j- 1)[0] << ' '<<image.at<Vec3b>(i,j- 1)[1]<< ' '<<image.at<Vec3b>(i,j- 1)[2] << endl;

				g -> add_edge(id(i,j),id(i,j - 1),col,col); 
			}
		}
	return g;
}

int lst_node,node;
Mat min_cut(Mat&img,Mat Fmat)
{
	Mat fimg(img,Rect(A.x,A.y,abs(B.x - A.x),abs(B.y - A.y))); //rect(x,y,w,h)
	
	GraphType *g = build_Graph(fimg);
	Mat bmat;
	Fmat.copyTo(bmat);
	bitwise_not(bmat,bmat);
	Mat fmat(Fmat,Rect(A.x,A.y,abs(B.x - A.x),abs(B.y - A.y)));
	
	GMM obj,bck;
	for(int i = 0;i < N; ++ i)
		for(int j = 0;j < M; ++ j)	
			if(mask_source.at<uchar>(i+A.y,j + A.x) > 0)
				fmat.at<uchar>(i,j) = 255;
			else if(mask_target.at<uchar>(i+A.y,j + A.x) > 0)
				bmat.at<uchar>(i+A.y,j + A.x) = 255;
	obj.init(fimg,fmat);
	bck.init(img,bmat);
	obj.train();
	bck.train();
//	cout << N << ' ' << M << endl;	
	
	for(int i = 0;i < N; ++ i)
		for(int j = 0;j < M; ++ j)	
		{
			double X[3] = {(double)fimg.at<Vec3b>(i,j)[0],(double)fimg.at<Vec3b>(i,j)[1],(double)fimg.at<Vec3b>(i,j)[2]};
			g -> add_tweights(id(i,j), bck.pred(X) + (mask_source.at<uchar>(i+A.y,j + A.x) > 0 ? 100000:0), obj.pred(X) + (mask_target.at<uchar>(i+A.y,j + A.x) > 0? 100000:0));
//			cout <<obj.pred(X) << ' '<<  bck.pred(X)<< endl;
		}
	double max_flow = g -> maxflow();
	cout << fixed << setprecision(6) << "MAX_flow: "<< max_flow << endl;
	
	node = 0;
	for(int i = 0;i < N; ++ i)
		for(int j = 0;j < M; ++ j)
			Fmat.at<uchar>(i + A.y,j + A.x) = g->what_segment(id(i,j)) == GraphType::SOURCE ? ++ node,255 : 0;
	
	delete g;
	return Fmat;
}

void Grabcut(Mat&image,Mat fmat)
{
	Mat showimg,bmat; //չʾͼ��ǰ�����룬�������� 
	image.copyTo(showimg);
	
	double st = clock(); 
	cout <<"start time :"<< st / CLOCKS_PER_SEC<< endl;
	
	int step = 0,maxsteps = 10;
	while(step ++ < maxsteps)
	{
		double ST = clock();
		fmat = min_cut(image,fmat);
		cout << "Grabcut Once time: " << (clock() - ST) / CLOCKS_PER_SEC << endl; 
		if(abs(node - lst_node) < abs(B.x - A.x) * abs(B.y - A.y) / 100) break;
		lst_node = node;
	}
		
	Mat res;
	image.copyTo(res,fmat);
	Mat fimg(res,Rect(A.x,A.y,abs(B.x - A.x),abs(B.y - A.y))); //rect(x,y,w,h
	namedWindow("Out",WINDOW_NORMAL);
	imshow("Out", fimg);
//	imwrite("E:\\c++\\Grabcut\\image\\out.jpg",out);
	
	cout << "cost time : " << (clock() - st) / CLOCKS_PER_SEC << endl; 
}


int main()
{
//	double a = sqrt(2*acos(-1));
//	double b = a * a * a;
//	cout <<fixed << setprecision(9)<< b << endl;
	Mat image;   //����һ����ͼ��image
	
	image = imread("E:\\c++\\Grabcut\\image\\right\\flowers.png");  //��ȡ�ļ����е�ͼ��

	//���ͼ���Ƿ���سɹ�
	if (image.empty())  //���image�������ݣ������� image.empty()���� ��
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	 
	namedWindow("IMAGE",WINDOW_NORMAL);  //������ʾ���ڣ��������д��룬Ҳ����ʾ��Ĭ�ϴ��ڴ�С���ܸı�
	imshow("IMAGE", image);  //�ڴ�����ʾͼ��
	
	mask_source = Mat::zeros(image.size(),CV_8U);  //CV_8U�Ҷ�ͼ,CV_8UC3��ɫͼ 
	mask_target = Mat::zeros(image.size(),CV_8U);
	
	cout << "������ü�����" << endl;
	Mat fmat = Rec(image);
	Grabcut(image,fmat);
	
	while(Cir(image)) Grabcut(image,fmat);
	cout << "exit normal" << endl;
	return 0;
}

