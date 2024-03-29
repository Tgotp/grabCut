#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Rect A,B;
bool s_flag;
void RecOnMouse(int event, int x, int y, int flags, void* param)  //evnet:鼠标事件类型 x,y:鼠标坐标 flags：鼠标哪个键
{
	Mat* im = reinterpret_cast<Mat*>(param);
	Mat showimg;
	(*im).copyTo(showimg); 
	if(EVENT_LBUTTONDOWN == event)
	{
		A.x = x; A.y = y;
		s_flag = 1;
	}
	if(EVENT_MOUSEMOVE == event && s_flag)
	{
		rectangle(showimg,Point(A.x,A.y),Point(x,y),Scalar(0,0,255),2);
		imshow("IMAGE",showimg);
	}
	if(EVENT_LBUTTONUP == event)
	{
		B.x = x; B.y = y;
		s_flag = 0;
		cout << "choose 'Y' to confirm this rectangle" << endl;
	}
}

Rect C; 
bool lflag,rflag;
Mat mask_source,mask_target;
void LOnMouse(int event, int x, int y, int flags, void* param)  //evnet:鼠标事件类型 x,y:鼠标坐标 flags：鼠标哪个键
{
	Mat* im = reinterpret_cast<Mat*>(param);
	if(EVENT_LBUTTONDOWN == event || EVENT_RBUTTONDOWN == event)
	{
		C.x = x;C.y = y;
		if(EVENT_LBUTTONDOWN == event) lflag = 1;
		else rflag = 1;

	}
	
	if(EVENT_MOUSEMOVE == event && (lflag|rflag))
	{
		C.x = x;C.y = y;
		circle(*im,Point(x,y),6,lflag?Scalar(0,0,255):Scalar(255,0,0),-1);
		imshow("IMAGE",*im);
	}
	
	if(lflag) circle(mask_source,Point(x,y),6,255,-1);
	if(rflag) circle(mask_target,Point(x,y),6,255,-1);
		
	if(EVENT_LBUTTONUP == event || EVENT_RBUTTONUP == event)
	{
		C.x = x;C.y = y;
		if(EVENT_LBUTTONUP == event) lflag = 0;
		else rflag = 0;
	}
}
Mat Rec(Mat&showimg)
{
	cout << "如果画完就输入'Y' " << endl;
	Mat fmat = Mat :: zeros(showimg.rows,showimg.cols,CV_8UC1);
	while(true)
	{
		setMouseCallback("IMAGE",RecOnMouse,reinterpret_cast<void*>(&showimg));
		
		char r_key = waitKey(0);
		if(r_key == 'y' || r_key == 'Y')
		{
			if(A.x > B.x) swap(A.x,B.x);
			if(A.y > B.y) swap(A.y,B.y);
			cout <<"Point informain:" << A.x << ' ' << A.y << ' '<<B.x << ' '<< B.y << endl;
//			A.x = 29,A.y = 23,B.x = 467,B.y = 335;//debug
//			A.x = 29,A.y = 23,B.x = 200,B.y = 200;//debug
//			A.x = 157,A.y = 43,B.x = 303,B.y = 238;//debug
//			A.x = 144,A.y = 39,B.x = 309,B.y = 273; // carsten head
			rectangle(fmat,Point(A.x,A.y),Point(B.x,B.y),255,-1);
			break;
		}
	}
	return fmat;
} 
 
bool Cir(Mat&img)
{
	cout << "如果画完就输入'Y' 退出按Q" << endl;
	Mat showimg;
	img.copyTo(showimg); 
	while(true)
	{
		setMouseCallback("IMAGE",LOnMouse,reinterpret_cast<void*>(&showimg));
		
		char r_key = waitKey(0);
		if(r_key == 'y' || r_key == 'Y')
			return 1;
		if(r_key == 'q' || r_key == 'Q')
			return 0;
	}
}
