#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Rect A,B,C;
bool s_flag;
void RecOnMouse(int event, int x, int y, int flags, void* param)  //evnet:鼠标事件类型 x,y:鼠标坐标 flags：鼠标哪个键
{
	Mat* im = reinterpret_cast<Mat*>(param);
	Mat C;
	(*im).copyTo(C); 
	if(EVENT_LBUTTONDOWN == event)
	{
		A.x = x;
		A.y = y;
		s_flag = 1;
	}
	if(EVENT_MOUSEMOVE == event && s_flag)
	{
		rectangle(C,Point(A.x,A.y),Point(x,y),Scalar(0,0,255),2);
		imshow("IMAGE",C);
	}
	if(EVENT_LBUTTONUP == event)
	{
		B.x = x;
		B.y = y;
		s_flag = 0;
		cout << "choose 'Y' to confirm this rectangle" << endl;
	}
}

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
