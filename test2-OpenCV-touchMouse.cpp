#include<opencv2/opencv.hpp>
#include<iostream>
#include "onMouse.h"

using namespace cv;
using namespace std;

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

	namedWindow("IMAGE");  //创建显示窗口，不加这行代码，也能显示，默认窗口大小不能改变
	imshow("IMAGE", image);  //在窗口显示图像
	
	while(true)
	{
		setMouseCallback("IMAGE",onMouse,reinterpret_cast<void*>(&image));
		
		char r_key = waitKey(20);
		if(r_key == 'y' || r_key == 'Y')
		{
			break;
		}  //暂停，保持图像显示，等待按键结束
	}
	Mat imageROI(image,Rect(A.x,A.y,B.x,B.y)); 
	imshow("cutIMAGE",imageROI); 
	
	waitKey(0);
	
	return 0;

}
