#include<opencv2/opencv.hpp>
#include<iostream>
#include "onMouse.h"

using namespace cv;
using namespace std;

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

	namedWindow("IMAGE");  //������ʾ���ڣ��������д��룬Ҳ����ʾ��Ĭ�ϴ��ڴ�С���ܸı�
	imshow("IMAGE", image);  //�ڴ�����ʾͼ��
	
	while(true)
	{
		setMouseCallback("IMAGE",onMouse,reinterpret_cast<void*>(&image));
		
		char r_key = waitKey(20);
		if(r_key == 'y' || r_key == 'Y')
		{
			break;
		}  //��ͣ������ͼ����ʾ���ȴ���������
	}
	Mat imageROI(image,Rect(A.x,A.y,B.x,B.y)); 
	imshow("cutIMAGE",imageROI); 
	
	waitKey(0);
	
	return 0;

}
