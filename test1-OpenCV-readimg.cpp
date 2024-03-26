#include<opencv2/opencv.hpp>
#include<iostream>


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

	imwrite("1.png", image); //����ͼ��Ϊpng��ʽ���ļ�����Ϊ1

	waitKey(0);  //��ͣ������ͼ����ʾ���ȴ���������

	return 0;

}
