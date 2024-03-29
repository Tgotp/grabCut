#include "GMM.h"
#include <opencv2/highgui/highgui_c.h>
#include <cstring>
//#pragma GCC optimize(3,"Ofast","inline")
//#pragma GCC optimize(2)

void ShowMat(Mat mat,string name = "undefine",int flag = 0)
{
	cout << name << endl;
	cout << "rows and cols:"<< mat.rows << ' ' << mat.cols << endl;
	
	if(flag&1)
	for (int i = 0;i<mat.rows;i++)  
	{  
		for (int j = 0;j<mat.cols ;j++)  
		{  
			printf("%lf%c",mat.at<double>(i,j)," \n"[j == (mat.cols-1)]);  
		}  
	}
	cout << endl;
}
void GMM :: init(Mat img,Mat mask,int clustNum,double eps_end,double eps,int max_steps)
{
//	this -> data_lst = Mat::zeros(img.size(),CV_8UC3);
	this -> clustNum = clustNum;
	this -> max_steps = max_steps;
	this -> eps = eps;
	this -> eps_end = eps_end;
	this -> logL_lst = 2;
	
	this -> N = img.rows;
	this -> M = img.cols; 
	
//	cout << N*M << endl;
	
	imgMat = Mat :: zeros(this -> N * this -> M,3,CV_64F);
	Mat mean = Mat :: zeros(1,3,CV_64F),sigma;
	n = 0;
//	ShowMat(imgMat,"before mask");
	for(int i = 0;i < this -> N; ++ i)
		for(int j = 0;j < this -> M; ++ j) if(mask.at<uchar>(i,j) != 0)
		{
			imgMat.at<double>(n,0) = img.at<Vec3b>(i,j)[0];
			imgMat.at<double>(n,1) = img.at<Vec3b>(i,j)[1];
			imgMat.at<double>(n,2) = img.at<Vec3b>(i,j)[2];
			++ n;
		}
	imgMat.resize(n);
//	ShowMat(imgMat,"after mask");
	calcCovarMatrix(imgMat,sigma,mean,CV_COVAR_NORMAL|CV_COVAR_ROWS,CV_64F);
	sigma /= n-1; //must!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
//	ShowMat(sigma,"cov",1);
//	ShowMat(mean,"mean",1);
	belong.resize(n);
	weight.resize(clustNum);
	prob.resize(clustNum);
//	prob_lst.resize(clustNum);
	sigmas.resize(clustNum);
	means.resize(clustNum);
	det.resize(clustNum);
	sigmas_inv.resize(clustNum);
	memberships.resize(n);
	for(int i = 0;i < n; ++ i)
		memberships[i].resize(clustNum);
	
	Mat sigma_inv;
	invert(sigma,sigma_inv);
//	ShowMat(sigma_inv,"InvCov",1);
	for(int i = 0;i < clustNum;++ i)
	{
		means[i] = Mat :: zeros(1,3,CV_64F); 
		weight[i] = 1.0 / clustNum;
		sigmas[i] = sigma;
		sigmas_inv[i] = sigma_inv;
		det[i] = sqrt(determinant(sigma));
		
		means[i].at<double>(0) = 255.0 * i / clustNum;
		means[i].at<double>(1) = 255.0 * i / clustNum;
		means[i].at<double>(2) = 255.0 * i / clustNum;
	}
}

bool flag;
void GMM :: train()
{
//	ShowMat(imgMat,"imgMat",1);
	int step = 0;
//	cout << clustNum << ' '<<n << endl;
	do
	{
		for(int i = 0;i < clustNum;++ i)
			prob[i] = 0;
		double logL = 0;
		//瓶颈待优化 
		double stgaus = clock();
//		double ys = 0; 
		for(int i = 0;i < n; ++ i)
		{
			double sum[6] = {};
			double X[3] = {imgMat.at<double>(i,0),imgMat.at<double>(i,1),imgMat.at<double>(i,2)};
			// 循环展开 
			sum[0] = weight[0] * gauss(0,X);  
			sum[1] = weight[1] * gauss(1,X);
			sum[2] = weight[2] * gauss(2,X);
			sum[3] = weight[3] * gauss(3,X);
			sum[4] = weight[4] * gauss(4,X);
			
			// p(i) i数据出现的概率
			sum[5] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4];
//			cout <<fixed << setprecision(16) <<" qwq:"<< sum[0] << ' '<< sum[1] << ' '<<sum[2]<< ' ' <<sum[3]<< ' ' <<sum[4] << endl;
//			cout <<fixed << setprecision(16) <<" qwq222:"<< sum[5] << endl;
//			if(sum[5] == 0) 
//			{
//				cout << step << endl;
//				flag = 1;
//				for(int j = 0;j < clustNum; ++j)
//					cout << gauss(j,X) << endl;
//				exit(0); 
//			}
			logL -= log(sum[5]);
			//p(i|k) i数据在第k类中的比例 
			memberships[i][0] = sum[0] / sum[5];
			memberships[i][1] = sum[1] / sum[5];
			memberships[i][2] = sum[2] / sum[5];
			memberships[i][3] = sum[3] / sum[5];
			memberships[i][4] = sum[4] / sum[5];
//			cout << i << ' '<< memberships[i][0] << endl;
			
			prob[0] += memberships[i][0];
			prob[1] += memberships[i][1];
			prob[2] += memberships[i][2];
			prob[3] += memberships[i][3];
			prob[4] += memberships[i][4];
			
//				if(i % 50000 == 0) 
//				{
//					cout<<fixed << setprecision(6) << "cost time : " << (clock() - stgaus) / CLOCKS_PER_SEC << endl; 
//					stgaus = clock();
//				}
			
//			}
		}
		
		// mean[k] = 1/Nk *sum(p[i][k]*x[i])
		for(int i = 0;i < clustNum;++ i) means[i] = Mat :: zeros(1,3,CV_64F); 
		for(int i = 0;i < n; ++ i)
		{
//			for(int j = 0;j < clustNum;++ j)
//				means[j] +=imgMat.row(i) * memberships[i][j];
			// 循环展开 
			means[0].at<double>(0) += imgMat.at<double>(i,0) * memberships[i][0];
			means[0].at<double>(1) += imgMat.at<double>(i,1) * memberships[i][0];
			means[0].at<double>(2) += imgMat.at<double>(i,2) * memberships[i][0];
			
			means[1].at<double>(0) += imgMat.at<double>(i,0) * memberships[i][1];
			means[1].at<double>(1) += imgMat.at<double>(i,1) * memberships[i][1];
			means[1].at<double>(2) += imgMat.at<double>(i,2) * memberships[i][1];
			
			means[2].at<double>(0) += imgMat.at<double>(i,0) * memberships[i][2];
			means[2].at<double>(1) += imgMat.at<double>(i,1) * memberships[i][2];
			means[2].at<double>(2) += imgMat.at<double>(i,2) * memberships[i][2];
			
			means[3].at<double>(0) += imgMat.at<double>(i,0) * memberships[i][3];
			means[3].at<double>(1) += imgMat.at<double>(i,1) * memberships[i][3];
			means[3].at<double>(2) += imgMat.at<double>(i,2) * memberships[i][3];
			
			means[4].at<double>(0) += imgMat.at<double>(i,0) * memberships[i][4];
			means[4].at<double>(1) += imgMat.at<double>(i,1) * memberships[i][4];
			means[4].at<double>(2) += imgMat.at<double>(i,2) * memberships[i][4];
			
		}
		for(int i = 0;i < clustNum;++ i)	
			means[i] /= prob[i];
		for(int j = 0;j < clustNum;++ j)
		{
			sigmas[j] = Mat :: zeros(3,3,CV_64F); 
			for(int i = 0;i < n;++ i)
			{
				double x1 = imgMat.at<double>(i,0) - means[j].at<double>(0);
				double x2 = imgMat.at<double>(i,1) - means[j].at<double>(0);
				double x3 = imgMat.at<double>(i,2) - means[j].at<double>(0);
				sigmas[j].at<double>(0) += memberships[i][j] * x1 * x1 + eps;
				sigmas[j].at<double>(1) += memberships[i][j] * x1 * x2;
				sigmas[j].at<double>(2) += memberships[i][j] * x1 * x3;
				sigmas[j].at<double>(3) += memberships[i][j] * x2 * x1;
				sigmas[j].at<double>(4) += memberships[i][j] * x2 * x2 + eps;
				sigmas[j].at<double>(5) += memberships[i][j] * x2 * x3;
				sigmas[j].at<double>(6) += memberships[i][j] * x3 * x1;
				sigmas[j].at<double>(7) += memberships[i][j] * x3 * x2;
				sigmas[j].at<double>(8) += memberships[i][j] * x3 * x3 + eps;
//				sigmas[j] += memberships[i][j] * (Mat_<double>(3,3) << x1*x1,x1*x2,x1*x3,x2*x1,x2*x2,x2*x3,x3*x1,x3*x2,x3*x3);
//				sigmas[j] += memberships[i][j] * (X.t() * X);
//				cout <<fixed << setprecision(2) << x1 << ' ' << x2 << ' ' << x3 << endl;	
//				ShowMat(X.t() * X,"X",1);
			}
			sigmas[j] /= prob[j];
		}
//		cout << "step: " << step << " logl1:" << logL << ' '<<logL/n<< endl; 
		
		if(abs(logL - logL_lst) / n < eps_end) break;
		logL_lst = logL;
//		ShowMat(sigmas[0],"sigmas",1);
		
		// sum should equal
		double p = 0; 
		for(int i = 0;i < clustNum;++ i)
			p += prob[i];
//		for(int i = 0;i < 5;++ i)
//			cout <<"prob "<< i << ':' << prob[i] << endl;
//		cout<< p << ' ' << n << ' ' <<logL<< endl; 
		
		for(int i = 0;i < clustNum;++ i)
		{
			weight[i] = prob[i] / p;
			invert(sigmas[i],sigmas_inv[i]);
			det[i] = sqrt(determinant(sigmas[i]));
//			cout << det[i] << endl;
		}
//		ShowMat(sigmas[0],"sigma",1);
//		ShowMat(sigmas_inv[0],"sigmaINv",1);
//		break;
	}while( (step ++) < max_steps);
}

double GMM :: pred(double X[])
{
	double p = 0; // p (c | F),F表示前景，举例。 
	for(int i = 0;i < clustNum;++ i)
		p += weight[i] * gauss(i,X);
//	cout << p << endl;
	if(p < 1e-9) return 1e9;
	return - log(p);
}

const double PREpdf = 15.749609946; //(2*pi)^(3/2) 

double GMM :: gauss(int m,double X[]) // m=类,x=样本 
{
	double x1 = X[0] - means[m].at<double>(0);
	double x2 = X[1] - means[m].at<double>(1);
	double x3 = X[2] - means[m].at<double>(2);
	double a1 = sigmas_inv[m].at<double>(0);
	double a2 = sigmas_inv[m].at<double>(1);
	double a3 = sigmas_inv[m].at<double>(2);
	double a4 = sigmas_inv[m].at<double>(3);
	double a5 = sigmas_inv[m].at<double>(4);
	double a6 = sigmas_inv[m].at<double>(5);
	double a7 = sigmas_inv[m].at<double>(6);
	double a8 = sigmas_inv[m].at<double>(7);
	double a9 = sigmas_inv[m].at<double>(8);
	
//	 cout << x1 << ' ' <<x2 <<" "<< x3 << ' ' <<det[m]<< endl;
//	if(flag) cout << "unknow"<<x1 * x1 * a1 + x1 * x2 * (a2 + a4) + x1 * x3 * (a3 + a7) + x2 * x2 * a5+ x2 * x3 * (a6+a8) + x3 * x3 * a9<<endl;
	return max(min(1e9,exp(-0.5* (x1 * x1 * a1 + x1 * x2 * (a2 + a4) + x1 * x3 * (a3 + a7) 
							+ x2 * x2 * a5 + x2 * x3 * (a6+a8) + x3 * x3 * a9)) 
								* 10000.0/det[m]
//								/ (PREpdf * det[m])  //* 100000
								),eps);
//	Mat ans = -0.5 * imgMat.row(x) * sigmas_inv[m] * imgMat.row(x).t();
//	return exp(ans.at<double>(0))/ (PREpdf * det[m]);
}
//double GMM :: gauss(Mat x,Mat inv,double det)
//{
//	//determinant求行列式|sigma|
//	double a = 1 / PREpdf * det;
////	Mat c = -1.0/2 * x * inv * x.t();
////	ShowMat(c,"Gaussion");
//	Mat result;
//	gemm(x, inv, -0.5, cv::Mat(), 0, result, 0);
//	gemm(result, inv, 1, cv::Mat(), 0, result, 0);
//	return a * exp(result.at<double>(0));
//}
