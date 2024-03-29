#include "TempGMM.h"
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
void GMM :: init(Mat img,Mat mask,int clustNum)
{
//	this -> data_lst = Mat::zeros(img.size(),CV_8UC3);
	this -> clustNum = clustNum;
	
	this -> N = img.rows;
	this -> M = img.cols; 
	
//	cout << N*M << endl;
	
	imgMat = Mat :: zeros(this -> N * this -> M,3,CV_64F);
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
}

void GMM :: train()
{
	model = EM::create();
    model->setClustersNumber(clustNum);
    model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
    model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
    model->trainEM(imgMat, noArray(), noArray() , probs); // logLikelihoods,labels,probs
//    ShowMat(probs,"PROBS",1);
}

double GMM :: pred(Mat&X)
{
	Mat cnt; double ans = 0;
//	cout << X.size() << endl;
	Vec2d res = model -> predict2(X,cnt);
//	printf("logLike :%.5lf IndexMax: %.5lf\n",res[0],res[1]);
//	ShowMat(res,"pred");
	Mat weights = model -> getWeights();
//	ShowMat(weights,"weights");
	for(int i = 0;i < clustNum; ++ i)
		ans += cnt.at<double>(i) * weights.at<double>(i);
//	cout <<fixed << setprecision(5) << ans << endl;
	return -log(ans);
}
