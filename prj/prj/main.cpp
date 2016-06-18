#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>

#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

static int TRAIN_NUM = 3;			//训练图像对个数

using namespace std;
using namespace cv;

int main(void)
{
	vector<CFeature> featureSet(TRAIN_NUM);	//特征集

	// 1).提取所有图像的特征
	for(int i=0; i < TRAIN_NUM; i++)
	{
		Mat img1 = imread("lena.jpg", IMREAD_GRAYSCALE);
		Mat img2 = imread("lena.jpg", IMREAD_GRAYSCALE);

		featureSet[i] = CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
	}

	//2).训练模型
	CModelInt *model = new CModelSVM();
	model->train(featureSet);
//	model->saveModel("modelSvm");

	//3).读取模型
//	model->loadModel("modelSvm");

	return 0;
}