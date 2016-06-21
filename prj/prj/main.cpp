#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>

#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

#include "iofile.h"

static int TRAIN_NUM = 4;			//训练图像对个数

using namespace std;
using namespace cv;

int main(void)
{
	iofile coupleImgDataSet(40, 10);

	vector<CFeature> featureSet(TRAIN_NUM);	//特征集
	vector<float> labelSet(TRAIN_NUM);

	// 1).提取所有图像的特征
	for(int i=0; i < TRAIN_NUM; i++)
	{
		string img1_path, img2_path;

		if( i < TRAIN_NUM/2 )
		{
			labelSet[i] = 1.0;
			coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i+1, true);		// 第i对正样本
		}
		else
		{
			labelSet[i] = -1.0;
			coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i-TRAIN_NUM/2+1, false);		// 第i对负样本
		}

		Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
		Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);

//		printf("start %d\n", i);
//		imshow("img1", img1);imshow("img2", img2);
		featureSet[i] = CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
//		imshow("after img1", img1);imshow("after img2", img2);
		printf("finish %d\n", i);

		waitKey(0);
	}

	//2).训练模型
	CModelInt *model = new CModelSVM();
	model->train(featureSet, labelSet);
	model->validation_model(featureSet, labelSet);

//	model->saveModel("modelSvm");

	//3).读取模型
//	model->loadModel("modelSvm");

	return 0;
}