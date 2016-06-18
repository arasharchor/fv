#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>

#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

static int TRAIN_NUM = 3;			//ѵ��ͼ��Ը���

using namespace std;
using namespace cv;

int main(void)
{
	vector<CFeature> featureSet(TRAIN_NUM);	//������

	// 1).��ȡ����ͼ�������
	for(int i=0; i < TRAIN_NUM; i++)
	{
		Mat img1 = imread("lena.jpg", IMREAD_GRAYSCALE);
		Mat img2 = imread("lena.jpg", IMREAD_GRAYSCALE);

		featureSet[i] = CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
	}

	//2).ѵ��ģ��
	CModelInt *model = new CModelSVM();
	model->train(featureSet);
//	model->saveModel("modelSvm");

	//3).��ȡģ��
//	model->loadModel("modelSvm");

	return 0;
}