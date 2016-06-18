#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>

#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

#include "iofile.h"

static int TRAIN_NUM = 3;			//ѵ��ͼ��Ը���

using namespace std;
using namespace cv;

int main(void)
{
	iofile coupleImgDataSet(40, 10);

	vector<CFeature> featureSet(TRAIN_NUM);	//������

	// 1).��ȡ����ͼ�������
	for(int i=0; i < TRAIN_NUM; i++)
	{
		string img1_path, img2_path;
		coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i, true);		// ��i��������
		//coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i, false);		// ��i�Ը�����

		Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
		Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);

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