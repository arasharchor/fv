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

static int TRAIN_NUM = 1000;			//ѵ��ͼ��Ը���

using namespace std;
using namespace cv;

int main(void)
{
	iofile coupleImgDataSet("datalist.txt");

	vector<CFeature> featureSet(TRAIN_NUM);	//������
	vector<float> labelSet(TRAIN_NUM);

	// 1).��ȡ����ͼ�������
	for(int i = 0; i < TRAIN_NUM; i++)
	{
		tuple<string, string> path;

		if( i < TRAIN_NUM / 2 )					// ��������ǰ TRAIN_NUM/2 ��
		{
			labelSet[i] = 1.0;
			coupleImgDataSet.extCoupleImg_path(path, i, true);						// ��i��������
		}
		else									// ��������ǰ TRAIN_NUM/2 ��
		{
			labelSet[i] = -1.0;
			coupleImgDataSet.extCoupleImg_path(path, i - TRAIN_NUM / 2, false);		// ��i�Ը�����
		}

		Mat img1 = imread(get<0>(path), IMREAD_GRAYSCALE);
		Mat img2 = imread(get<1>(path), IMREAD_GRAYSCALE);

//		printf("start %d\n", i);
		imshow("img1", img1);imshow("img2", img2);
		featureSet[i] =  CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
		imshow("after img1", img1);imshow("after img2", img2);
		printf("finish %d\n", i);

		waitKey(0);
	}

	//2).ѵ��ģ��
	CModelInt *model = new CModelSVM();
	model->train(featureSet, labelSet);
	model->validation_model(featureSet, labelSet);

//	model->saveModel("modelSvm");

	//3).��ȡģ��
//	model->loadModel("modelSvm");

	return 0;
}