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
	for(int i=2; i < TRAIN_NUM; i++)
	{
		string img1_path, img2_path;
		coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i+1, true);		// ��i��������
		//coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i, false);		// ��i�Ը�����

		Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
		Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);

		printf("start %d\n", i);
		imshow("img1", img1);imshow("img2", img2);
		featureSet[i] = CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
		imshow("after img1", img1);imshow("after img2", img2);
		printf("finish %d\n", i);

		waitKey(0);
	}

	//2).ѵ��ģ��
//	CModelInt *model = new CModelSVM();
//	model->train(featureSet);
//	model->saveModel("modelSvm");

	//3).��ȡģ��
//	model->loadModel("modelSvm");

	return 0;
}