#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>

#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

#include "iofile.h"

static int TRAIN_NUM = 3;			//ѵ��ͼ��Ը���

using namespace std;
using namespace cv;


// ��intת����string 
string itos(int i)
{
	stringstream s;
	s << i;
	return s.str();
}

int main(void)
{
	iofile coupleImgDataSet(40, 10);

	vector<CFeature> featureSet(TRAIN_NUM);	//������

	// 1).��ȡ����ͼ�������
	for(int i = 1; i < TRAIN_NUM; i++)
	{
		string img1_path, img2_path;
		coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i, true);		// ��i��������
		//coupleImgDataSet.extCoupleImg_path(img1_path, img2_path, i, false);		// ��i�Ը�����

		//Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
		//Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);

		//ImgWrap imgWrap1(&img1), imgWrap2(&img2);

		//featureSet[i] = CFeature(&imgWrap1, &imgWrap2);

		/*finish:
		img�ռ��ͷţ�����imgWrap�е�ָ����ָ�ᱻ�ͷ�
		���˲���Ҫ���ڴ���ȫ������
		*/
	}
	//2).ѵ��ģ��
	CModelInt *model = new CModelSVM();
	model->train(featureSet);
//	model->saveModel("modelSvm");

	//3).��ȡģ��
//	model->loadModel("modelSvm");

	return 0;
}