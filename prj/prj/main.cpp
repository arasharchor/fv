#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>

#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

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
	vector<CFeature> featureSet(TRAIN_NUM);	//������

	// 1).��ȡ����ͼ�������
	for(int i = 0; i < TRAIN_NUM; i++)
	{
		Mat img1 = imread("lena.jpg", IMREAD_GRAYSCALE);
		Mat img2 = imread("lena.jpg", IMREAD_GRAYSCALE);
		ImgWrap imgWrap1(&img1), imgWrap2(&img2);

		featureSet[i] = CFeature(&imgWrap1, &imgWrap2);

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