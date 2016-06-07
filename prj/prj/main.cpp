#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>

#include "cfeature.h"
#include "cmodelDemo.h"
#include "wrap.h"

static int TRAIN_NUM = 10;			//ѵ��ͼ�����

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
		Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
		ImgWrap imgWrap(&img);					//��img��װ����
		featureSet[i] = CFeature( &imgWrap );

		/*finish:
		img�ռ��ͷţ�����imgWrap�е�ָ����ָ�ᱻ�ͷ�
		���˲���Ҫ���ڴ���ȫ������
		*/
	}
	//2).ѵ��ģ��
	CModelInt *model = new CModelDemo();
	model->train(featureSet);
	model->storeModel("modelDemo");

	//3).��ȡģ��
	model->readModel("modelDemo");

	return 0;
}
