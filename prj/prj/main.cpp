#include <iostream>
#include <vector>

#include "cfeature.h"
#include "cmodelSVM.h"
#include "wrap.h"

static int TRAIN_NUM=10;			//ѵ��ͼ�����

using namespace std;

int main(void)
{
	vector<CFeature> featureSet(TRAIN_NUM);	//������

	//1).��ȡ����ͼ�������
	for(int i=0; i<TRAIN_NUM; i++)
	{
		int img = i;
		ImgWrap imgWrap(&img);					//��img1 img2��װ����

		featureSet[i] = CFeature( &imgWrap );

		/*finish:
		img�ռ��ͷţ�����imgWrap�е�ָ����ָ�ᱻ�ͷ�
		���˲���Ҫ���ڴ���ȫ������
		*/
	}

	//2).ѵ��ģ��
	CModelInt *model = new CModelSVM();
	model->train(featureSet);
//	model->saveModel("modelDemo");

	//3).��ȡģ��
//	model->loadModel("modelDemo");

	return 0;
}