#include <iostream>
#include <vector>

#include "cfeature.h"
#include "cmodelDemo.h"
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
		img1 img2�ռ��ͷţ�����imgWrap1, imgWrap2�е�ָ����ָ�ᱻ�ͷ�
		imgWrap1, imgWrap2�Ŀռ��ͷ�
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