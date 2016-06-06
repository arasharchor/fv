#include <iostream>
#include <vector>

#include "cfeature.h"				//����CFeature��, ֻ��һ����Ա����
#include "wrap.h"

static int N=100;					//ѵ��ͼ�����

using namespace std;

int main(void)
{
	vector<CFeature> featureSet(N);	//������

	//1).��ȡ����
	for(int i=0; i<N; i++)
	{
		int img1 = i, img2 = i+1;
		ImgWrap imgWrap1(&img1), imgWrap2(&img2);					//��img1 img2��װ����

		featureSet[i] = CFeature(new ImgWrap(), new ImgWrap());

		/*finish:
		img1 img2�ռ��ͷţ�����imgWrap1, imgWrap2�е�ָ����ָ�ᱻ�ͷ�
		imgWrap1, imgWrap2�Ŀռ��ͷ�
		���˲���Ҫ���ڴ���ȫ������
		*/
	}

	return 0;
}