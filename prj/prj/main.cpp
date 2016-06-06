#include <iostream>
#include <vector>

#include "cfeature.h"				//特征CFeature类, 只有一个成员变量
#include "wrap.h"

static int N=100;					//训练图像个数

using namespace std;

int main(void)
{
	vector<CFeature> featureSet(N);	//特征集

	//1).提取特征
	for(int i=0; i<N; i++)
	{
		int img1 = i, img2 = i+1;
		ImgWrap imgWrap1(&img1), imgWrap2(&img2);					//将img1 img2包装起来

		featureSet[i] = CFeature(new ImgWrap(), new ImgWrap());

		/*finish:
		img1 img2空间释放，所以imgWrap1, imgWrap2中的指针所指会被释放
		imgWrap1, imgWrap2的空间释放
		至此不需要的内存完全被回收
		*/
	}

	return 0;
}