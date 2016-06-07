
#include <iostream>
#include "cfeature.h"
#include "cextfeatDemo.h"
#include "cpreprocessDemo.h"

CFeature::CFeature(ImgWrap *imgWrapSrc)
{
	CPreprocessInt *preprocess=new CPreprocessDemo();
	CExtfeatInt *extfeat=new CExtfeatDemo();

	//预处理
	if(preprocess)
	{
//		preprocess->doit(imgWrapSrc);
	}

	//提取特征
	if(extfeat)
	{
		extfeat->doit(imgWrapSrc, &this->mFeatureStore);
	}
}