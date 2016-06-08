
#include <iostream>
#include "cfeature.h"
#include "cextfeature.h"
#include "cpreprocessDemo.h"

CFeature::CFeature(ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2)
{
	CPreprocessInt *preprocess = new CPreprocessDemo();
	CExtfeatInt *extfeat = new CExtfeature();

	//预处理
	if(preprocess)
	{
		preprocess->doit(imgWrapSrc1);
		preprocess->doit(imgWrapSrc2);
	}

	//提取特征
	if(extfeat)
	{
		extfeat->doit(imgWrapSrc1, &this->mFeatureImgA);
		extfeat->doit(imgWrapSrc2, &this->mFeatureImgB);
		this->_mixfeature(&mFeatureImgA, &mFeatureImgB, &mFeatureMode);
	}
}
