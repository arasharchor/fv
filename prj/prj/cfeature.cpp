
#include <iostream>
#include "cfeature.h"
#include "cextfeatDemo.h"
#include "cpreprocessDemo.h"

CFeature::CFeature(ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2)
{
	CPreprocessInt *preprocess=new CPreprocessDemo();
	CExtfeatInt *extfeat=new CExtfeatDemo();

	//Ԥ����
	if(preprocess)
	{
		preprocess->doit(imgWrapSrc1);
		preprocess->doit(imgWrapSrc2);
	}

	//��ȡ����
	if(extfeat)
	{
		extfeat->doit(imgWrapSrc1, imgWrapSrc2, &this->mFeatureModel);
	}
}