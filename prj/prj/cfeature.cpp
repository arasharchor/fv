
#include <iostream>
#include "cfeature.h"
#include "cextfeatDemo.h"
#include "cpreprocessDemo.h"

CFeature::CFeature(ImgWrap *imgWrapSrc)
{
	CPreprocessInt *preprocess=new CPreprocessDemo();
	CExtfeatInt *extfeat=new CExtfeatDemo();

	//Ԥ����
	if(preprocess)
	{
//		preprocess->doit(imgWrapSrc);
	}

	//��ȡ����
	if(extfeat)
	{
		extfeat->doit(imgWrapSrc, &this->mFeatureStore);
	}
}