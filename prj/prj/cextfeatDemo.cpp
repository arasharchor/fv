
#include <iostream>
#include "cfeature.h"
#include "cextfeatDemo.h"
#include "wrap.h"

void CExtfeatDemo::doit( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureModel *featModel )
{
	_do(imgWrapSrc1, imgWrapSrc2, featModel);
}

void CExtfeatDemo::_do( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureModel *featModel )
{
	imgWrapSrc1->context = NULL;
	imgWrapSrc2->context = NULL;
	featModel->context = NULL;
}