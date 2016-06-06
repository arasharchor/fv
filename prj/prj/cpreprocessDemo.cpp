
#include <iostream>
#include "cpreprocessDemo.h"
#include "wrap.h"

void CPreprocessDemo::doit( ImgWrap *imgWrapSrc )
{
	_do(imgWrapSrc);
}

void CPreprocessDemo::_do( ImgWrap *imgWrapSrc)
{
	imgWrapSrc->context = NULL;
}