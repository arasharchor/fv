
#include <iostream>
#include "cfeature.h"
#include "cextfeatDemo.h"
#include "wrap.h"

using namespace std;

void CExtfeatDemo::doit( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureStore *featStore )
{
	_do(imgWrapSrc1, imgWrapSrc2, featStore);
}

void CExtfeatDemo::_do( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureStore *featStore )
{
	imgWrapSrc1->context = NULL;
	imgWrapSrc2->context = NULL;
	featStore->featStore.push_back( vector<double>(1, 10) );
	featStore->numberPerFeatType.push_back( 1 );
}