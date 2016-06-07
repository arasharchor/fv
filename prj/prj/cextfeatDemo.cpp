
#include <iostream>
#include "cfeature.h"
#include "cextfeatDemo.h"
#include "wrap.h"

using namespace std;

void CExtfeatDemo::doit( const ImgWrap *imgWrapSrc, CFeatureStore *featStore )
{
	_do(imgWrapSrc, featStore);
}

void CExtfeatDemo::_do( const ImgWrap *imgWrapSrc, CFeatureStore *featStore )
{
	void *imgPtr = imgWrapSrc->context;
	featStore->featStore.push_back( vector<double>(1, 10) );
	featStore->numberPerFeatType.push_back( 1 );
}