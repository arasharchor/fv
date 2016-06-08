#ifndef __CEXT_FEAT_INT_H
#define __CEXT_FEAT_INT_H

class ImgWrap;
class CFeatureStore;

class CExtfeatInt
{
	/* interface */
public:
	virtual void doit( const ImgWrap *imgWrapSrc, CFeatureImg *featImg )=0;
};

#endif