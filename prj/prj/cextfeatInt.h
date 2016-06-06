#ifndef __CEXT_FEAT_INT_H
#define __CEXT_FEAT_INT_H

class ImgWrap;
class CFeatureStore;

class CExtfeatInt
{
	/* interface */
public:
	virtual void doit( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureStore *featStore)=0;
};

#endif