#ifndef __CEXT_FEAT_H
#define __CEXT_FEAT_H

class ImgWrap;
class CFeatureModel;

class CExtfeat
{
	/* interface */
public:
	virtual void doit( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureModel *featModel)=0;
};

#endif