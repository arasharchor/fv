#ifndef __CEXT_FEAT_INT_H
#define __CEXT_FEAT_INT_H

class ImgWrap;
class CFeatureImg;

class CExtfeatInt
{
	/* ctor and de-ctor */
public:
	virtual ~CExtfeatInt(){}

	/* interface */
public:
	virtual void doit( const ImgWrap *imgWrapSrc, CFeatureImg *featImg )=0;

	/* var member */
public:
};

#endif
