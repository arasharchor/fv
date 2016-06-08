#ifndef __CEXT_FEAT_INT_H
#define __CEXT_FEAT_INT_H

class ImgWrap;
class CFeatureImg;

class CExtfeatInt
{
	/* interface */
public:
	CExtfeatInt();
	virtual void doit( const ImgWrap *imgWrapSrc, CFeatureImg *featImg )=0;

	/* var member */
public:
	int *utable;
};

#endif
