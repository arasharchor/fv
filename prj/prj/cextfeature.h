#ifndef __CEXTFEAT_H
#define __CEXTFEAT_H

#include <iostream>
#include <vector>

#include "cextfeatInt.h"

class CExtfeature : public CExtfeatInt
{
	/* ctor and de-ctor */
public:
	CExtfeature();

	/* interface */
public:
	void doit(const ImgWrap *imgWrapSrc, CFeatureImg *featImg) override;

	/* member fun */
//private:
	void _do(const ImgWrap *imgWrapSrc, CFeatureImg *featImg);

	void _cextlbp(const ImgWrap *imgWrapSrc, CFeatureImg *featImg);				// lbp
	void _cextlbp(const ImgWrap *imgWrapSrc, CFeatureImg *featImg, int scale);		// mb-lbp

	void _cextsift(const ImgWrap *imgWrapSrc, CFeatureImg *featImg);				// sift

	void _cextgabor(const ImgWrap *imgWrapSrc, CFeatureImg *featImg);				// gabor

	void _ccatgabor(CFeatureImg *featImg, int pooling);

	/* member var */
private:
	int *utable;
	int high;
	int width;

	enum{mean_pooling, max_pooling};
};

#endif
