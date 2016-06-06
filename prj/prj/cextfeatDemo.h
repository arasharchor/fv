#ifndef __CEXTFEAT_H
#define __CEXTFEAT_H

#include "cextfeat.h"

class CExtfeatDemo : public CExtfeat
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	void doit( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureModel *featModel ) override;

	/* member fun */
private:
	void _do( ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2, CFeatureModel *featModel );

	/* member var */
private:
};

#endif