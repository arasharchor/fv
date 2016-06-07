#ifndef __CEXTFEAT_H
#define __CEXTFEAT_H

#include "cextfeatInt.h"

class CExtfeatDemo : public CExtfeatInt
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	void doit( const ImgWrap *imgWrapSrc, CFeatureStore *featStore ) override;

	/* member fun */
private:
	void _do( const ImgWrap *imgWrapSrc, CFeatureStore *featStore );

	/* member var */
private:
};

#endif