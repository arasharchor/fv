#ifndef __CPREPROCESS_DEMO_H
#define __CPREPROCESS_DEMO_H

#include "cpreprocessInt.h"

class CPreprocessDemo : public CPreprocessInt
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	void doit( ImgWrap *imgWrapSrc ) override;

	/* member fun */
private:
	void _do( ImgWrap *imgWrapSrc );

	/* member var */
private:
};

#endif